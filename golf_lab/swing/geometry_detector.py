"""First phase detector: address and top of backswing, from hand motion.

Both phases are found from the path of the hands, because that is the one
signal a single 2D camera reads reliably from either angle. Shoulder and hip
rotation are more meaningful biomechanically but project very differently
face-on versus down-the-line, so a detector built on them would need to be two
detectors.

The method, in order:

1. **Address** is the end of the initial quiet period. Before a swing starts
   the hands barely move; the first sustained movement is the takeaway. Taking
   the *last* quiet frame rather than the first means a long waggle before the
   swing does not drag the address position backwards.

2. **Peak speed** is the fastest hand movement in the clip. On any real swing
   that is the downswing, which is several times faster than the backswing.
   It is used only as a landmark to bound the search.

3. **Top of backswing** is the highest hand position between address and peak
   speed. Bounding it that way is what prevents the finish — where the hands
   are usually just as high — from being mistaken for the top.

Only the two phases are reported. The others are deliberately not guessed at.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from golf_lab.logging_config import get_logger
from golf_lab.models.video import CameraView
from golf_lab.pose import landmarks as lm
from golf_lab.pose.sequence import PoseSequence
from golf_lab.swing.phases import (
    PHASE_SCHEMA_VERSION,
    PhaseResult,
    SwingPhase,
    SwingPhases,
)
from golf_lab.swing.results import ResultStatus

logger = get_logger(__name__)

DETECTOR_NAME = "hand-path-geometry"
DETECTOR_VERSION = "2"

# A swing needs enough frames to have a shape at all. Below this, whatever the
# maxima happen to be is noise.
MIN_FRAMES = 12
# Fraction of detected frames required before phases are attempted.
MIN_DETECTION_RATE = 0.5
# Hands must be visible this well on a frame for it to anchor a phase.
MIN_HAND_VISIBILITY = 0.3
# Movement below this fraction of peak swing speed counts as "still".
QUIET_SPEED_FRACTION = 0.08
# Landmark jitter sets a noise floor even when the golfer is motionless, so the
# quiet threshold is never allowed below this multiple of it.
NOISE_FLOOR_MULTIPLE = 3.0
# ...but the noise estimate is only trustworthy when part of the clip is
# actually still. On a clip that is moving from frame one, a percentile of the
# speeds measures the swing, not the noise, and would raise the threshold above
# the backswing itself. Capping its contribution keeps that failure bounded.
NOISE_CAP_FRACTION = 0.15
# The takeaway is sustained motion, not one noisy frame. Requiring several
# consecutive moving frames is what keeps landmark jitter from ending the
# address period early.
SUSTAINED_MOTION_FRAMES = 3
# A genuine address is held for a moment. Accepting a one- or two-frame lull —
# which path smoothing can manufacture at the very start of a clip — would
# report an address on footage that begins mid-swing.
MIN_ADDRESS_QUIET_FRAMES = 4
# Frames averaged when smoothing the hand path before differentiating. Raw
# per-frame differences of noisy landmarks are dominated by jitter.
PATH_SMOOTHING_WINDOW = 3
# The hands must rise at least this much (in shoulder widths) between address
# and the candidate top, or the clip probably does not contain a backswing.
MIN_BACKSWING_RISE = 0.35
# The finish is the mirror of address: sustained quiet after the swing. The
# golfer holds a finish position for a moment, so the same "several consecutive
# frames" test applies rather than accepting one slow frame.
SUSTAINED_QUIET_FRAMES = 3
# Half-width of the impact region, in frames at 30 fps, scaled with frame rate.
# Impact is reported as a REGION and never as a frame: without clubhead
# tracking the hands only tell us roughly when they returned to ball height,
# and at 30 fps the clubhead crosses the ball in far less than one frame.
IMPACT_HALF_WIDTH_AT_30FPS = 2
# The hands must come back down to within this many shoulder widths of their
# address height for a pass to count as an impact region.
MAX_IMPACT_HEIGHT_ERROR = 0.55


class HandPathPhaseDetector:
    """Locates the swing phases from the hand path.

    Camera-view agnostic by design — it uses hand height and speed, which read
    similarly face-on and down-the-line. It still receives the camera view
    because the protocol passes it and future detectors will need it.

    Phases are found in dependency order, and each one that cannot be found
    stops its dependents with a reason naming what was missing rather than
    substituting a guess:

        address ──► takeaway ──► top ──► downswing ──► impact region
                                                   └──► follow-through ──► finish
    """

    name = DETECTOR_NAME
    version = DETECTOR_VERSION
    supported_phases: Sequence[SwingPhase] = (
        SwingPhase.ADDRESS,
        SwingPhase.TAKEAWAY,
        SwingPhase.TOP_OF_BACKSWING,
        SwingPhase.DOWNSWING,
        SwingPhase.IMPACT_REGION,
        SwingPhase.FOLLOW_THROUGH,
        SwingPhase.FINISH,
    )

    def detect(
        self,
        pose_sequence: PoseSequence,
        camera_view: CameraView,
        timeline_is_approximate: bool = False,
    ) -> SwingPhases:
        phases = SwingPhases(
            detector_name=self.name,
            detector_version=self.version,
            schema_version=PHASE_SCHEMA_VERSION,
            camera_view=camera_view,
            frame_count=pose_sequence.frame_count,
            preview_fps=pose_sequence.fps,
            timeline_is_approximate=timeline_is_approximate,
        )
        if timeline_is_approximate:
            phases.notes.append(
                "This clip's preview timeline is resampled, so phase frames are "
                "preview frame numbers and cannot be converted to exact times in "
                "the original file. See docs/KNOWN_ISSUES.md (GSL-1)."
            )

        blocker = self._precheck(pose_sequence)
        if blocker is not None:
            status, reason = blocker
            for phase in self.supported_phases:
                phases.set(PhaseResult.unavailable(phase, status, reason))
            return phases

        hands, usable = self._hand_path(pose_sequence)
        hands = self._smooth_path(hands, usable)
        speed = self._speed(hands, usable)

        address = self._detect_address(hands, usable, speed)
        if address is None:
            reason = (
                "No settled address position was found — the hands are moving "
                "from the first detected frame. Start the clip a moment earlier."
            )
            phases.set(
                PhaseResult.unavailable(
                    SwingPhase.ADDRESS, ResultStatus.DETECTION_FAILED, reason
                )
            )
            self._block_dependents(
                phases,
                (
                    SwingPhase.TAKEAWAY,
                    SwingPhase.TOP_OF_BACKSWING,
                    SwingPhase.DOWNSWING,
                    SwingPhase.IMPACT_REGION,
                    SwingPhase.FOLLOW_THROUGH,
                    SwingPhase.FINISH,
                ),
                "address",
            )
            return phases

        address_frame, address_confidence, motion_start = address
        phases.set(
            PhaseResult.found(
                SwingPhase.ADDRESS,
                frame=address_frame,
                confidence=address_confidence,
                detail={"method": "end of initial low-motion period"},
            )
        )

        # --- takeaway: where sustained motion begins --------------------
        phases.set(
            PhaseResult.found(
                SwingPhase.TAKEAWAY,
                frame=motion_start,
                confidence=address_confidence,
                detail={
                    "method": "first frame of sustained hand motion after address",
                },
            )
        )

        top = self._detect_top(pose_sequence, hands, usable, speed, address_frame)
        if top is None:
            phases.set(
                PhaseResult.unavailable(
                    SwingPhase.TOP_OF_BACKSWING,
                    ResultStatus.DETECTION_FAILED,
                    "No backswing was found after address — the hands never rise "
                    "far enough above their address height. The clip may be "
                    "truncated, or may not contain a full swing.",
                )
            )
            self._block_dependents(
                phases,
                (
                    SwingPhase.DOWNSWING,
                    SwingPhase.IMPACT_REGION,
                    SwingPhase.FOLLOW_THROUGH,
                    SwingPhase.FINISH,
                ),
                "the top of the backswing",
            )
            return phases

        top_frame, top_confidence, detail = top
        phases.set(
            PhaseResult.found(
                SwingPhase.TOP_OF_BACKSWING,
                frame=top_frame,
                confidence=top_confidence,
                detail=detail,
            )
        )

        # --- impact region: hands back to address height ----------------
        impact = self._detect_impact_region(
            pose_sequence, hands, usable, address_frame, top_frame
        )
        if impact is None:
            phases.set(
                PhaseResult.unavailable(
                    SwingPhase.IMPACT_REGION,
                    ResultStatus.DETECTION_FAILED,
                    "The hands never returned close to their address height after "
                    "the top, so there is no evidence of a downswing through the "
                    "ball. The clip may end before impact.",
                )
            )
            self._block_dependents(
                phases,
                (SwingPhase.DOWNSWING, SwingPhase.FOLLOW_THROUGH, SwingPhase.FINISH),
                "the impact region",
            )
            return phases

        impact_start, impact_end, impact_confidence, impact_detail = impact
        phases.set(
            PhaseResult.found(
                SwingPhase.IMPACT_REGION,
                frame=impact_start,
                end_frame=impact_end,
                confidence=impact_confidence,
                detail=impact_detail,
            )
        )

        # --- downswing: between the top and impact ----------------------
        if impact_start - 1 >= top_frame + 1:
            phases.set(
                PhaseResult.found(
                    SwingPhase.DOWNSWING,
                    frame=top_frame + 1,
                    end_frame=impact_start - 1,
                    confidence=min(top_confidence, impact_confidence),
                    detail={"method": "frames between the top and the impact region"},
                )
            )
        else:
            phases.set(
                PhaseResult.unavailable(
                    SwingPhase.DOWNSWING,
                    ResultStatus.INSUFFICIENT_FRAMES,
                    f"The top (frame {top_frame}) and the impact region (frame "
                    f"{impact_start}) are adjacent, leaving no frames between them. "
                    "A higher frame rate is needed to resolve the downswing.",
                )
            )

        # --- finish: sustained quiet after impact -----------------------
        finish = self._detect_finish(hands, usable, speed, impact_end)
        if finish is None:
            phases.set(
                PhaseResult.unavailable(
                    SwingPhase.FINISH,
                    ResultStatus.DETECTION_FAILED,
                    "The hands are still moving when the clip ends, so no held "
                    "finish position was found. Record a moment longer after the "
                    "ball is struck.",
                )
            )
            phases.set(
                PhaseResult.unavailable(
                    SwingPhase.FOLLOW_THROUGH,
                    ResultStatus.INSUFFICIENT_FRAMES,
                    "The follow-through runs from impact to the finish, which was "
                    "not found.",
                )
            )
            return phases

        finish_frame, finish_confidence = finish
        phases.set(
            PhaseResult.found(
                SwingPhase.FINISH,
                frame=finish_frame,
                confidence=finish_confidence,
                detail={"method": "first sustained low-motion frame after impact"},
            )
        )

        # --- follow-through: between impact and the finish --------------
        if finish_frame - 1 >= impact_end + 1:
            phases.set(
                PhaseResult.found(
                    SwingPhase.FOLLOW_THROUGH,
                    frame=impact_end + 1,
                    end_frame=finish_frame - 1,
                    confidence=min(impact_confidence, finish_confidence),
                    detail={
                        "method": "frames between the impact region and the finish"
                    },
                )
            )
        else:
            phases.set(
                PhaseResult.unavailable(
                    SwingPhase.FOLLOW_THROUGH,
                    ResultStatus.INSUFFICIENT_FRAMES,
                    f"The impact region ends at frame {impact_end} and the finish "
                    f"is at frame {finish_frame}, leaving no frames between them.",
                )
            )

        return phases

    @staticmethod
    def _block_dependents(
        phases: SwingPhases, dependents: Sequence[SwingPhase], missing: str
    ) -> None:
        """Mark phases unavailable because something they depend on is missing.

        Reported as INSUFFICIENT_FRAMES rather than DETECTION_FAILED: nothing
        was attempted and failed here — the search never ran, and saying so
        keeps a genuine detection failure distinguishable from a cascade.
        """
        for phase in dependents:
            phases.set(
                PhaseResult.unavailable(
                    phase,
                    ResultStatus.INSUFFICIENT_FRAMES,
                    f"Not searched for, because {missing} was not found.",
                )
            )

    # -- checks ----------------------------------------------------------
    def _precheck(
        self, sequence: PoseSequence
    ) -> Optional[Tuple[ResultStatus, str]]:
        if sequence.frame_count < MIN_FRAMES:
            return (
                ResultStatus.INSUFFICIENT_FRAMES,
                f"The clip has {sequence.frame_count} frames; at least "
                f"{MIN_FRAMES} are needed to locate a swing.",
            )
        if sequence.detected_count < MIN_FRAMES:
            return (
                ResultStatus.INSUFFICIENT_FRAMES,
                f"Only {sequence.detected_count} frames have a detected pose; "
                f"at least {MIN_FRAMES} are needed.",
            )
        if sequence.detection_rate < MIN_DETECTION_RATE:
            return (
                ResultStatus.INSUFFICIENT_FRAMES,
                f"Pose was detected on only {sequence.detection_rate * 100:.0f}% "
                "of frames. Phase detection needs a mostly continuous swing.",
            )

        wrists = (lm.LEFT_WRIST, lm.RIGHT_WRIST)
        detected = np.asarray(sequence.detected)
        best = float(np.max(sequence.visibility[detected][:, list(wrists)]))
        if best < MIN_HAND_VISIBILITY:
            return (
                ResultStatus.MISSING_LANDMARKS,
                "The hands were never located with usable confidence, and both "
                "phases are found from the hand path.",
            )
        return None

    # -- signals ---------------------------------------------------------
    def _hand_path(self, sequence: PoseSequence) -> Tuple[np.ndarray, np.ndarray]:
        """Mid-hand position per frame, and which frames are usable.

        Uses whichever wrist is better seen on each frame rather than always
        averaging both: on a down-the-line view the trail wrist is occluded for
        much of the swing, and averaging in an unseen landmark would drag the
        path toward a position nobody observed.
        """
        count = sequence.frame_count
        path = np.full((count, 2), np.nan, dtype=np.float64)
        usable = np.zeros(count, dtype=bool)

        for index in range(count):
            if not sequence.detected[index]:
                continue
            left_vis = float(sequence.visibility[index, lm.LEFT_WRIST])
            right_vis = float(sequence.visibility[index, lm.RIGHT_WRIST])
            if max(left_vis, right_vis) < MIN_HAND_VISIBILITY:
                continue

            if min(left_vis, right_vis) >= MIN_HAND_VISIBILITY:
                point = (
                    sequence.landmarks[index, lm.LEFT_WRIST, :2]
                    + sequence.landmarks[index, lm.RIGHT_WRIST, :2]
                ) / 2.0
            else:
                better = lm.LEFT_WRIST if left_vis >= right_vis else lm.RIGHT_WRIST
                point = sequence.landmarks[index, better, :2]

            path[index] = point
            usable[index] = True

        return path, usable

    @staticmethod
    def _smooth_path(path: np.ndarray, usable: np.ndarray) -> np.ndarray:
        """Lightly average the hand path over usable frames.

        Differentiating raw landmarks measures jitter as much as motion, which
        made a motionless address look like it was already moving. Averaging is
        restricted to a window of usable frames so a gap is never bridged with
        invented positions.
        """
        smoothed = path.copy()
        half = PATH_SMOOTHING_WINDOW // 2
        for index in range(len(path)):
            if not usable[index]:
                continue
            low = max(0, index - half)
            high = min(len(path), index + half + 1)
            window = [j for j in range(low, high) if usable[j]]
            if window:
                smoothed[index] = np.mean(path[window], axis=0)
        return smoothed

    @staticmethod
    def _speed(path: np.ndarray, usable: np.ndarray) -> np.ndarray:
        """Per-frame hand speed, NaN where it cannot be measured.

        Gaps are left as NaN rather than bridged: a speed computed across a
        gap would be an artefact of the gap length, not of the swing.
        """
        speed = np.full(len(path), np.nan, dtype=np.float64)
        for index in range(1, len(path)):
            if usable[index] and usable[index - 1]:
                speed[index] = float(np.linalg.norm(path[index] - path[index - 1]))
        return speed

    # -- phases ----------------------------------------------------------
    def _detect_address(
        self, path: np.ndarray, usable: np.ndarray, speed: np.ndarray
    ) -> Optional[Tuple[int, float, int]]:
        """Last frame of the initial quiet period.

        Returns ``(address_frame, confidence, motion_start)``. ``motion_start``
        is the first frame of sustained motion, which is the takeaway — the two
        are the same measurement read from either side, so deriving both here
        keeps them from ever disagreeing.
        """
        finite = speed[np.isfinite(speed)]
        if finite.size == 0:
            return None

        peak = float(np.max(finite))
        if peak <= 1e-9:
            return None

        # The quiet threshold has to clear the landmark noise floor, otherwise
        # jitter on a motionless golfer reads as the takeaway. The 10th
        # percentile estimates that floor, capped relative to peak speed so a
        # clip with no still period cannot push the threshold above its own
        # backswing.
        noise_floor = float(np.percentile(finite, 10))
        threshold = max(
            peak * QUIET_SPEED_FRACTION,
            min(noise_floor * NOISE_FLOOR_MULTIPLE, peak * NOISE_CAP_FRACTION),
        )

        first_usable = int(np.argmax(usable)) if usable.any() else 0

        # Find where motion *starts*: the first run of consecutive frames over
        # the threshold. One frame over is noise; several in a row is the
        # takeaway. Address is the last still frame before that run.
        #
        # Deliberately the start of the first sustained run, not "the last
        # quiet frame anywhere" — the hands also slow down at the top of the
        # backswing, and the looser rule reported that as address.
        motion_start = None
        run_start = None
        run = 0
        for index in range(first_usable + 1, len(speed)):
            if not np.isfinite(speed[index]):
                continue
            if speed[index] > threshold:
                if run == 0:
                    run_start = index
                run += 1
                if run >= SUSTAINED_MOTION_FRAMES:
                    motion_start = run_start
                    break
            else:
                run = 0
                run_start = None

        if motion_start is None:
            # Nothing ever moved convincingly — not a swing.
            return None

        quiet = [
            index
            for index in range(first_usable, motion_start)
            if usable[index]
        ]
        if len(quiet) < MIN_ADDRESS_QUIET_FRAMES:
            # The hands are already moving when the clip starts.
            return None
        address = quiet[-1]

        # Confidence reflects how clearly the quiet period stands apart from
        # the swing that follows: a still address against a fast swing is
        # unambiguous, a drifting one is not.
        quiet_span = len(quiet)
        separation = threshold / peak if peak > 0 else 0.0
        confidence = float(
            np.clip(0.35 + 0.5 * min(quiet_span / 8.0, 1.0) + 0.15 * (1 - separation), 0, 1)
        )
        return address, confidence, motion_start

    def _detect_top(
        self,
        sequence: PoseSequence,
        path: np.ndarray,
        usable: np.ndarray,
        speed: np.ndarray,
        address_frame: int,
    ) -> Optional[Tuple[int, float, Dict[str, object]]]:
        """Highest hand position between address and peak hand speed."""
        after = np.arange(len(speed)) > address_frame
        candidates = np.isfinite(speed) & after
        if not candidates.any():
            return None

        peak_speed_frame = int(np.nanargmax(np.where(candidates, speed, np.nan)))

        window = np.arange(len(path))
        in_window = (window > address_frame) & (window <= peak_speed_frame) & usable
        if not in_window.any():
            return None

        # Image y grows downward, so the highest hands are the minimum y.
        heights = np.where(in_window, path[:, 1], np.inf)
        top_frame = int(np.argmin(heights))

        rise = float(path[address_frame, 1] - path[top_frame, 1])
        scale = self._shoulder_width(sequence, address_frame)
        rise_in_widths = rise / scale if scale > 1e-6 else 0.0
        if rise_in_widths < MIN_BACKSWING_RISE:
            return None

        # A top is convincing when the hands are clearly higher than at address
        # and the apex is well inside the window rather than pinned to its edge.
        margin = min(rise_in_widths / (MIN_BACKSWING_RISE * 3.0), 1.0)
        interior = 1.0 if address_frame < top_frame < peak_speed_frame else 0.5
        hand_visibility = float(
            np.max(sequence.visibility[top_frame, [lm.LEFT_WRIST, lm.RIGHT_WRIST]])
        )
        confidence = float(np.clip(0.3 + 0.4 * margin + 0.3 * interior * hand_visibility, 0, 1))

        detail: Dict[str, object] = {
            "method": "highest hands between address and peak hand speed",
            "peak_speed_frame": peak_speed_frame,
            "rise_shoulder_widths": round(rise_in_widths, 3),
        }
        return top_frame, confidence, detail

    def _detect_impact_region(
        self,
        sequence: PoseSequence,
        path: np.ndarray,
        usable: np.ndarray,
        address_frame: int,
        top_frame: int,
    ) -> Optional[Tuple[int, int, float, Dict[str, object]]]:
        """A frame *range* around the hands returning to address height.

        Reported as a region on principle, not as a hedge. Without clubhead
        tracking the hands only indicate roughly when the club came back
        through the ball, and at 30 fps the clubhead crosses the ball in well
        under a single frame — so a specific "impact frame" would be a
        precision this input cannot support. The half-width scales with frame
        rate, because a 240 fps clip genuinely localises it better.
        """
        after_top = np.arange(len(path)) > top_frame
        candidates = after_top & usable
        if not candidates.any():
            return None

        scale = self._shoulder_width(sequence, address_frame)
        if scale <= 1e-6:
            return None

        address_height = path[address_frame, 1]
        # Descent back toward the ball: the frame whose hand height is closest
        # to where it started.
        error = np.where(candidates, np.abs(path[:, 1] - address_height), np.inf)
        centre = int(np.argmin(error))
        best_error = float(error[centre] / scale)
        if not np.isfinite(best_error) or best_error > MAX_IMPACT_HEIGHT_ERROR:
            return None

        fps = sequence.fps if sequence.fps > 0 else 30.0
        half_width = max(1, int(round(IMPACT_HALF_WIDTH_AT_30FPS * fps / 30.0)))
        start = max(top_frame + 1, centre - half_width)
        end = min(len(path) - 1, centre + half_width)

        # Closer return to address height means a more confident region; a
        # wider window relative to the clip means a vaguer one.
        closeness = 1.0 - min(best_error / MAX_IMPACT_HEIGHT_ERROR, 1.0)
        hand_visibility = float(
            np.max(sequence.visibility[centre, [lm.LEFT_WRIST, lm.RIGHT_WRIST]])
        )
        confidence = float(np.clip(0.25 + 0.5 * closeness + 0.25 * hand_visibility, 0, 1))

        detail: Dict[str, object] = {
            "method": "hands closest to address height after the top",
            "centre_frame": centre,
            "height_error_shoulder_widths": round(best_error, 3),
            "half_width_frames": half_width,
            "note": (
                "A region, not a frame — the clubhead is not tracked, so impact "
                "cannot be localised to a single preview frame."
            ),
        }
        return start, end, confidence, detail

    def _detect_finish(
        self,
        path: np.ndarray,
        usable: np.ndarray,
        speed: np.ndarray,
        impact_end: int,
    ) -> Optional[Tuple[int, float]]:
        """First frame of sustained quiet after impact — the held finish.

        The mirror of address detection, and it uses the same threshold, so a
        clip that settles the way it started is measured consistently at both
        ends. A clip that stops while the golfer is still moving reports no
        finish rather than treating the last frame as one.
        """
        finite = speed[np.isfinite(speed)]
        if finite.size == 0:
            return None
        peak = float(np.max(finite))
        if peak <= 1e-9:
            return None
        noise_floor = float(np.percentile(finite, 10))
        threshold = max(
            peak * QUIET_SPEED_FRACTION,
            min(noise_floor * NOISE_FLOOR_MULTIPLE, peak * NOISE_CAP_FRACTION),
        )

        run = 0
        run_start = None
        for index in range(impact_end + 1, len(speed)):
            if not np.isfinite(speed[index]):
                continue
            if speed[index] <= threshold:
                if run == 0:
                    run_start = index
                run += 1
                if run >= SUSTAINED_QUIET_FRAMES:
                    quiet_after = int(
                        np.count_nonzero(
                            np.isfinite(speed[run_start:]) & (speed[run_start:] <= threshold)
                        )
                    )
                    confidence = float(
                        np.clip(0.4 + 0.6 * min(quiet_after / 8.0, 1.0), 0, 1)
                    )
                    return run_start, confidence
            else:
                run = 0
                run_start = None
        return None

    @staticmethod
    def _shoulder_width(sequence: PoseSequence, frame: int) -> float:
        return float(
            np.linalg.norm(
                sequence.landmarks[frame, lm.LEFT_SHOULDER, :2]
                - sequence.landmarks[frame, lm.RIGHT_SHOULDER, :2]
            )
        )


def default_detector() -> HandPathPhaseDetector:
    return HandPathPhaseDetector()
