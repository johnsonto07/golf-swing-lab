"""Phase detection on deterministic synthetic swings.

The fixture builds a swing whose address and top frames are known exactly, so
these assert the detector found *the right frame*, not merely a plausible one.
"""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.models.video import CameraView
from golf_lab.pose.sequence import PoseSequence
from golf_lab.swing.geometry_detector import HandPathPhaseDetector, default_detector
from golf_lab.swing.phases import (
    PHASE_ORDER,
    PhaseResult,
    SwingPhase,
    SwingPhaseDetector,
    SwingPhases,
)
from golf_lab.swing.results import ResultStatus


@pytest.fixture()
def detector():
    return default_detector()


class TestProtocolConformance:
    def test_detector_satisfies_the_protocol(self, detector):
        assert isinstance(detector, SwingPhaseDetector)

    def test_detector_declares_identity_and_scope(self, detector):
        assert detector.name
        assert detector.version
        # It declares what it attempts rather than emitting placeholders for
        # phases it cannot find. As of v2 that is the full set.
        assert set(detector.supported_phases) == set(PHASE_ORDER)

    def test_unsupported_phases_are_simply_absent(self, detector, swing_pose_factory):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        for phase in PHASE_ORDER:
            if phase not in detector.supported_phases:
                assert result.get(phase) is None, (
                    f"{phase.value} was reported by a detector that does not "
                    "claim to detect it"
                )

    def test_result_records_provenance(self, detector, swing_pose_factory):
        sequence = swing_pose_factory()
        result = detector.detect(sequence, CameraView.DOWN_THE_LINE)
        assert result.detector_name == detector.name
        assert result.detector_version == detector.version
        assert result.camera_view is CameraView.DOWN_THE_LINE
        assert result.frame_count == sequence.frame_count
        assert result.created_at


class TestAddressDetection:
    def test_finds_the_end_of_the_quiet_period(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(address_frames=12)
        result = detector.detect(sequence, CameraView.FACE_ON)

        address = result.get(SwingPhase.ADDRESS)
        assert address.status.is_usable
        expected = int(sequence.metadata["expected_address"])
        assert abs(address.start_frame - expected) <= 2, (
            f"address at {address.start_frame}, expected near {expected}"
        )

    def test_takes_the_last_quiet_frame_not_the_first(self, detector, swing_pose_factory):
        # A long settle before the swing must not drag address to frame 0.
        sequence = swing_pose_factory(address_frames=25)
        address = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.ADDRESS)
        assert address.start_frame > 15

    def test_fails_cleanly_when_the_swing_is_already_moving(
        self, detector, swing_pose_factory
    ):
        sequence = swing_pose_factory(address_frames=0, backswing_frames=30)
        address = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.ADDRESS)

        assert not address.status.is_usable
        assert address.start_frame is None
        assert "hands are moving" in address.reason

    def test_top_is_not_reported_when_address_failed(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(address_frames=0)
        top = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.TOP_OF_BACKSWING)
        assert not top.status.is_usable
        assert top.start_frame is None


class TestTopDetection:
    def test_finds_the_top_of_the_backswing(self, detector, swing_pose_factory):
        sequence = swing_pose_factory()
        top = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.TOP_OF_BACKSWING)

        assert top.status.is_usable
        expected = int(sequence.metadata["expected_top"])
        assert abs(top.start_frame - expected) <= 3, (
            f"top at {top.start_frame}, expected near {expected}"
        )

    def test_does_not_mistake_the_finish_for_the_top(self, detector, swing_pose_factory):
        # The finish has the hands just as high. Bounding the search by peak
        # hand speed is what keeps them apart, and this is the test that
        # would fail if that bound were removed.
        sequence = swing_pose_factory(follow_frames=40)
        result = detector.detect(sequence, CameraView.FACE_ON)

        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        address = result.get(SwingPhase.ADDRESS)
        assert top.start_frame < sequence.frame_count * 0.65
        assert top.start_frame > address.start_frame

    def test_top_comes_after_address(self, detector, swing_pose_factory):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        assert (
            result.get(SwingPhase.TOP_OF_BACKSWING).start_frame
            > result.get(SwingPhase.ADDRESS).start_frame
        )

    def test_records_how_it_decided(self, detector, swing_pose_factory):
        top = detector.detect(swing_pose_factory(), CameraView.FACE_ON).get(
            SwingPhase.TOP_OF_BACKSWING
        )
        assert "peak_speed_frame" in top.detail
        assert top.detail["rise_shoulder_widths"] > 0

    def test_no_backswing_means_no_top(self, detector, swing_pose_factory):
        """Hands move but never rise: a putting stroke, or a clip cut short.

        Address is deliberately kept detectable — a still opening followed by
        purely horizontal motion — so this exercises "searched and found no
        backswing" rather than the cascade from a missing address.
        """
        from golf_lab.pose import landmarks as lmk

        sequence = swing_pose_factory(address_frames=12, backswing_frames=30)
        count = sequence.frame_count

        flat = np.full(count, 0.75, dtype=np.float32)
        drift = np.concatenate(
            [np.full(12, 0.45, dtype=np.float32), np.linspace(0.45, 0.62, count - 12, dtype=np.float32)]
        )
        for wrist, offset in ((lmk.LEFT_WRIST, 0.0), (lmk.RIGHT_WRIST, 0.01)):
            sequence.landmarks[:, wrist, 1] = flat
            sequence.landmarks[:, wrist, 0] = drift + offset

        result = detector.detect(sequence, CameraView.FACE_ON)
        assert result.get(SwingPhase.ADDRESS).status.is_usable, (
            "address should still be found; otherwise this tests the cascade"
        )

        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        assert not top.status.is_usable
        assert "never rise" in top.reason

    def test_missing_top_blocks_its_dependents_with_a_named_reason(
        self, detector, swing_pose_factory
    ):
        from golf_lab.pose import landmarks as lmk

        sequence = swing_pose_factory(address_frames=12, backswing_frames=30)
        count = sequence.frame_count
        drift = np.concatenate(
            [np.full(12, 0.45, dtype=np.float32), np.linspace(0.45, 0.62, count - 12, dtype=np.float32)]
        )
        for wrist, offset in ((lmk.LEFT_WRIST, 0.0), (lmk.RIGHT_WRIST, 0.01)):
            sequence.landmarks[:, wrist, 1] = np.full(count, 0.75, dtype=np.float32)
            sequence.landmarks[:, wrist, 0] = drift + offset

        result = detector.detect(sequence, CameraView.FACE_ON)
        for phase in (
            SwingPhase.DOWNSWING,
            SwingPhase.IMPACT_REGION,
            SwingPhase.FOLLOW_THROUGH,
            SwingPhase.FINISH,
        ):
            outcome = result.get(phase)
            assert not outcome.status.is_usable
            assert outcome.start_frame is None
            # A cascade is INSUFFICIENT_FRAMES, not DETECTION_FAILED: nothing
            # was attempted here, and conflating the two would hide the one
            # real failure among four downstream ones.
            assert outcome.status is ResultStatus.INSUFFICIENT_FRAMES
            assert "top of the backswing" in outcome.reason


class TestTakeaway:
    def test_takeaway_is_the_first_frame_of_sustained_motion(
        self, detector, swing_pose_factory
    ):
        sequence = swing_pose_factory(address_frames=12)
        result = detector.detect(sequence, CameraView.FACE_ON)

        address = result.get(SwingPhase.ADDRESS)
        takeaway = result.get(SwingPhase.TAKEAWAY)
        assert takeaway.status.is_usable
        # Address and takeaway are the same measurement read from either side,
        # so they must be adjacent — never overlapping, never separated.
        assert takeaway.start_frame == address.start_frame + 1

    def test_takeaway_precedes_the_top(self, detector, swing_pose_factory):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        assert (
            result.get(SwingPhase.TAKEAWAY).start_frame
            < result.get(SwingPhase.TOP_OF_BACKSWING).start_frame
        )


class TestImpactRegion:
    def test_impact_is_a_region_not_a_frame(self, detector, swing_pose_factory):
        # Without clubhead tracking, a single "impact frame" would claim a
        # precision the input cannot support.
        impact = detector.detect(swing_pose_factory(), CameraView.FACE_ON).get(
            SwingPhase.IMPACT_REGION
        )
        assert impact.status.is_usable
        assert impact.is_range
        assert impact.end_frame > impact.start_frame
        assert "region, not a frame" in impact.detail["note"]

    def test_region_is_centred_where_the_hands_return_to_address_height(
        self, detector, swing_pose_factory
    ):
        sequence = swing_pose_factory(
            address_frames=12, backswing_frames=30, downswing_frames=8
        )
        result = detector.detect(sequence, CameraView.FACE_ON)
        impact = result.get(SwingPhase.IMPACT_REGION)

        # The synthetic swing returns through address height at the end of the
        # downswing segment: 12 + 30 + 8 - 1 = 49.
        assert abs(int(impact.detail["centre_frame"]) - 49) <= 2
        assert impact.start_frame <= impact.detail["centre_frame"] <= impact.end_frame

    def test_region_falls_between_the_top_and_the_finish(
        self, detector, swing_pose_factory
    ):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        assert (
            result.get(SwingPhase.TOP_OF_BACKSWING).start_frame
            < result.get(SwingPhase.IMPACT_REGION).start_frame
        )
        assert (
            result.get(SwingPhase.IMPACT_REGION).end_frame
            < result.get(SwingPhase.FINISH).start_frame
        )

    def test_region_widens_with_frame_rate(self, detector, swing_pose_factory):
        # A 120 fps clip localises impact better in *time*, so the region
        # covers more frames while representing a similar real interval.
        slow = swing_pose_factory(fps=30.0)
        fast = swing_pose_factory(fps=120.0)

        slow_impact = detector.detect(slow, CameraView.FACE_ON).get(SwingPhase.IMPACT_REGION)
        fast_impact = detector.detect(fast, CameraView.FACE_ON).get(SwingPhase.IMPACT_REGION)

        assert int(fast_impact.detail["half_width_frames"]) > int(
            slow_impact.detail["half_width_frames"]
        )

    def test_no_impact_when_the_hands_never_come_back_down(
        self, detector, swing_pose_factory
    ):
        """Top is found, but the hands only descend part way — no impact.

        Built by damping the descent rather than truncating the clip, so the
        top is still locatable and this exercises the impact search itself
        rather than the cascade from a missing top.
        """
        from golf_lab.pose import landmarks as lmk

        sequence = swing_pose_factory(address_frames=12, backswing_frames=30)
        top = int(sequence.metadata["expected_top"])
        for wrist in (lmk.LEFT_WRIST, lmk.RIGHT_WRIST):
            top_y = float(sequence.landmarks[top, wrist, 1])
            after = sequence.landmarks[top + 1 :, wrist, 1]
            # Descend only a quarter of the way back toward the ball.
            sequence.landmarks[top + 1 :, wrist, 1] = top_y + 0.25 * (after - top_y)

        result = detector.detect(sequence, CameraView.FACE_ON)
        assert result.get(SwingPhase.TOP_OF_BACKSWING).status.is_usable, (
            "top should still be found; otherwise this tests the cascade"
        )

        impact = result.get(SwingPhase.IMPACT_REGION)
        assert not impact.status.is_usable
        assert impact.start_frame is None
        assert "never returned" in impact.reason

    def test_backswing_only_clip_finds_no_top_and_says_so(
        self, detector, swing_pose_factory
    ):
        """A documented limitation, pinned so it cannot regress silently.

        On a clip containing only a backswing, the fastest hand movement is
        inside the backswing itself, so the peak-speed bound that normally
        separates the top from the finish collapses. The detector reports no
        top rather than guessing, and the dependent phases cascade.
        """
        sequence = swing_pose_factory(
            address_frames=12, backswing_frames=30, downswing_frames=0, follow_frames=0
        )
        result = detector.detect(sequence, CameraView.FACE_ON)

        assert result.get(SwingPhase.ADDRESS).status.is_usable
        assert not result.get(SwingPhase.TOP_OF_BACKSWING).status.is_usable
        impact = result.get(SwingPhase.IMPACT_REGION)
        assert impact.status is ResultStatus.INSUFFICIENT_FRAMES
        assert "top of the backswing" in impact.reason


class TestDownswingAndFollowThrough:
    def test_downswing_spans_the_top_to_impact(self, detector, swing_pose_factory):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        downswing = result.get(SwingPhase.DOWNSWING)
        impact = result.get(SwingPhase.IMPACT_REGION)

        assert downswing.status.is_usable
        assert downswing.start_frame == top.start_frame + 1
        assert downswing.end_frame == impact.start_frame - 1

    def test_follow_through_spans_impact_to_the_finish(
        self, detector, swing_pose_factory
    ):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        impact = result.get(SwingPhase.IMPACT_REGION)
        follow = result.get(SwingPhase.FOLLOW_THROUGH)
        finish = result.get(SwingPhase.FINISH)

        assert follow.status.is_usable
        assert follow.start_frame == impact.end_frame + 1
        assert follow.end_frame == finish.start_frame - 1

    def test_phases_never_overlap_and_stay_in_order(
        self, detector, swing_pose_factory
    ):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        located = result.available
        assert len(located) == len(PHASE_ORDER), "expected a full swing to resolve"

        previous_end = -1
        for outcome in located:
            assert outcome.start_frame > previous_end, (
                f"{outcome.phase.value} starts at {outcome.start_frame}, "
                f"overlapping the previous phase ending at {previous_end}"
            )
            previous_end = outcome.end_frame


class TestFinish:
    def test_finish_is_where_motion_settles(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(follow_frames=30)
        finish = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.FINISH)

        assert finish.status.is_usable
        # The synthetic swing decelerates into a held finish near the end.
        assert finish.start_frame > sequence.frame_count * 0.6

    def test_no_finish_when_the_clip_ends_mid_motion(
        self, detector, swing_pose_factory
    ):
        # Cut the clip immediately after impact: the hands are still moving.
        sequence = swing_pose_factory(follow_frames=3)
        result = detector.detect(sequence, CameraView.FACE_ON)

        finish = result.get(SwingPhase.FINISH)
        assert not finish.status.is_usable
        assert finish.start_frame is None
        assert "still moving when the clip ends" in finish.reason

    def test_missing_finish_blocks_follow_through(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(follow_frames=3)
        follow = detector.detect(sequence, CameraView.FACE_ON).get(
            SwingPhase.FOLLOW_THROUGH
        )
        assert not follow.status.is_usable
        assert follow.start_frame is None
        assert follow.status is ResultStatus.INSUFFICIENT_FRAMES


class TestFullSwingRobustness:
    def test_all_phases_survive_scattered_gaps(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(failed_frames=(14, 15, 33, 48, 60))
        result = detector.detect(sequence, CameraView.FACE_ON)

        for phase in PHASE_ORDER:
            outcome = result.get(phase)
            assert outcome is not None and outcome.status.is_usable, (
                f"{phase.value} lost to gaps: "
                f"{outcome.status.value if outcome else 'absent'}"
            )

    def test_all_phases_survive_landmark_jitter(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(jitter=0.004, seed=11)
        result = detector.detect(sequence, CameraView.FACE_ON)

        expected_top = int(sequence.metadata["expected_top"])
        assert abs(result.get(SwingPhase.TOP_OF_BACKSWING).start_frame - expected_top) <= 6
        assert result.get(SwingPhase.IMPACT_REGION).status.is_usable
        assert result.get(SwingPhase.FINISH).status.is_usable

    def test_every_phase_reports_confidence_and_status(
        self, detector, swing_pose_factory
    ):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        for phase in PHASE_ORDER:
            outcome = result.get(phase)
            assert outcome is not None
            assert outcome.status in set(ResultStatus)
            if outcome.status.is_usable:
                assert outcome.confidence is not None
                assert 0.0 <= outcome.confidence <= 1.0
            else:
                assert outcome.reason

    def test_no_phase_exposes_a_duration_in_seconds(
        self, detector, swing_pose_factory
    ):
        # GSL-1 is open. Ranges carry frame indices only.
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        for outcome in result.attempted:
            assert not hasattr(outcome, "duration_seconds")
            for key in outcome.detail:
                assert "second" not in key.lower()
                assert "tempo" not in key.lower()


class TestInsufficientData:
    def test_too_few_frames(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(
            address_frames=2, backswing_frames=2, downswing_frames=1, follow_frames=1
        )
        result = detector.detect(sequence, CameraView.FACE_ON)

        for phase in detector.supported_phases:
            outcome = result.get(phase)
            assert outcome.status is ResultStatus.INSUFFICIENT_FRAMES
            assert outcome.start_frame is None

    def test_mostly_undetected_clip_is_refused(self, detector, swing_pose_factory):
        sequence = swing_pose_factory()
        # Knock out 70% of frames: too sparse to describe a swing.
        for index in range(0, sequence.frame_count):
            if index % 10 >= 3:
                sequence.mark_failed(index)

        result = detector.detect(sequence, CameraView.FACE_ON)
        assert result.get(SwingPhase.ADDRESS).status is ResultStatus.INSUFFICIENT_FRAMES

    def test_empty_sequence(self, detector):
        result = detector.detect(PoseSequence.empty(0, fps=30.0), CameraView.FACE_ON)
        assert result.get(SwingPhase.ADDRESS).status is ResultStatus.INSUFFICIENT_FRAMES


class TestMissingLandmarks:
    def test_invisible_hands_are_reported_as_missing_landmarks(
        self, detector, swing_pose_factory
    ):
        # Both phases are derived from the hand path; without hands there is
        # nothing to measure, and that is a different problem from a short clip.
        sequence = swing_pose_factory(hand_visibility=0.01)
        result = detector.detect(sequence, CameraView.FACE_ON)

        address = result.get(SwingPhase.ADDRESS)
        assert address.status is ResultStatus.MISSING_LANDMARKS
        assert address.start_frame is None
        assert "hands" in address.reason.lower()

    def test_partially_visible_hands_still_work(self, detector, swing_pose_factory):
        # Down-the-line footage routinely occludes one wrist; using the better
        # of the two must keep the detector working.
        sequence = swing_pose_factory(hand_visibility=0.45)
        result = detector.detect(sequence, CameraView.FACE_ON)
        assert result.get(SwingPhase.ADDRESS).status.is_usable


class TestGapsAndNoise:
    def test_survives_scattered_detection_gaps(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(failed_frames=(15, 16, 31, 47))
        result = detector.detect(sequence, CameraView.FACE_ON)

        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        assert top.status.is_usable
        expected = int(sequence.metadata["expected_top"])
        assert abs(top.start_frame - expected) <= 5

    def test_never_anchors_a_point_phase_to_an_undetected_frame(
        self, detector, swing_pose_factory
    ):
        """A phase located *at* a frame must be located at an observed frame.

        Range phases are excluded deliberately. Their endpoints are boundaries
        derived from the neighbouring point phases — the downswing runs from
        the frame after the top to the frame before impact — so an endpoint can
        legitimately land in a gap. What must never happen is a *point* phase
        such as address or the top being pinned to a frame where no pose was
        ever seen.
        """
        sequence = swing_pose_factory(failed_frames=(11, 12, 13, 41, 42))
        result = detector.detect(sequence, CameraView.FACE_ON)

        point_phases = {
            SwingPhase.ADDRESS,
            SwingPhase.TAKEAWAY,
            SwingPhase.TOP_OF_BACKSWING,
            SwingPhase.FINISH,
        }
        checked = 0
        for outcome in result.available:
            if outcome.phase not in point_phases:
                continue
            checked += 1
            assert sequence.detected[outcome.start_frame], (
                f"{outcome.phase.value} anchored to undetected frame "
                f"{outcome.start_frame}"
            )
        assert checked >= 2, "expected at least a couple of point phases to check"

    def test_tolerates_landmark_jitter(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(jitter=0.004, seed=3)
        result = detector.detect(sequence, CameraView.FACE_ON)

        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        assert top.status.is_usable
        expected = int(sequence.metadata["expected_top"])
        assert abs(top.start_frame - expected) <= 6


class TestCameraViewIndependence:
    @pytest.mark.parametrize(
        "view",
        [CameraView.FACE_ON, CameraView.DOWN_THE_LINE, CameraView.OTHER, CameraView.UNKNOWN],
    )
    def test_phase_detection_works_from_any_view(
        self, detector, swing_pose_factory, view
    ):
        # Phases come from the hand path, which reads similarly from any angle.
        # Metric *gating* is what depends on the view — that is tested
        # separately, and conflating the two would be a design error.
        result = detector.detect(swing_pose_factory(), view)
        assert result.get(SwingPhase.ADDRESS).status.is_usable
        assert result.get(SwingPhase.TOP_OF_BACKSWING).status.is_usable


class TestTimelineHonesty:
    def test_approximate_timeline_is_recorded_and_explained(
        self, detector, swing_pose_factory
    ):
        result = detector.detect(
            swing_pose_factory(), CameraView.FACE_ON, timeline_is_approximate=True
        )
        assert result.timeline_is_approximate
        assert any("GSL-1" in note for note in result.notes)
        assert any("preview frame numbers" in note for note in result.notes)

    def test_exact_timeline_carries_no_warning(self, detector, swing_pose_factory):
        result = detector.detect(
            swing_pose_factory(), CameraView.FACE_ON, timeline_is_approximate=False
        )
        assert not result.timeline_is_approximate
        assert result.notes == []

    def test_no_duration_between_phases_is_offered(self, detector, swing_pose_factory):
        # Durations in seconds are blocked by GSL-1. The container must not
        # expose one, or a caller will use it.
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        assert not hasattr(result, "duration_seconds")
        assert not hasattr(result, "tempo_ratio")


class TestSwingPhasesContainer:
    def test_available_excludes_failed_phases(self, swing_pose_factory):
        phases = SwingPhases()
        phases.set(PhaseResult.found(SwingPhase.ADDRESS, frame=5, confidence=0.9))
        phases.set(
            PhaseResult.unavailable(
                SwingPhase.TOP_OF_BACKSWING, ResultStatus.DETECTION_FAILED, "nope"
            )
        )
        assert [r.phase for r in phases.available] == [SwingPhase.ADDRESS]
        assert len(phases.attempted) == 2

    def test_frame_for_returns_none_for_unusable_phases(self):
        phases = SwingPhases()
        phases.set(
            PhaseResult.unavailable(
                SwingPhase.ADDRESS, ResultStatus.MISSING_LANDMARKS, "no hands"
            )
        )
        assert phases.frame_for(SwingPhase.ADDRESS) is None

    def test_available_is_returned_in_swing_order(self):
        phases = SwingPhases()
        phases.set(PhaseResult.found(SwingPhase.TOP_OF_BACKSWING, frame=40, confidence=0.9))
        phases.set(PhaseResult.found(SwingPhase.ADDRESS, frame=10, confidence=0.9))
        assert [r.phase for r in phases.available] == [
            SwingPhase.ADDRESS,
            SwingPhase.TOP_OF_BACKSWING,
        ]

    def test_round_trip(self, detector, swing_pose_factory):
        original = detector.detect(
            swing_pose_factory(), CameraView.FACE_ON, timeline_is_approximate=True
        )
        restored = SwingPhases.from_dict(original.to_dict())

        assert restored.detector_name == original.detector_name
        assert restored.detector_version == original.detector_version
        assert restored.camera_view is original.camera_view
        assert restored.timeline_is_approximate
        assert restored.notes == original.notes
        for phase in original.results:
            assert restored.get(phase) == original.get(phase)
