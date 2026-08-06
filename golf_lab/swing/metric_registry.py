"""Which metrics a camera angle can actually support.

A single 2D camera cannot see what it is not pointed at. Lateral hip sway is
meaningful face-on and meaningless down-the-line; hand depth is the reverse.
Computing one anyway produces a number that is not wrong so much as
*meaningless*, and a meaningless number is worse than a missing one because it
looks like evidence.

The gating therefore lives in one registry rather than as scattered `if
camera_view == ...` checks in the UI. A metric declares the views it supports
and the landmarks it needs; `evaluate` refuses on the wrong view before any
arithmetic happens, and the refusal is a first-class result the UI can show.

Adding a metric means adding a spec and a compute function here. It does not
mean touching a page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from golf_lab.models.video import CameraView
from golf_lab.pose import landmarks as lm
from golf_lab.pose.sequence import PoseSequence
from golf_lab.swing.results import MetricResult, ResultStatus

# Below this per-landmark visibility a metric is reported as low confidence
# rather than presented as a clean measurement.
LOW_CONFIDENCE_VISIBILITY = 0.5
# Below this the landmark is treated as not found at all.
MISSING_VISIBILITY = 0.2


@dataclass(frozen=True)
class MetricSpec:
    """Declaration of a metric: what it needs and where it is valid."""

    key: str
    display_name: str
    description: str
    camera_views: FrozenSet[CameraView]
    required_landmarks: Tuple[int, ...]
    unit: str = ""
    # Set when a metric cannot be computed correctly yet for a reason outside
    # this module — e.g. anything needing true source timing (GSL-1).
    blocked_reason: Optional[str] = None
    decimals: int = 1

    def supports(self, camera_view: CameraView) -> bool:
        return camera_view in self.camera_views

    @property
    def view_names(self) -> str:
        return ", ".join(
            sorted(view.value.replace("_", " ") for view in self.camera_views)
        )


# A compute function receives everything it could need and returns a raw float.
# Landmark availability, camera-view gating, and status assignment are handled
# by `evaluate` so no individual metric has to remember to do them.
ComputeFn = Callable[[PoseSequence, Dict[str, int]], float]


@dataclass
class _Registration:
    spec: MetricSpec
    compute: Optional[ComputeFn]
    required_phases: Tuple[str, ...] = ()


_REGISTRY: Dict[str, _Registration] = {}


def register(
    spec: MetricSpec,
    compute: Optional[ComputeFn] = None,
    required_phases: Sequence[str] = (),
) -> MetricSpec:
    """Add a metric. ``compute=None`` declares it without implementing it yet."""
    _REGISTRY[spec.key] = _Registration(
        spec=spec, compute=compute, required_phases=tuple(required_phases)
    )
    return spec


def get_spec(key: str) -> MetricSpec:
    return _REGISTRY[key].spec


def all_specs() -> List[MetricSpec]:
    return [reg.spec for reg in _REGISTRY.values()]


def specs_for_view(camera_view: CameraView) -> List[MetricSpec]:
    """Metrics that are valid for this camera angle."""
    return [reg.spec for reg in _REGISTRY.values() if reg.spec.supports(camera_view)]

def is_implemented(key: str) -> bool:
    return _REGISTRY[key].compute is not None


# --- landmark helpers ----------------------------------------------------
def _visibility_at(sequence: PoseSequence, frame: int, indices: Sequence[int]) -> float:
    return float(np.min(sequence.visibility[frame, list(indices)]))


def _point(sequence: PoseSequence, frame: int, index: int) -> np.ndarray:
    return sequence.landmarks[frame, index, :2].astype(np.float64)


def _midpoint(sequence: PoseSequence, frame: int, a: int, b: int) -> np.ndarray:
    return (_point(sequence, frame, a) + _point(sequence, frame, b)) / 2.0


def _shoulder_width(sequence: PoseSequence, frame: int) -> float:
    """Body-scale reference so measurements are camera-distance independent."""
    return float(
        np.linalg.norm(
            _point(sequence, frame, lm.LEFT_SHOULDER)
            - _point(sequence, frame, lm.RIGHT_SHOULDER)
        )
    )


# --- metric computations -------------------------------------------------
def _head_sway(sequence: PoseSequence, frames: Dict[str, int]) -> float:
    """Lateral head movement from address to top, in shoulder widths.

    Face-on only: from down-the-line the same motion is mostly toward or away
    from the camera and does not project onto the image x-axis.
    """
    address, top = frames["address"], frames["top_of_backswing"]
    scale = _shoulder_width(sequence, address)
    if scale <= 1e-6:
        raise ValueError("shoulder width is degenerate")
    dx = _point(sequence, top, lm.NOSE)[0] - _point(sequence, address, lm.NOSE)[0]
    return float(dx / scale)


def _hip_sway(sequence: PoseSequence, frames: Dict[str, int]) -> float:
    """Lateral hip-centre movement from address to top, in shoulder widths."""
    address, top = frames["address"], frames["top_of_backswing"]
    scale = _shoulder_width(sequence, address)
    if scale <= 1e-6:
        raise ValueError("shoulder width is degenerate")
    start = _midpoint(sequence, address, lm.LEFT_HIP, lm.RIGHT_HIP)
    end = _midpoint(sequence, top, lm.LEFT_HIP, lm.RIGHT_HIP)
    return float((end[0] - start[0]) / scale)


def _line_tilt_degrees(start: np.ndarray, end: np.ndarray) -> float:
    """Tilt of a body line from horizontal, wrapped to -90..90 degrees.

    Wrapping is essential, not cosmetic. MediaPipe labels landmarks
    anatomically, so for a golfer facing the camera the *left* shoulder appears
    on the *right* of the image. Raw ``arctan2`` then returns ~180 degrees for
    perfectly level shoulders — a number that looks like a measurement and is
    off by a half turn. Wrapping makes level read as 0 whichever way round the
    landmarks happen to fall.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        raise ValueError("the two landmarks coincide")
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def _shoulder_tilt(sequence: PoseSequence, frames: Dict[str, int]) -> float:
    """Shoulder-line angle at address, degrees from horizontal.

    0 is level. The sign follows image orientation rather than being claimed
    as "lead shoulder higher", because which shoulder leads depends on
    handedness and the camera side, and this metric does not know either.
    """
    address = frames["address"]
    return _line_tilt_degrees(
        _point(sequence, address, lm.LEFT_SHOULDER),
        _point(sequence, address, lm.RIGHT_SHOULDER),
    )


def _head_movement(sequence: PoseSequence, frames: Dict[str, int]) -> float:
    """Total head displacement from address to top, in shoulder widths.

    Magnitude only, so it stays meaningful down-the-line where the direction
    of the movement is partly along the camera axis and cannot be resolved.
    """
    address, top = frames["address"], frames["top_of_backswing"]
    scale = _shoulder_width(sequence, address)
    if scale <= 1e-6:
        raise ValueError("shoulder width is degenerate")
    delta = _point(sequence, top, lm.NOSE) - _point(sequence, address, lm.NOSE)
    return float(np.linalg.norm(delta) / scale)


def _spine_angle(sequence: PoseSequence, frames: Dict[str, int]) -> float:
    """Forward spine lean at address, degrees from vertical.

    Down-the-line only: face-on, the same posture is edge-on to the camera and
    the projection says nothing about how much the golfer is bent over.
    """
    address = frames["address"]
    shoulders = _midpoint(sequence, address, lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER)
    hips = _midpoint(sequence, address, lm.LEFT_HIP, lm.RIGHT_HIP)
    dx, dy = shoulders[0] - hips[0], hips[1] - shoulders[1]
    if abs(dy) < 1e-9:
        raise ValueError("spine is horizontal in the image")
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


# --- registrations -------------------------------------------------------
FACE_ON = frozenset({CameraView.FACE_ON})
DOWN_THE_LINE = frozenset({CameraView.DOWN_THE_LINE})

HEAD_SWAY = register(
    MetricSpec(
        key="head_sway",
        display_name="Head sway (address → top)",
        description=(
            "Lateral head movement between address and the top of the "
            "backswing, in shoulder widths. Positive is toward the trail side."
        ),
        camera_views=FACE_ON,
        required_landmarks=(lm.NOSE, lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER),
        unit="shoulder widths",
        decimals=2,
    ),
    compute=_head_sway,
    required_phases=("address", "top_of_backswing"),
)

HIP_SWAY = register(
    MetricSpec(
        key="hip_sway",
        display_name="Hip sway (address → top)",
        description=(
            "Lateral movement of the hip centre between address and the top, "
            "in shoulder widths."
        ),
        camera_views=FACE_ON,
        required_landmarks=(
            lm.LEFT_HIP,
            lm.RIGHT_HIP,
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,
        ),
        unit="shoulder widths",
        decimals=2,
    ),
    compute=_hip_sway,
    required_phases=("address", "top_of_backswing"),
)

SHOULDER_TILT = register(
    MetricSpec(
        key="shoulder_tilt",
        display_name="Shoulder tilt at address",
        description="Angle of the shoulder line at address, degrees from horizontal.",
        camera_views=FACE_ON,
        required_landmarks=(lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER),
        unit="°",
    ),
    compute=_shoulder_tilt,
    required_phases=("address",),
)

HEAD_MOVEMENT = register(
    MetricSpec(
        key="head_movement",
        display_name="Head movement (address → top)",
        description=(
            "Total head displacement between address and the top, in shoulder "
            "widths. Magnitude only — direction is not resolvable from this view."
        ),
        camera_views=DOWN_THE_LINE,
        required_landmarks=(lm.NOSE, lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER),
        unit="shoulder widths",
        decimals=2,
    ),
    compute=_head_movement,
    required_phases=("address", "top_of_backswing"),
)

SPINE_ANGLE = register(
    MetricSpec(
        key="spine_angle",
        display_name="Spine angle at address",
        description="Forward lean of the spine at address, degrees from vertical.",
        camera_views=DOWN_THE_LINE,
        required_landmarks=(
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,
            lm.LEFT_HIP,
            lm.RIGHT_HIP,
        ),
        unit="°",
    ),
    compute=_spine_angle,
    required_phases=("address",),
)

# Declared but not implemented. Registered so the UI can show what a camera
# view will eventually support, and so gating is defined in one place from the
# start. `evaluate` reports them as unavailable rather than inventing a value.
LEAD_ARM_ANGLE = register(
    MetricSpec(
        key="lead_arm_angle",
        display_name="Lead-arm angle at top",
        description="Angle of the lead arm at the top of the backswing.",
        camera_views=FACE_ON,
        required_landmarks=(lm.LEFT_SHOULDER, lm.LEFT_ELBOW, lm.LEFT_WRIST),
        unit="°",
    ),
    required_phases=("top_of_backswing",),
)

HIP_LINE = register(
    MetricSpec(
        key="hip_line",
        display_name="Hip-line estimate at address",
        description="Angle of the hip line at address, degrees from horizontal.",
        camera_views=FACE_ON,
        required_landmarks=(lm.LEFT_HIP, lm.RIGHT_HIP),
        unit="°",
    ),
    required_phases=("address",),
)

HIP_DEPTH = register(
    MetricSpec(
        key="hip_depth",
        display_name="Hip depth change",
        description="Change in hip depth between address and the top.",
        camera_views=DOWN_THE_LINE,
        required_landmarks=(lm.LEFT_HIP, lm.RIGHT_HIP),
        unit="shoulder widths",
    ),
    required_phases=("address", "top_of_backswing"),
)

POSTURE_CHANGE = register(
    MetricSpec(
        key="posture_change",
        display_name="Posture change (address → top)",
        description="Change in spine angle between address and the top.",
        camera_views=DOWN_THE_LINE,
        required_landmarks=(
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,
            lm.LEFT_HIP,
            lm.RIGHT_HIP,
        ),
        unit="°",
    ),
    required_phases=("address", "top_of_backswing"),
)

SHOULDER_LINE = register(
    MetricSpec(
        key="shoulder_line",
        display_name="Shoulder-line estimate at address",
        description="Orientation of the shoulder line at address.",
        camera_views=DOWN_THE_LINE,
        required_landmarks=(lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER),
        unit="°",
    ),
    required_phases=("address",),
)


# --- evaluation ----------------------------------------------------------
def evaluate(
    key: str,
    sequence: PoseSequence,
    camera_view: CameraView,
    phase_frames: Dict[str, int],
) -> MetricResult:
    """Compute one metric, or explain precisely why it cannot be computed.

    The checks run in order of how fundamental the obstacle is, so the reason
    the user sees is the one they can actually act on.
    """
    registration = _REGISTRY[key]
    spec = registration.spec

    # 1. Wrong camera angle — no amount of good data would help.
    if not spec.supports(camera_view):
        view = camera_view.value.replace("_", " ")
        return MetricResult.unavailable(
            spec.key,
            spec.display_name,
            ResultStatus.UNSUPPORTED_CAMERA_VIEW,
            f"Needs a {spec.view_names} view; this swing is recorded as {view}.",
        )

    # 2. Blocked for a reason outside this module (e.g. source timing).
    if spec.blocked_reason:
        return MetricResult.unavailable(
            spec.key, spec.display_name, ResultStatus.BLOCKED_BY_TIMING, spec.blocked_reason
        )

    # 3. Not implemented yet — reported honestly, never as a value.
    if registration.compute is None:
        return MetricResult.unavailable(
            spec.key,
            spec.display_name,
            ResultStatus.DETECTION_FAILED,
            "This metric is declared for this camera view but is not implemented yet.",
        )

    # 4. The phases it depends on must have been located.
    missing_phases = [p for p in registration.required_phases if p not in phase_frames]
    if missing_phases:
        return MetricResult.unavailable(
            spec.key,
            spec.display_name,
            ResultStatus.INSUFFICIENT_FRAMES,
            "Needs "
            + " and ".join(p.replace("_", " ") for p in missing_phases)
            + ", which was not detected for this swing.",
        )

    frames = {name: phase_frames[name] for name in registration.required_phases}

    # 5. The landmarks it reads must actually be present on those frames.
    worst_visibility = 1.0
    for frame in frames.values():
        if not sequence.detected[frame]:
            return MetricResult.unavailable(
                spec.key,
                spec.display_name,
                ResultStatus.MISSING_LANDMARKS,
                f"No pose was detected on preview frame {frame}.",
            )
        worst_visibility = min(
            worst_visibility, _visibility_at(sequence, frame, spec.required_landmarks)
        )

    if worst_visibility < MISSING_VISIBILITY:
        return MetricResult.unavailable(
            spec.key,
            spec.display_name,
            ResultStatus.MISSING_LANDMARKS,
            f"A required landmark was barely visible (visibility "
            f"{worst_visibility:.2f}); the measurement would be meaningless.",
        )

    # 6. Compute. A geometric degeneracy is a failure, not a zero.
    try:
        value = registration.compute(sequence, frames)
    except (ValueError, ZeroDivisionError, IndexError) as exc:
        return MetricResult.unavailable(
            spec.key,
            spec.display_name,
            ResultStatus.DETECTION_FAILED,
            f"Could not be computed from the detected landmarks: {exc}",
        )

    if not np.isfinite(value):
        return MetricResult.unavailable(
            spec.key,
            spec.display_name,
            ResultStatus.DETECTION_FAILED,
            "The computation produced a non-finite result.",
        )

    if worst_visibility < LOW_CONFIDENCE_VISIBILITY:
        return MetricResult.low_confidence(
            spec.key,
            spec.display_name,
            value,
            reason=(
                f"Lowest required-landmark visibility was {worst_visibility:.2f}. "
                "The value is real but should be treated as indicative."
            ),
            unit=spec.unit,
            confidence=worst_visibility,
            detail={"frames": frames},
        )

    return MetricResult.available(
        spec.key,
        spec.display_name,
        value,
        unit=spec.unit,
        confidence=worst_visibility,
        detail={"frames": frames},
    )


def evaluate_all(
    sequence: PoseSequence,
    camera_view: CameraView,
    phase_frames: Dict[str, int],
    include_unsupported: bool = False,
) -> List[MetricResult]:
    """Evaluate every registered metric for this swing.

    By default only metrics valid for the camera view are returned, because a
    list dominated by "not valid from this view" trains people to skim past
    the statuses that matter.
    """
    results = []
    for key, registration in _REGISTRY.items():
        if not include_unsupported and not registration.spec.supports(camera_view):
            continue
        results.append(evaluate(key, sequence, camera_view, phase_frames))
    return results
