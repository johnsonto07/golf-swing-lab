"""What a tracer is made of: an impact frame, ball points, and curve controls.

Three rules shape everything here.

**Provenance is never lost.** A point the user clicked, a point a tracker
found, and a point the curve invented are stored as different things and can
always be told apart. The failure being guarded against is the one that makes
a tracer dishonest: an estimated point that looks, after a save and a reload,
exactly like something that was observed.

**The tracer is a drawing, not a measurement.** Nothing here is a physical
model of ball flight. Curvature and apex are screen-space shape controls in
normalized image coordinates, and calling them anything else would invite
comparison with launch-monitor numbers this project cannot produce.

**Nothing exists before impact.** A tracer line drawn before the ball is
struck is not an artistic choice, it is a false claim about the video. The
impact frame is therefore required before any point can be placed, and
:meth:`TracerSpec.is_visible_at` is the single place that rule is enforced.

Coordinates are normalized to 0-1 against the **preview** frame, matching
``pose.sequence``. Storing pixels would silently break if the preview were
ever regenerated at a different width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from golf_lab.models.video import Handedness, ShotShape

# Bump when a change could alter how a stored tracer is interpreted, so an
# existing drawing can be flagged rather than silently redrawn wrong.
TRACER_SCHEMA_VERSION = 1


class PointSource(str, Enum):
    """Where a ball point came from. Never inferred, never upgraded.

    The ordering of these three is a claim about evidence, not about quality:
    ``CONFIRMED`` is something a person saw, ``TRACKED`` is something an
    algorithm found and a person may not have checked, and ``ESTIMATED`` is
    the curve filling a gap. Rendering keeps them visually distinct for the
    same reason storage keeps them separate.
    """

    CONFIRMED = "confirmed"
    TRACKED = "tracked"
    ESTIMATED = "estimated"

    @property
    def is_observed(self) -> bool:
        """Whether this point reflects something actually seen in a frame."""
        return self is not PointSource.ESTIMATED

    @property
    def display_name(self) -> str:
        return self.value.title()


class ImpactSource(str, Enum):
    """Who decided where impact is.

    The detector only ever *suggests*. ``phases`` locates an impact region and
    is routinely unable to (on the clip that drove the timing work it found
    none at all), so a tracer that could not be built without it would be
    unusable on exactly the footage that needs it most.
    """

    USER = "user"
    DETECTOR = "detector"


class TracerHeight(str, Enum):
    """Apex preset. Seeds the curve; the user edits it afterwards."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def display_name(self) -> str:
        return self.value.title()


class TracerError(ValueError):
    """A tracer was asked to represent something it must not represent."""


@dataclass(frozen=True)
class BallPoint:
    """One ball position on one preview frame.

    ``x``/``y`` are normalized against the preview frame, with the origin at
    top-left to match image convention. Confirmed points must land inside the
    frame — a click outside the displayed image is a bug in the caller, not a
    ball. Tracked and estimated points may fall outside it, because a struck
    ball genuinely leaves the frame and the curve has to follow it there.
    """

    frame: int
    x: float
    y: float
    source: PointSource
    # Only meaningful for TRACKED points; a tracker reports how sure it is.
    # Confirmed points do not carry one: a person either placed it or did not.
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise TracerError(
                f"Ball point frame must be a frame index, got {self.frame}."
            )
        if self.source is PointSource.CONFIRMED and not self.is_inside_frame:
            raise TracerError(
                f"A confirmed ball point must lie inside the frame, got "
                f"({self.x:.3f}, {self.y:.3f}). Normalized coordinates run 0-1."
            )
        if self.source is not PointSource.TRACKED and self.confidence is not None:
            raise TracerError(
                f"Only tracked points carry a confidence; {self.source.value} "
                f"points must not."
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise TracerError(
                f"Confidence must be between 0 and 1, got {self.confidence}."
            )

    @property
    def is_inside_frame(self) -> bool:
        return 0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0

    def to_pixels(self, width: int, height: int) -> Tuple[float, float]:
        """Convert to pixel coordinates, refusing if the size is unknown.

        Mirrors ``PoseSequence.pixel_landmarks``: a zero dimension means the
        caller does not actually know the frame size, and multiplying by it
        would produce a confident (0, 0) rather than an error.
        """
        if width <= 0 or height <= 0:
            raise TracerError(
                "Frame dimensions are unknown, so normalized ball coordinates "
                "cannot be converted to pixels."
            )
        return self.x * width, self.y * height

    def moved_to(self, x: float, y: float) -> "BallPoint":
        """A user correction of this point. Always becomes confirmed.

        Correcting a tracked point is an observation, so the result carries no
        tracker confidence — keeping the old number would attribute a person's
        judgement to the algorithm.
        """
        return BallPoint(
            frame=self.frame, x=x, y=y, source=PointSource.CONFIRMED
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "frame": self.frame,
            "x": self.x,
            "y": self.y,
            "source": self.source.value,
        }
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BallPoint":
        confidence = data.get("confidence")
        return cls(
            frame=int(data["frame"]),
            x=float(data["x"]),
            y=float(data["y"]),
            source=PointSource(data["source"]),
            confidence=None if confidence is None else float(confidence),
        )


@dataclass
class CurveControls:
    """The editable shape of the drawn line, in screen space.

    Defaults describe a straight vertical line with no curvature, which is the
    honest starting point: it asserts nothing about shape until the user
    chooses a preset or drags a control.
    """

    # Degrees from straight up the frame, positive to the right. This is a
    # direction on screen, not a launch angle relative to the target line.
    launch_direction_degrees: float = 0.0
    # Apex height as a fraction of frame height above the impact point.
    apex_height: float = 0.5
    # Signed lateral bend: positive bends right on screen, negative left.
    # Zero is straight, and stays straight regardless of shot shape.
    curvature: float = 0.0
    # Where the line stops, normalized. None means "not yet placed", which is
    # different from placing it at the origin.
    endpoint: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "launch_direction_degrees": self.launch_direction_degrees,
            "apex_height": self.apex_height,
            "curvature": self.curvature,
            "endpoint": list(self.endpoint) if self.endpoint else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurveControls":
        endpoint = data.get("endpoint")
        return cls(
            launch_direction_degrees=float(
                data.get("launch_direction_degrees", 0.0)
            ),
            apex_height=float(data.get("apex_height", 0.5)),
            curvature=float(data.get("curvature", 0.0)),
            endpoint=(
                (float(endpoint[0]), float(endpoint[1])) if endpoint else None
            ),
        )


# How strongly each shape bends the line, before handedness is applied.
# These seed an editable control; they are not measurements of real curvature.
# Sign convention is "as seen by a right-handed golfer's camera": positive
# bends right. PUSH and PULL are deliberately 0.0 — they start offline and fly
# straight, which is exactly what distinguishes them from a fade and a draw.
_SHAPE_CURVATURE: Dict[ShotShape, float] = {
    ShotShape.STRAIGHT: 0.0,
    ShotShape.FADE: 0.25,
    ShotShape.SLICE: 0.6,
    ShotShape.DRAW: -0.25,
    ShotShape.HOOK: -0.6,
    ShotShape.PUSH: 0.0,
    ShotShape.PULL: 0.0,
    ShotShape.UNKNOWN: 0.0,
}

# Initial launch direction for shapes that start offline, in screen degrees.
_SHAPE_LAUNCH: Dict[ShotShape, float] = {
    ShotShape.PUSH: 8.0,
    ShotShape.PULL: -8.0,
}

_HEIGHT_APEX: Dict[TracerHeight, float] = {
    TracerHeight.LOW: 0.25,
    TracerHeight.MEDIUM: 0.5,
    TracerHeight.HIGH: 0.75,
}


def seed_controls(
    shape: ShotShape,
    height: TracerHeight,
    handedness: Handedness = Handedness.RIGHT,
) -> CurveControls:
    """Turn a shape and height preset into a starting curve.

    Handedness mirrors the lateral component: a left-handed golfer's fade
    bends the other way on screen. Height is unaffected by it.
    """
    mirror = -1.0 if handedness is Handedness.LEFT else 1.0
    return CurveControls(
        launch_direction_degrees=_SHAPE_LAUNCH.get(shape, 0.0) * mirror,
        apex_height=_HEIGHT_APEX[height],
        curvature=_SHAPE_CURVATURE[shape] * mirror,
    )


@dataclass
class TracerSpec:
    """Everything needed to draw one swing's tracer, and nothing derived.

    Deliberately holds no rendered curve. The spline is a pure function of
    these fields, so storing its output would create a second source of truth
    that could disagree with the controls after an edit.
    """

    swing_id: str
    # Preview frame index of impact. Optional because a tracer is created
    # before it is confirmed, and every point placement checks it.
    impact_frame: Optional[int] = None
    impact_source: Optional[ImpactSource] = None

    points: List[BallPoint] = field(default_factory=list)

    shape: ShotShape = ShotShape.UNKNOWN
    height: TracerHeight = TracerHeight.MEDIUM
    handedness: Handedness = Handedness.RIGHT
    controls: CurveControls = field(default_factory=CurveControls)

    # Ties the drawing to the media it was drawn on. If the preview is
    # regenerated the coordinates describe a frame that no longer exists.
    preview_fingerprint: str = ""

    schema_version: int = TRACER_SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = self.created_at
        if (self.impact_frame is None) != (self.impact_source is None):
            raise TracerError(
                "An impact frame and its source must be set together, so a "
                "stored impact can always say who chose it."
            )
        if self.impact_frame is not None and self.impact_frame < 0:
            raise TracerError(
                f"Impact frame must be a frame index, got {self.impact_frame}."
            )

    @property
    def has_impact(self) -> bool:
        return self.impact_frame is not None

    @property
    def confirmed_points(self) -> List[BallPoint]:
        return [p for p in self.points if p.source is PointSource.CONFIRMED]

    @property
    def observed_points(self) -> List[BallPoint]:
        """Points backed by a frame someone or something looked at."""
        return [p for p in self.points if p.source.is_observed]

    def confirm_impact(self, frame: int, source: ImpactSource) -> None:
        """Set the impact frame, dropping anything that predates it.

        Moving impact later can strand points behind it. Those points were
        placed in good faith, but a point before impact cannot be part of the
        flight, so they are removed rather than kept and hidden.
        """
        if frame < 0:
            raise TracerError(f"Impact frame must be a frame index, got {frame}.")
        self.impact_frame = frame
        self.impact_source = source
        self.points = [p for p in self.points if p.frame >= frame]
        self.touch()

    def add_point(self, point: BallPoint) -> None:
        """Place a point, replacing any existing point on the same frame.

        One frame holds one ball. Appending instead would let two positions
        for the same instant coexist, and the curve would have to silently
        pick one.
        """
        if not self.has_impact:
            raise TracerError(
                "Confirm the impact frame before placing the ball. A point "
                "placed first has nothing to be measured from."
            )
        assert self.impact_frame is not None  # narrowed by has_impact
        if point.frame < self.impact_frame:
            raise TracerError(
                f"Frame {point.frame} is before the confirmed impact frame "
                f"{self.impact_frame}, so the ball has not been struck yet."
            )
        self.points = [p for p in self.points if p.frame != point.frame]
        self.points.append(point)
        self.points.sort(key=lambda p: p.frame)
        self.touch()

    def remove_point(self, frame: int) -> bool:
        """Delete the point on a frame. Returns whether one was there."""
        before = len(self.points)
        self.points = [p for p in self.points if p.frame != frame]
        if len(self.points) != before:
            self.touch()
            return True
        return False

    def point_at(self, frame: int) -> Optional[BallPoint]:
        return next((p for p in self.points if p.frame == frame), None)

    def is_visible_at(self, frame: int) -> bool:
        """Whether the tracer may be drawn on this frame at all.

        The one rule the renderer must not be trusted to reimplement.
        """
        if self.impact_frame is None:
            return False
        return frame >= self.impact_frame

    def apply_preset(
        self, shape: ShotShape, height: TracerHeight
    ) -> None:
        """Re-seed the curve from presets, discarding manual control edits."""
        self.shape = shape
        self.height = height
        self.controls = seed_controls(shape, height, self.handedness)
        self.touch()

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "swing_id": self.swing_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "impact_frame": self.impact_frame,
            "impact_source": (
                self.impact_source.value if self.impact_source else None
            ),
            "points": [p.to_dict() for p in self.points],
            "shape": self.shape.value,
            "height": self.height.value,
            "handedness": self.handedness.value,
            "controls": self.controls.to_dict(),
            "preview_fingerprint": self.preview_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TracerSpec":
        impact_source = data.get("impact_source")
        impact_frame = data.get("impact_frame")
        return cls(
            swing_id=data["swing_id"],
            impact_frame=None if impact_frame is None else int(impact_frame),
            impact_source=(
                ImpactSource(impact_source) if impact_source else None
            ),
            points=[BallPoint.from_dict(p) for p in data.get("points") or []],
            shape=ShotShape(data.get("shape", ShotShape.UNKNOWN.value)),
            height=TracerHeight(data.get("height", TracerHeight.MEDIUM.value)),
            handedness=Handedness(
                data.get("handedness", Handedness.RIGHT.value)
            ),
            controls=CurveControls.from_dict(data.get("controls") or {}),
            preview_fingerprint=data.get("preview_fingerprint", ""),
            schema_version=int(data.get("schema_version", 0)),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
