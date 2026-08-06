"""Swing phases: what they are, and how a detection reports itself.

Phases are located as **preview-frame indices**, never as source timestamps.
That distinction is load-bearing rather than pedantic: on variable-frame-rate
footage the preview timeline is resampled, so a preview frame index cannot be
converted to a time in the original file (see docs/KNOWN_ISSUES.md, GSL-1).
Anything expressed in seconds is therefore reported as preview time and
labelled as such, and durations between phases are withheld entirely until
that mapping exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from golf_lab.models.video import CameraView
from golf_lab.pose.sequence import PoseSequence
from golf_lab.swing.results import ResultError, ResultStatus, STATUS_ICONS, STATUS_LABELS

# Bump when a change could alter previously-computed phases, so stored results
# can be flagged stale without re-running MediaPipe.
PHASE_SCHEMA_VERSION = 1


class SwingPhase(str, Enum):
    """The phases this project recognises, in swing order.

    A deliberately coarse vocabulary. P1–P9 exists in coaching literature, but
    most of those positions cannot be located reliably from a single 2D camera,
    and naming them would imply a precision the input does not support.
    """

    ADDRESS = "address"
    TAKEAWAY = "takeaway"
    TOP_OF_BACKSWING = "top_of_backswing"
    DOWNSWING = "downswing"
    IMPACT_REGION = "impact_region"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


PHASE_ORDER: Sequence[SwingPhase] = (
    SwingPhase.ADDRESS,
    SwingPhase.TAKEAWAY,
    SwingPhase.TOP_OF_BACKSWING,
    SwingPhase.DOWNSWING,
    SwingPhase.IMPACT_REGION,
    SwingPhase.FOLLOW_THROUGH,
    SwingPhase.FINISH,
)


@dataclass(frozen=True)
class PhaseResult:
    """Where one phase was found, or why it was not.

    ``start_frame``/``end_frame`` are **preview** frame indices, inclusive.
    A single-frame phase (address, top) has them equal. As with metrics, an
    unavailable result carries no frame numbers at all — a phase reported at
    frame 0 because nothing was found is indistinguishable from a phase
    genuinely at frame 0.
    """

    phase: SwingPhase
    status: ResultStatus
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    confidence: Optional[float] = None
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status.is_usable:
            if self.start_frame is None:
                raise ResultError(
                    f"Phase '{self.phase.value}' reports {self.status.value} but "
                    "has no frame. A usable status must locate the phase."
                )
            if self.end_frame is not None and self.end_frame < self.start_frame:
                raise ResultError(
                    f"Phase '{self.phase.value}' ends ({self.end_frame}) before "
                    f"it starts ({self.start_frame})."
                )
        elif self.start_frame is not None:
            raise ResultError(
                f"Phase '{self.phase.value}' reports {self.status.value} but "
                f"still carries frame {self.start_frame}. Unavailable phases "
                "must not carry frame numbers."
            )
        if not self.status.is_available and not self.reason:
            raise ResultError(
                f"Phase '{self.phase.value}' reports {self.status.value} without "
                "a reason. Every non-available result must explain itself."
            )

    @classmethod
    def found(
        cls,
        phase: SwingPhase,
        frame: int,
        confidence: float,
        end_frame: Optional[int] = None,
        detail: Optional[Dict[str, Any]] = None,
        low_confidence_below: float = 0.5,
        low_confidence_reason: str = "",
    ) -> "PhaseResult":
        """Build a located phase, downgrading to LOW_CONFIDENCE when weak."""
        weak = confidence < low_confidence_below
        return cls(
            phase=phase,
            status=ResultStatus.LOW_CONFIDENCE if weak else ResultStatus.AVAILABLE,
            start_frame=int(frame),
            end_frame=int(end_frame) if end_frame is not None else int(frame),
            confidence=float(confidence),
            reason=(
                low_confidence_reason
                or (
                    f"Detection confidence {confidence:.2f} is below "
                    f"{low_confidence_below:.2f}; treat the frame as approximate."
                )
                if weak
                else ""
            ),
            detail=detail or {},
        )

    @classmethod
    def unavailable(
        cls,
        phase: SwingPhase,
        status: ResultStatus,
        reason: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> "PhaseResult":
        if status.is_usable:
            raise ResultError(
                f"unavailable() called with usable status {status.value}."
            )
        return cls(phase=phase, status=status, reason=reason, detail=detail or {})

    # -- display ---------------------------------------------------------
    @property
    def label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def icon(self) -> str:
        return STATUS_ICONS[self.status]

    @property
    def is_range(self) -> bool:
        return (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame > self.start_frame
        )

    def preview_seconds(self, preview_fps: float) -> Optional[float]:
        """Position on the **preview** timeline, in seconds.

        Named to make misuse awkward. This is not a time in the original file
        whenever the preview was resampled; see GSL-1.
        """
        if self.start_frame is None or preview_fps <= 0:
            return None
        return self.start_frame / preview_fps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": self.confidence,
            "reason": self.reason,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseResult":
        return cls(
            phase=SwingPhase(data["phase"]),
            status=ResultStatus(data["status"]),
            start_frame=data.get("start_frame"),
            end_frame=data.get("end_frame"),
            confidence=data.get("confidence"),
            reason=data.get("reason", ""),
            detail=data.get("detail") or {},
        )


@dataclass
class SwingPhases:
    """Every phase result from one detector run, plus how it was produced."""

    results: Dict[SwingPhase, PhaseResult] = field(default_factory=dict)
    detector_name: str = "unknown"
    detector_version: str = "0"
    schema_version: int = PHASE_SCHEMA_VERSION
    camera_view: CameraView = CameraView.UNKNOWN
    frame_count: int = 0
    preview_fps: float = 0.0
    timeline_is_approximate: bool = False
    created_at: str = ""
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def get(self, phase: SwingPhase) -> Optional[PhaseResult]:
        return self.results.get(phase)

    def set(self, result: PhaseResult) -> None:
        self.results[result.phase] = result

    @property
    def available(self) -> List[PhaseResult]:
        """Located phases, in swing order. Includes low-confidence ones."""
        return [
            self.results[phase]
            for phase in PHASE_ORDER
            if phase in self.results and self.results[phase].status.is_usable
        ]

    @property
    def attempted(self) -> List[PhaseResult]:
        return [self.results[p] for p in PHASE_ORDER if p in self.results]

    def frame_for(self, phase: SwingPhase) -> Optional[int]:
        result = self.results.get(phase)
        return result.start_frame if result and result.status.is_usable else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "camera_view": self.camera_view.value,
            "frame_count": self.frame_count,
            "preview_fps": self.preview_fps,
            "timeline_is_approximate": self.timeline_is_approximate,
            "created_at": self.created_at,
            "notes": list(self.notes),
            "results": [r.to_dict() for r in self.attempted],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwingPhases":
        instance = cls(
            detector_name=data.get("detector_name", "unknown"),
            detector_version=data.get("detector_version", "0"),
            schema_version=int(data.get("schema_version", 0)),
            camera_view=CameraView(data.get("camera_view", "unknown")),
            frame_count=int(data.get("frame_count", 0)),
            preview_fps=float(data.get("preview_fps", 0.0)),
            timeline_is_approximate=bool(data.get("timeline_is_approximate", False)),
            created_at=data.get("created_at", ""),
            notes=list(data.get("notes") or []),
        )
        for payload in data.get("results") or []:
            instance.set(PhaseResult.from_dict(payload))
        return instance


@runtime_checkable
class SwingPhaseDetector(Protocol):
    """Locates swing phases in a pose sequence.

    Implementations declare which phases they attempt via ``supported_phases``
    and simply omit the rest, rather than emitting placeholder results for
    phases they cannot find. "Not attempted by this detector" and "attempted
    and failed" are different facts, and conflating them would make the second
    one — the one worth acting on — invisible.
    """

    name: str
    version: str
    supported_phases: Sequence[SwingPhase]

    def detect(
        self,
        pose_sequence: PoseSequence,
        camera_view: CameraView,
        timeline_is_approximate: bool = False,
    ) -> SwingPhases:
        ...
