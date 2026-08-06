"""How a phase or metric reports what it knows — including that it knows nothing.

The single rule this module exists to enforce: **a result always carries an
explicit status, and an unavailable result carries no number.**

The tempting alternative — returning 0.0, or None with a comment, or a value
plus a separate "valid" flag someone forgets to check — is how a tool ends up
displaying "head sway: 0.0 cm" for a swing where the head was never located.
A number that looks precise is read as precise. So `value` is only populated
when `status` is AVAILABLE, and the constructors refuse to build anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ResultStatus(str, Enum):
    """Why a phase or metric is, or is not, usable.

    Every unavailable status names a *different remedy*, which is the point of
    distinguishing them: a user can act on "film this face-on" but not on a
    generic "unavailable".
    """

    AVAILABLE = "available"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_LANDMARKS = "missing_landmarks"
    UNSUPPORTED_CAMERA_VIEW = "unsupported_camera_view"
    INSUFFICIENT_FRAMES = "insufficient_frames"
    BLOCKED_BY_TIMING = "blocked_by_timing"
    DETECTION_FAILED = "detection_failed"

    @property
    def is_available(self) -> bool:
        return self is ResultStatus.AVAILABLE

    @property
    def is_usable(self) -> bool:
        """Whether a value exists at all.

        LOW_CONFIDENCE still carries a value — it is a real measurement the
        user should weigh, not a missing one. Everything else has no value.
        """
        return self in (ResultStatus.AVAILABLE, ResultStatus.LOW_CONFIDENCE)


# Shown verbatim in the UI, so they read as sentences rather than enum names.
STATUS_LABELS: Dict[ResultStatus, str] = {
    ResultStatus.AVAILABLE: "Available",
    ResultStatus.LOW_CONFIDENCE: "Low confidence",
    ResultStatus.MISSING_LANDMARKS: "Required landmarks not found",
    ResultStatus.UNSUPPORTED_CAMERA_VIEW: "Not valid from this camera view",
    ResultStatus.INSUFFICIENT_FRAMES: "Not enough detected frames",
    ResultStatus.BLOCKED_BY_TIMING: "Blocked by a known timing limitation",
    ResultStatus.DETECTION_FAILED: "Detection failed",
}

STATUS_ICONS: Dict[ResultStatus, str] = {
    ResultStatus.AVAILABLE: "🟢",
    ResultStatus.LOW_CONFIDENCE: "🟡",
    ResultStatus.MISSING_LANDMARKS: "⚪",
    ResultStatus.UNSUPPORTED_CAMERA_VIEW: "🔵",
    ResultStatus.INSUFFICIENT_FRAMES: "⚪",
    ResultStatus.BLOCKED_BY_TIMING: "🟠",
    ResultStatus.DETECTION_FAILED: "🔴",
}


class ResultError(RuntimeError):
    """Raised when a result is constructed in a self-contradictory state."""


@dataclass(frozen=True)
class MetricResult:
    """One measured quantity, or an explanation of why there isn't one.

    ``value`` is ``None`` unless the status carries a value. That is enforced,
    not merely documented, because the failure it prevents — a plausible number
    standing in for an unmeasured one — is silent and unrecoverable downstream.
    """

    key: str
    display_name: str
    status: ResultStatus
    value: Optional[float] = None
    unit: str = ""
    confidence: Optional[float] = None
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status.is_usable:
            if self.value is None:
                raise ResultError(
                    f"Metric '{self.key}' reports {self.status.value} but has no "
                    "value. A usable status must carry a measurement."
                )
        elif self.value is not None:
            raise ResultError(
                f"Metric '{self.key}' reports {self.status.value} but still "
                f"carries value {self.value!r}. Unavailable results must not "
                "carry numbers — that is how a placeholder becomes a fact."
            )
        if not self.status.is_available and not self.reason:
            raise ResultError(
                f"Metric '{self.key}' reports {self.status.value} without a "
                "reason. Every non-available result must explain itself."
            )

    @classmethod
    def available(
        cls,
        key: str,
        display_name: str,
        value: float,
        unit: str = "",
        confidence: Optional[float] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> "MetricResult":
        return cls(
            key=key,
            display_name=display_name,
            status=ResultStatus.AVAILABLE,
            value=float(value),
            unit=unit,
            confidence=confidence,
            detail=detail or {},
        )

    @classmethod
    def low_confidence(
        cls,
        key: str,
        display_name: str,
        value: float,
        reason: str,
        unit: str = "",
        confidence: Optional[float] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> "MetricResult":
        return cls(
            key=key,
            display_name=display_name,
            status=ResultStatus.LOW_CONFIDENCE,
            value=float(value),
            unit=unit,
            confidence=confidence,
            reason=reason,
            detail=detail or {},
        )

    @classmethod
    def unavailable(
        cls,
        key: str,
        display_name: str,
        status: ResultStatus,
        reason: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> "MetricResult":
        if status.is_usable:
            raise ResultError(
                f"unavailable() called with usable status {status.value}; "
                "use available() or low_confidence() instead."
            )
        return cls(
            key=key,
            display_name=display_name,
            status=status,
            reason=reason,
            detail=detail or {},
        )

    # -- display ---------------------------------------------------------
    @property
    def label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def icon(self) -> str:
        return STATUS_ICONS[self.status]

    def display_value(self, decimals: int = 1) -> str:
        """Formatted value, or an em dash. Never a zero standing in for absent."""
        if self.value is None:
            return "—"
        return f"{self.value:.{decimals}f}{(' ' + self.unit) if self.unit else ''}"

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "status": self.status.value,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "reason": self.reason,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricResult":
        return cls(
            key=data["key"],
            display_name=data.get("display_name", data["key"]),
            status=ResultStatus(data["status"]),
            value=data.get("value"),
            unit=data.get("unit", ""),
            confidence=data.get("confidence"),
            reason=data.get("reason", ""),
            detail=data.get("detail") or {},
        )
