"""Storage for derived analysis: phases and metrics.

Written to ``swing_analysis.json``, deliberately **separate** from
``pose_raw.npz`` / ``pose_smoothed.npz``. Pose inference is the expensive,
irreproducible step — it needs the model, takes tens of seconds, and its raw
output is the evidence everything else rests on. Phase detection and metric
extraction are cheap, pure functions of that evidence.

Keeping them in different files means a detector change invalidates only the
derived results. Re-running phase detection never touches MediaPipe, and can
never overwrite or degrade the landmarks it was computed from.

Staleness is therefore two-level: derived results go stale when the detector
version, the schema, or the *pose analysis they were computed from* changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from golf_lab.config import ANALYSIS_VERSION, APP_VERSION
from golf_lab.logging_config import get_logger
from golf_lab.storage.file_repository import swing_dir
from golf_lab.swing.phases import PHASE_SCHEMA_VERSION, SwingPhases
from golf_lab.swing.results import MetricResult

logger = get_logger(__name__)

ANALYSIS_FILENAME = "swing_analysis.json"


@dataclass
class SwingAnalysis:
    """Derived phases and metrics for one swing, plus what produced them."""

    swing_id: str
    phases: SwingPhases
    metrics: List[MetricResult] = field(default_factory=list)

    # Ties this result to the pose analysis it was derived from. If that is
    # recomputed, these are stale even though nothing here changed.
    pose_created_at: str = ""
    pose_video_fingerprint: str = ""

    schema_version: int = PHASE_SCHEMA_VERSION
    analysis_version: str = ANALYSIS_VERSION
    app_version: str = APP_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def metric(self, key: str) -> Optional[MetricResult]:
        return next((m for m in self.metrics if m.key == key), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "swing_id": self.swing_id,
            "schema_version": self.schema_version,
            "analysis_version": self.analysis_version,
            "app_version": self.app_version,
            "created_at": self.created_at,
            "pose_created_at": self.pose_created_at,
            "pose_video_fingerprint": self.pose_video_fingerprint,
            "phases": self.phases.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwingAnalysis":
        return cls(
            swing_id=data["swing_id"],
            phases=SwingPhases.from_dict(data.get("phases") or {}),
            metrics=[MetricResult.from_dict(m) for m in data.get("metrics") or []],
            pose_created_at=data.get("pose_created_at", ""),
            pose_video_fingerprint=data.get("pose_video_fingerprint", ""),
            schema_version=int(data.get("schema_version", 0)),
            analysis_version=data.get("analysis_version", "0"),
            app_version=data.get("app_version", "0"),
            created_at=data.get("created_at", ""),
        )


def analysis_path(swing_id: str, root: Optional[Path] = None) -> Path:
    return swing_dir(swing_id, root) / ANALYSIS_FILENAME


def has_analysis(swing_id: str, root: Optional[Path] = None) -> bool:
    return analysis_path(swing_id, root).exists()


def save_analysis(
    analysis: SwingAnalysis, root: Optional[Path] = None
) -> Path:
    """Write derived results atomically. Never touches the pose artifacts."""
    path = analysis_path(analysis.swing_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(analysis.to_dict(), indent=2), encoding="utf-8"
    )
    temp_path.replace(path)
    logger.info(
        "Saved derived analysis for %s (%d phases located, %d metrics)",
        analysis.swing_id,
        len(analysis.phases.available),
        len(analysis.metrics),
    )
    return path


def load_analysis(
    swing_id: str, root: Optional[Path] = None
) -> Optional[SwingAnalysis]:
    path = analysis_path(swing_id, root)
    if not path.exists():
        return None
    try:
        return SwingAnalysis.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Unreadable derived analysis for %s: %s", swing_id, exc)
        return None


def delete_analysis(swing_id: str, root: Optional[Path] = None) -> None:
    """Remove derived results. Pose data and video are untouched."""
    try:
        analysis_path(swing_id, root).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete derived analysis for %s: %s", swing_id, exc)


def staleness_reasons(
    analysis: Optional[SwingAnalysis],
    pose_info: Optional[Any] = None,
    detector_name: Optional[str] = None,
    detector_version: Optional[str] = None,
) -> List[str]:
    """Why derived results should not be trusted, as user-facing sentences.

    Note what is *not* checked here: the video. That is the pose layer's job,
    and if the video changed the pose analysis is already stale, which this
    detects through ``pose_created_at``. Checking it again here would produce
    two warnings for one cause.
    """
    if analysis is None:
        return ["Swing phases have not been detected for this swing yet."]

    reasons: List[str] = []

    if analysis.schema_version != PHASE_SCHEMA_VERSION:
        reasons.append(
            f"It was stored with phase schema v{analysis.schema_version}; this "
            f"build uses v{PHASE_SCHEMA_VERSION}."
        )
    if analysis.analysis_version != ANALYSIS_VERSION:
        reasons.append(
            f"It was computed with analysis version {analysis.analysis_version}; "
            f"this build uses version {ANALYSIS_VERSION}."
        )
    if detector_name and analysis.phases.detector_name != detector_name:
        reasons.append(
            f"It was produced by the '{analysis.phases.detector_name}' detector; "
            f"this build uses '{detector_name}'."
        )
    if detector_version and analysis.phases.detector_version != detector_version:
        reasons.append(
            f"It was produced by detector version "
            f"{analysis.phases.detector_version}; this build uses "
            f"{detector_version}."
        )
    if pose_info is not None:
        pose_created = getattr(pose_info, "created_at", "")
        if (
            analysis.pose_created_at
            and pose_created
            and analysis.pose_created_at != pose_created
        ):
            reasons.append(
                "The pose analysis it was derived from has been recomputed."
            )
    return reasons


def is_stale(
    analysis: Optional[SwingAnalysis],
    pose_info: Optional[Any] = None,
    detector_name: Optional[str] = None,
    detector_version: Optional[str] = None,
) -> bool:
    return bool(
        staleness_reasons(analysis, pose_info, detector_name, detector_version)
    )
