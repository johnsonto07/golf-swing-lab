"""Swing phases and qualitative metrics (Milestone 3).

Layered so each stage consumes the previous one's output and never mutates it:

    pose inference  ->  pose_raw.npz       (MediaPipe; expensive, evidence)
    smoothing       ->  pose_smoothed.npz  (filtered copy; raw retained)
    phase detection ->  swing_analysis.json
    metric extraction   (same file)

Only the first stage needs the model. Everything here is a pure function of
stored landmarks, so re-running it is cheap and cannot damage the pose data.
"""

from golf_lab.swing.metric_registry import (
    MetricSpec,
    all_specs,
    evaluate,
    evaluate_all,
    get_spec,
    is_implemented,
    specs_for_view,
)
from golf_lab.swing.phases import (
    PHASE_ORDER,
    PHASE_SCHEMA_VERSION,
    PhaseResult,
    SwingPhase,
    SwingPhaseDetector,
    SwingPhases,
)
from golf_lab.swing.results import (
    STATUS_ICONS,
    STATUS_LABELS,
    MetricResult,
    ResultError,
    ResultStatus,
)

__all__ = [
    "PHASE_ORDER",
    "PHASE_SCHEMA_VERSION",
    "STATUS_ICONS",
    "STATUS_LABELS",
    "MetricResult",
    "MetricSpec",
    "PhaseResult",
    "ResultError",
    "ResultStatus",
    "SwingPhase",
    "SwingPhaseDetector",
    "SwingPhases",
    "all_specs",
    "evaluate",
    "evaluate_all",
    "get_spec",
    "is_implemented",
    "specs_for_view",
]
