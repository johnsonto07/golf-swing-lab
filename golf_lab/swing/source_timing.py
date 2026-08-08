"""Swing timing expressed in source-video seconds.

This is what `blocked_by_timing` was reserved for. Phase detection locates
positions as preview frame indices; turning those into durations and a tempo
ratio requires knowing when each frame was actually shown, which is what
`video.timeline` measures.

Everything here returns a :class:`MetricResult`, so an unavailable timing is
reported with the same vocabulary as any other metric — with a status and a
reason, and never with a number. The failure being guarded against is a
plausible-looking tempo ratio computed from a nominal frame rate: it would look
exactly like a real one.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from golf_lab.swing.phases import SwingPhase, SwingPhases
from golf_lab.swing.results import MetricResult, ResultStatus
from golf_lab.video.timeline import SourceTimeline, TimelineConfidence

# Tempo is the backswing:downswing ratio. Both spans are defined between
# phases this project actually detects, rather than the P-positions used in
# coaching literature that a single 2D camera cannot locate.
BACKSWING_SPAN = (SwingPhase.TAKEAWAY, SwingPhase.TOP_OF_BACKSWING)
DOWNSWING_SPAN = (SwingPhase.TOP_OF_BACKSWING, SwingPhase.IMPACT_REGION)


def _timeline_blocker(timeline: Optional[SourceTimeline]) -> Optional[str]:
    """Why this timeline cannot support timing claims, or None."""
    if timeline is None:
        return (
            "No source timeline has been measured for this swing, so preview "
            "frame numbers cannot be converted to times in the original video."
        )
    if not timeline.frames:
        return "The measured source timeline for this swing is empty."
    if not timeline.confidence.supports_durations:
        return (
            "No frame timestamps could be read from this video, so its timing "
            "is derived from a nominal frame rate. A duration computed that way "
            "would look precise and be wrong."
        )
    return None


def phase_source_seconds(
    phases: SwingPhases, timeline: Optional[SourceTimeline]
) -> Dict[SwingPhase, Optional[float]]:
    """Source-video time of each located phase, keyed by phase.

    A phase maps to ``None`` when it was not located or when its frame has no
    measured timestamp.
    """
    out: Dict[SwingPhase, Optional[float]] = {}
    for result in phases.available:
        if timeline is None or result.start_frame is None:
            out[result.phase] = None
            continue
        timing = timeline.timing_for(result.start_frame)
        out[result.phase] = (
            timing.source_seconds
            if timing is not None and timing.method.value != "nominal"
            else None
        )
    return out


def _span_frames(
    phases: SwingPhases, span: Tuple[SwingPhase, SwingPhase]
) -> Optional[Tuple[int, int]]:
    start = phases.frame_for(span[0])
    end = phases.frame_for(span[1])
    if start is None or end is None or end < start:
        return None
    return start, end


def phase_duration(
    phases: SwingPhases,
    timeline: Optional[SourceTimeline],
    span: Tuple[SwingPhase, SwingPhase],
    key: str,
    display_name: str,
) -> MetricResult:
    """Elapsed source time across a phase span, or why it is unavailable."""
    blocker = _timeline_blocker(timeline)
    if blocker:
        return MetricResult.unavailable(
            key, display_name, ResultStatus.BLOCKED_BY_TIMING, blocker
        )

    frames = _span_frames(phases, span)
    if frames is None:
        missing = [p.display_name for p in span if phases.frame_for(p) is None]
        return MetricResult.unavailable(
            key,
            display_name,
            ResultStatus.INSUFFICIENT_FRAMES,
            "Needs "
            + " and ".join(missing or [p.display_name for p in span])
            + ", which was not detected for this swing.",
        )

    assert timeline is not None
    seconds = timeline.duration_between(*frames)
    if seconds is None:
        return MetricResult.unavailable(
            key,
            display_name,
            ResultStatus.BLOCKED_BY_TIMING,
            f"Preview frames {frames[0]} and {frames[1]} do not both carry a "
            "measured timestamp, so the elapsed time between them cannot be "
            "established.",
        )
    if seconds <= 0:
        return MetricResult.unavailable(
            key,
            display_name,
            ResultStatus.DETECTION_FAILED,
            f"The measured span is {seconds:.3f}s, which is not a usable "
            "duration — the phases may be out of order.",
        )

    detail = {
        "start_frame": frames[0],
        "end_frame": frames[1],
        "timeline_confidence": timeline.confidence.value,
    }
    if timeline.confidence is TimelineConfidence.DEGRADED:
        return MetricResult.low_confidence(
            key,
            display_name,
            seconds,
            reason=(
                "Some frame timestamps in this video were interpolated rather "
                "than read, so this duration is close but not exact."
            ),
            unit="s",
            detail=detail,
        )
    return MetricResult.available(key, display_name, seconds, unit="s", detail=detail)


def backswing_duration(
    phases: SwingPhases, timeline: Optional[SourceTimeline]
) -> MetricResult:
    return phase_duration(
        phases, timeline, BACKSWING_SPAN, "backswing_duration", "Backswing duration"
    )


def downswing_duration(
    phases: SwingPhases, timeline: Optional[SourceTimeline]
) -> MetricResult:
    return phase_duration(
        phases, timeline, DOWNSWING_SPAN, "downswing_duration", "Downswing duration"
    )


def tempo_ratio(
    phases: SwingPhases, timeline: Optional[SourceTimeline]
) -> MetricResult:
    """Backswing : downswing, computed from measured source time.

    Reported as the single number golfers quote — a 3.0 means "3 to 1". It is
    only produced when both spans are themselves available, so a blocked or
    missing component propagates rather than being silently treated as zero.
    """
    key, display_name = "tempo_ratio", "Tempo (backswing : downswing)"

    backswing = backswing_duration(phases, timeline)
    downswing = downswing_duration(phases, timeline)

    for component in (backswing, downswing):
        if not component.status.is_usable:
            return MetricResult.unavailable(
                key,
                display_name,
                component.status,
                f"{component.display_name} is unavailable: {component.reason}",
            )

    assert backswing.value is not None and downswing.value is not None
    if downswing.value <= 0:
        return MetricResult.unavailable(
            key,
            display_name,
            ResultStatus.DETECTION_FAILED,
            "The downswing duration is not positive, so a ratio cannot be formed.",
        )

    ratio = backswing.value / downswing.value
    detail = {
        "backswing_seconds": round(backswing.value, 4),
        "downswing_seconds": round(downswing.value, 4),
    }

    # A degraded component makes the ratio degraded too; confidence does not
    # improve by dividing two numbers.
    if (
        backswing.status is ResultStatus.LOW_CONFIDENCE
        or downswing.status is ResultStatus.LOW_CONFIDENCE
    ):
        return MetricResult.low_confidence(
            key,
            display_name,
            ratio,
            reason=(
                "Built from durations whose frame timestamps were partly "
                "interpolated, so treat the ratio as indicative."
            ),
            unit=": 1",
            detail=detail,
        )
    return MetricResult.available(key, display_name, ratio, unit=": 1", detail=detail)


def all_source_timings(
    phases: SwingPhases, timeline: Optional[SourceTimeline]
) -> list[MetricResult]:
    """Every source-time metric, each carrying its own status."""
    return [
        backswing_duration(phases, timeline),
        downswing_duration(phases, timeline),
        tempo_ratio(phases, timeline),
    ]
