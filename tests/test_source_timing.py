"""Source-time durations and tempo — and refusing them when timing is unknown.

Tempo is the number this project has most carefully avoided fabricating. A
ratio computed from a nominal frame rate looks exactly like a real one, so the
tests below spend most of their attention on the paths where no number should
appear at all.
"""

from __future__ import annotations

import pytest

from golf_lab.models.video import CameraView
from golf_lab.swing.phases import PhaseResult, SwingPhase, SwingPhases
from golf_lab.swing.results import ResultStatus
from golf_lab.swing.source_timing import (
    all_source_timings,
    backswing_duration,
    downswing_duration,
    phase_source_seconds,
    tempo_ratio,
)
from golf_lab.video.timeline import (
    FrameTiming,
    SourceTimeline,
    TimelineConfidence,
    TimingMethod,
)


def _timeline(count=100, fps=30.0, method=TimingMethod.MEASURED,
              confidence=TimelineConfidence.MEASURED):
    frames = [
        FrameTiming(
            preview_index=i,
            source_seconds=i / fps,
            method=method,
            source_frame_index=i,
        )
        for i in range(count)
    ]
    return SourceTimeline(frames=frames, confidence=confidence, nominal_fps=fps,
                          measured_fps=fps, is_constant_rate=True)


def _phases(takeaway=10, top=40, impact=52):
    """A swing with a 1.0 s backswing and 0.4 s downswing at 30 fps."""
    phases = SwingPhases(camera_view=CameraView.FACE_ON, preview_fps=30.0)
    phases.set(PhaseResult.found(SwingPhase.ADDRESS, frame=9, confidence=0.9))
    if takeaway is not None:
        phases.set(PhaseResult.found(SwingPhase.TAKEAWAY, frame=takeaway, confidence=0.9))
    if top is not None:
        phases.set(PhaseResult.found(SwingPhase.TOP_OF_BACKSWING, frame=top, confidence=0.9))
    if impact is not None:
        phases.set(
            PhaseResult.found(
                SwingPhase.IMPACT_REGION, frame=impact, end_frame=impact + 4, confidence=0.9
            )
        )
    return phases


class TestPhaseSourceTimes:
    def test_phases_map_to_source_seconds(self):
        times = phase_source_seconds(_phases(), _timeline())
        assert times[SwingPhase.TAKEAWAY] == pytest.approx(10 / 30.0)
        assert times[SwingPhase.TOP_OF_BACKSWING] == pytest.approx(40 / 30.0)

    def test_no_timeline_yields_no_times(self):
        times = phase_source_seconds(_phases(), None)
        assert all(v is None for v in times.values())

    def test_nominal_frames_yield_no_times(self):
        # A nominal timestamp is an assumption, not a measurement.
        timeline = _timeline(method=TimingMethod.NOMINAL,
                             confidence=TimelineConfidence.NOMINAL)
        times = phase_source_seconds(_phases(), timeline)
        assert all(v is None for v in times.values())


class TestDurations:
    def test_backswing_duration_is_measured(self):
        result = backswing_duration(_phases(), _timeline())
        assert result.status is ResultStatus.AVAILABLE
        assert result.value == pytest.approx(1.0, abs=0.01)
        assert result.unit == "s"

    def test_downswing_duration_is_measured(self):
        result = downswing_duration(_phases(), _timeline())
        assert result.status is ResultStatus.AVAILABLE
        assert result.value == pytest.approx(0.4, abs=0.01)

    def test_duration_uses_real_timestamps_on_a_variable_timeline(self):
        # Uneven frames: index/fps would give the wrong answer.
        frames = []
        t = 0.0
        for i in range(60):
            frames.append(
                FrameTiming(preview_index=i, source_seconds=t, method=TimingMethod.MEASURED)
            )
            t += 0.02 if i < 30 else 0.10
        timeline = SourceTimeline(frames=frames, confidence=TimelineConfidence.MEASURED)

        result = backswing_duration(_phases(takeaway=0, top=40), timeline)
        assert result.status.is_usable
        # 30 gaps of 0.02 then 10 of 0.10 = 0.6 + 1.0
        assert result.value == pytest.approx(1.6, abs=0.01)
        assert result.value != pytest.approx(40 / 30.0, rel=0.05)

    def test_missing_phase_reports_insufficient_frames_not_zero(self):
        result = backswing_duration(_phases(takeaway=None), _timeline())
        assert result.status is ResultStatus.INSUFFICIENT_FRAMES
        assert result.value is None
        assert "Takeaway" in result.reason

    def test_degraded_timeline_reports_low_confidence_with_a_value(self):
        timeline = _timeline(confidence=TimelineConfidence.DEGRADED)
        result = backswing_duration(_phases(), timeline)
        assert result.status is ResultStatus.LOW_CONFIDENCE
        assert result.value is not None
        assert "interpolated" in result.reason


class TestRefusalWithoutMeasuredTiming:
    def test_no_timeline_is_blocked_by_timing(self):
        result = backswing_duration(_phases(), None)
        assert result.status is ResultStatus.BLOCKED_BY_TIMING
        assert result.value is None

    def test_nominal_timeline_is_blocked_and_says_why(self):
        timeline = _timeline(method=TimingMethod.NOMINAL,
                             confidence=TimelineConfidence.NOMINAL)
        result = backswing_duration(_phases(), timeline)

        assert result.status is ResultStatus.BLOCKED_BY_TIMING
        assert result.value is None
        assert "nominal frame rate" in result.reason
        assert "look precise and be wrong" in result.reason

    def test_empty_timeline_is_blocked(self):
        result = backswing_duration(_phases(), SourceTimeline())
        assert result.status is ResultStatus.BLOCKED_BY_TIMING
        assert result.value is None


class TestTempo:
    def test_tempo_is_the_ratio_of_measured_spans(self):
        result = tempo_ratio(_phases(), _timeline())
        assert result.status is ResultStatus.AVAILABLE
        # 1.0 s backswing / 0.4 s downswing
        assert result.value == pytest.approx(2.5, abs=0.05)
        assert result.unit == ": 1"
        assert result.detail["backswing_seconds"] == pytest.approx(1.0, abs=0.01)

    def test_tempo_is_blocked_without_measured_timing(self):
        # The headline guard: no timeline, no ratio, no number.
        result = tempo_ratio(_phases(), None)
        assert result.status is ResultStatus.BLOCKED_BY_TIMING
        assert result.value is None

    def test_tempo_is_blocked_on_a_nominal_timeline(self):
        timeline = _timeline(method=TimingMethod.NOMINAL,
                             confidence=TimelineConfidence.NOMINAL)
        result = tempo_ratio(_phases(), timeline)
        assert result.status is ResultStatus.BLOCKED_BY_TIMING
        assert result.value is None

    def test_missing_impact_propagates_rather_than_defaulting(self):
        result = tempo_ratio(_phases(impact=None), _timeline())
        assert result.status is ResultStatus.INSUFFICIENT_FRAMES
        assert result.value is None
        assert "Downswing duration is unavailable" in result.reason

    def test_degraded_components_make_the_ratio_degraded(self):
        # Dividing two uncertain numbers does not produce a certain one.
        timeline = _timeline(confidence=TimelineConfidence.DEGRADED)
        result = tempo_ratio(_phases(), timeline)
        assert result.status is ResultStatus.LOW_CONFIDENCE
        assert result.value is not None

    def test_out_of_order_phases_do_not_produce_a_ratio(self):
        # top before takeaway: nonsense input must not yield a number.
        result = tempo_ratio(_phases(takeaway=40, top=10), _timeline())
        assert not result.status.is_usable
        assert result.value is None


class TestAllSourceTimings:
    def test_returns_every_metric_with_a_status(self):
        results = all_source_timings(_phases(), _timeline())
        keys = {r.key for r in results}
        assert keys == {"backswing_duration", "downswing_duration", "tempo_ratio"}
        for result in results:
            assert result.status in set(ResultStatus)

    def test_none_carry_a_value_when_blocked(self):
        for result in all_source_timings(_phases(), None):
            assert result.value is None
            assert result.status is ResultStatus.BLOCKED_BY_TIMING
            assert result.reason
