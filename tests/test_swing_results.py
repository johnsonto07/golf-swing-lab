"""Result status handling: the rules that stop absent data looking measured."""

from __future__ import annotations

import pytest

from golf_lab.swing.phases import PhaseResult, SwingPhase
from golf_lab.swing.results import (
    MetricResult,
    ResultError,
    ResultStatus,
    STATUS_ICONS,
    STATUS_LABELS,
)


class TestStatusVocabulary:
    def test_every_required_state_exists(self):
        # The full set the product promises to distinguish. Each names a
        # different remedy, which is the reason they are not one "unavailable".
        for name in (
            "AVAILABLE",
            "LOW_CONFIDENCE",
            "MISSING_LANDMARKS",
            "UNSUPPORTED_CAMERA_VIEW",
            "INSUFFICIENT_FRAMES",
            "BLOCKED_BY_TIMING",
            "DETECTION_FAILED",
        ):
            assert hasattr(ResultStatus, name)

    def test_every_status_has_a_label_and_icon(self):
        for status in ResultStatus:
            assert STATUS_LABELS[status]
            assert STATUS_ICONS[status]

    def test_only_available_and_low_confidence_carry_values(self):
        usable = {s for s in ResultStatus if s.is_usable}
        assert usable == {ResultStatus.AVAILABLE, ResultStatus.LOW_CONFIDENCE}


class TestMetricResultInvariants:
    def test_available_metric_carries_its_value(self):
        result = MetricResult.available("head_sway", "Head sway", 0.42, unit="sw")
        assert result.status is ResultStatus.AVAILABLE
        assert result.value == pytest.approx(0.42)
        assert result.display_value(2) == "0.42 sw"

    def test_low_confidence_still_carries_a_value(self):
        # It is a real measurement to weigh, not a missing one.
        result = MetricResult.low_confidence(
            "head_sway", "Head sway", 0.4, reason="landmarks were faint"
        )
        assert result.status.is_usable
        assert result.value is not None

    def test_unavailable_result_must_not_carry_a_number(self):
        # The whole point: a placeholder must never become a fact.
        with pytest.raises(ResultError, match="must not carry numbers"):
            MetricResult(
                key="head_sway",
                display_name="Head sway",
                status=ResultStatus.MISSING_LANDMARKS,
                value=0.0,
                reason="not visible",
            )

    def test_zero_is_rejected_just_like_any_other_value(self):
        # Zero is the tempting one: it looks like "nothing happened" but reads
        # as a measurement of no movement.
        with pytest.raises(ResultError):
            MetricResult(
                key="hip_sway",
                display_name="Hip sway",
                status=ResultStatus.DETECTION_FAILED,
                value=0.0,
                reason="failed",
            )

    def test_available_result_must_have_a_value(self):
        with pytest.raises(ResultError, match="must carry a measurement"):
            MetricResult(
                key="x", display_name="X", status=ResultStatus.AVAILABLE, value=None
            )

    def test_unavailable_result_must_explain_itself(self):
        with pytest.raises(ResultError, match="without a reason"):
            MetricResult(
                key="x", display_name="X", status=ResultStatus.INSUFFICIENT_FRAMES
            )

    def test_unavailable_helper_rejects_usable_status(self):
        with pytest.raises(ResultError):
            MetricResult.unavailable(
                "x", "X", ResultStatus.AVAILABLE, "should not be allowed"
            )

    def test_missing_value_displays_as_a_dash_not_zero(self):
        result = MetricResult.unavailable(
            "x", "X", ResultStatus.MISSING_LANDMARKS, "not visible"
        )
        assert result.display_value() == "—"

    def test_round_trip(self):
        original = MetricResult.available("spine_angle", "Spine angle", 31.5, unit="°")
        restored = MetricResult.from_dict(original.to_dict())
        assert restored == original

    def test_unavailable_round_trip_keeps_the_reason(self):
        original = MetricResult.unavailable(
            "hip_sway", "Hip sway", ResultStatus.UNSUPPORTED_CAMERA_VIEW, "wrong view"
        )
        restored = MetricResult.from_dict(original.to_dict())
        assert restored.status is ResultStatus.UNSUPPORTED_CAMERA_VIEW
        assert restored.reason == "wrong view"
        assert restored.value is None


class TestPhaseResultInvariants:
    def test_found_phase_carries_a_frame(self):
        result = PhaseResult.found(SwingPhase.ADDRESS, frame=11, confidence=0.9)
        assert result.status is ResultStatus.AVAILABLE
        assert result.start_frame == 11
        assert result.end_frame == 11

    def test_weak_detection_is_downgraded_to_low_confidence(self):
        result = PhaseResult.found(SwingPhase.ADDRESS, frame=11, confidence=0.2)
        assert result.status is ResultStatus.LOW_CONFIDENCE
        assert result.start_frame == 11
        assert result.reason

    def test_unavailable_phase_must_not_carry_a_frame(self):
        # A phase reported at frame 0 because nothing was found is
        # indistinguishable from a phase genuinely at frame 0.
        with pytest.raises(ResultError, match="must not carry frame numbers"):
            PhaseResult(
                phase=SwingPhase.ADDRESS,
                status=ResultStatus.DETECTION_FAILED,
                start_frame=0,
                reason="failed",
            )

    def test_unavailable_phase_must_explain_itself(self):
        with pytest.raises(ResultError, match="without a reason"):
            PhaseResult(phase=SwingPhase.TOP_OF_BACKSWING, status=ResultStatus.DETECTION_FAILED)

    def test_range_cannot_end_before_it_starts(self):
        with pytest.raises(ResultError, match="ends"):
            PhaseResult(
                phase=SwingPhase.DOWNSWING,
                status=ResultStatus.AVAILABLE,
                start_frame=40,
                end_frame=20,
                confidence=0.8,
            )

    def test_preview_seconds_is_named_for_the_preview_timeline(self):
        # Not source time. On VFR clips these differ; the name is the guard.
        result = PhaseResult.found(SwingPhase.ADDRESS, frame=30, confidence=0.9)
        assert result.preview_seconds(30.0) == pytest.approx(1.0)
        assert result.preview_seconds(0.0) is None

    def test_round_trip(self):
        original = PhaseResult.found(
            SwingPhase.TOP_OF_BACKSWING, frame=42, confidence=0.77
        )
        restored = PhaseResult.from_dict(original.to_dict())
        assert restored == original
