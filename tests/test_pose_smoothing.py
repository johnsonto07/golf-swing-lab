"""Smoothing: reduces jitter, keeps raw intact, never invents motion."""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.pose.landmarks import NUM_LANDMARKS
from golf_lab.pose.sequence import PoseSequence, PoseSequenceError
from golf_lab.pose.smoothing import (
    SmoothingSettings,
    detected_segments,
    smooth_sequence,
)


def _jitter_magnitude(sequence: PoseSequence) -> float:
    """Mean absolute second difference over detected frames: pure jitter.

    A steady drift has a near-zero second difference, so this measures the
    shake without punishing genuine motion.
    """
    detected = sequence.landmarks[np.asarray(sequence.detected)]
    if len(detected) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(detected, n=2, axis=0))))


class TestSettings:
    def test_rejects_unknown_method(self):
        with pytest.raises(PoseSequenceError):
            SmoothingSettings(method="magic")

    def test_rejects_polyorder_at_or_above_window(self):
        with pytest.raises(PoseSequenceError):
            SmoothingSettings(method="savgol", window_length=5, polyorder=5)

    def test_describe_is_reproducible(self):
        settings = SmoothingSettings(method="savgol", window_length=9, polyorder=3)
        assert settings.describe() == "savgol(window=9, polyorder=3)"
        assert SmoothingSettings(method="none").describe() == "none"


class TestDetectedSegments:
    def test_splits_on_gaps(self):
        detected = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
        assert detected_segments(detected) == [(0, 2), (3, 6), (8, 9)]

    def test_all_detected_is_one_segment(self):
        assert detected_segments(np.ones(5, dtype=bool)) == [(0, 5)]

    def test_none_detected_is_no_segments(self):
        assert detected_segments(np.zeros(5, dtype=bool)) == []

    def test_empty_input(self):
        assert detected_segments(np.zeros(0, dtype=bool)) == []


class TestSmoothingBehaviour:
    def test_reduces_jitter(self, pose_sequence_factory):
        raw = pose_sequence_factory(frame_count=60, jitter=0.02)
        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=9))
        assert _jitter_magnitude(smoothed) < _jitter_magnitude(raw) * 0.6

    def test_raw_is_never_modified(self, pose_sequence_factory):
        raw = pose_sequence_factory(frame_count=40, jitter=0.02)
        before = raw.landmarks.copy()

        smooth_sequence(raw, SmoothingSettings(window_length=9))

        np.testing.assert_array_equal(raw.landmarks, before)

    def test_preserves_the_underlying_trend(self, pose_sequence_factory):
        # Smoothing must remove shake without dragging the body off its path.
        raw = pose_sequence_factory(frame_count=60, jitter=0.01)
        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=9))

        difference = np.abs(smoothed.landmarks - raw.landmarks)
        assert np.nanmax(difference) < 0.05

    def test_method_none_is_a_faithful_copy(self, pose_sequence_factory):
        raw = pose_sequence_factory(frame_count=20, jitter=0.02)
        smoothed = smooth_sequence(raw, SmoothingSettings(method="none"))
        np.testing.assert_array_equal(smoothed.landmarks, raw.landmarks)
        assert smoothed.smoothing == "none"

    def test_moving_average_also_smooths(self, pose_sequence_factory):
        raw = pose_sequence_factory(frame_count=60, jitter=0.02)
        smoothed = smooth_sequence(
            raw, SmoothingSettings(method="moving_average", window_length=9)
        )
        assert _jitter_magnitude(smoothed) < _jitter_magnitude(raw)

    def test_records_the_settings_used(self, pose_sequence_factory):
        raw = pose_sequence_factory(frame_count=20)
        smoothed = smooth_sequence(
            raw, SmoothingSettings(window_length=7, polyorder=2)
        )
        assert smoothed.smoothing == "savgol(window=7, polyorder=2)"


class TestSmoothingRespectsGaps:
    def test_undetected_frames_stay_undetected(self, pose_sequence_factory):
        raw = pose_sequence_factory(frame_count=40, failed=(10, 11, 12), jitter=0.01)
        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=9))

        np.testing.assert_array_equal(smoothed.detected, raw.detected)
        assert np.isnan(smoothed.landmarks[10]).all()
        assert np.isnan(smoothed.landmarks[11]).all()
        assert np.isnan(smoothed.landmarks[12]).all()

    def test_gap_never_contaminates_neighbouring_values(self, pose_sequence_factory):
        # The real hazard: a NaN dragged into the filter window would turn
        # every nearby frame into NaN and quietly destroy good data.
        raw = pose_sequence_factory(frame_count=40, failed=(20,), jitter=0.01)
        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=9))

        detected_values = smoothed.landmarks[np.asarray(smoothed.detected)]
        assert np.isfinite(detected_values).all()

    def test_segments_are_smoothed_independently(self, pose_sequence_factory):
        # Frame 21 sits right after a gap. If smoothing crossed the gap it
        # would be pulled toward pre-gap values; independent segments mean it
        # is only influenced by frames on its own side.
        raw = pose_sequence_factory(frame_count=60, failed=tuple(range(18, 21)))
        raw.landmarks[21:] += 0.3  # a hard step change after the gap

        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=9))

        assert smoothed.landmarks[21, 0, 0] > raw.landmarks[17, 0, 0] + 0.2

    def test_short_segments_pass_through_unchanged(self, pose_sequence_factory):
        # Two detected frames cannot support a 9-wide filter; they must be
        # copied, not dropped and not padded with invented values.
        raw = pose_sequence_factory(
            frame_count=10, failed=(0, 1, 2, 5, 6, 7, 8, 9)
        )
        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=9))

        np.testing.assert_allclose(
            smoothed.landmarks[3:5], raw.landmarks[3:5], rtol=1e-6
        )
        assert smoothed.detected_count == 2

    def test_all_frames_failed_is_handled(self):
        raw = PoseSequence.empty(10, fps=30.0, frame_width=100, frame_height=100)
        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=7))
        assert smoothed.detected_count == 0
        assert np.isnan(smoothed.landmarks).all()

    def test_empty_sequence_is_handled(self):
        smoothed = smooth_sequence(PoseSequence.empty(0), SmoothingSettings())
        assert smoothed.frame_count == 0

    def test_window_larger_than_clip_still_works(self, pose_sequence_factory):
        raw = pose_sequence_factory(frame_count=5, jitter=0.01)
        smoothed = smooth_sequence(raw, SmoothingSettings(window_length=31))
        assert smoothed.detected_count == 5
        assert np.isfinite(smoothed.landmarks).all()
        assert smoothed.landmarks.shape == (5, NUM_LANDMARKS, 3)
