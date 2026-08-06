"""The inference loop: progress, cancellation, and honest failure handling.

Driven with a fake backend against the synthetic fixture clip, so these run
without MediaPipe installed and without a downloaded model.
"""

from __future__ import annotations

import pytest

from golf_lab.pose.backend import PoseBackendError
from golf_lab.pose.inference import (
    PoseInferenceCancelled,
    estimate_pose_sequence,
    should_report_progress,
)


class TestBasicRun:
    def test_covers_every_frame(self, fixture_video, fake_backend):
        backend = fake_backend()
        sequence = estimate_pose_sequence(fixture_video, backend)

        assert sequence.frame_count == 60
        assert sequence.detected_count == 60
        assert sequence.detection_rate == 1.0
        assert len(backend.calls) == 60

    def test_records_video_properties(self, fixture_video, fake_backend):
        sequence = estimate_pose_sequence(fixture_video, fake_backend())
        assert sequence.fps == pytest.approx(30.0, abs=0.1)
        assert sequence.frame_width == 320
        assert sequence.frame_height == 240

    def test_records_provenance_metadata(self, fixture_video, fake_backend):
        sequence = estimate_pose_sequence(fixture_video, fake_backend())
        assert sequence.metadata["backend"] == "fake/test-backend"
        assert sequence.metadata["source_video"] == fixture_video.name
        assert "elapsed_seconds" in sequence.metadata

    def test_timestamps_increase_monotonically(self, fixture_video, fake_backend):
        # VIDEO-mode tracking rejects a non-increasing timestamp outright, so
        # this is a hard requirement of the loop, not a nicety.
        backend = fake_backend()
        estimate_pose_sequence(fixture_video, backend)

        assert backend.timestamps == sorted(backend.timestamps)
        assert len(set(backend.timestamps)) == len(backend.timestamps)

    def test_original_file_is_not_modified(self, fixture_video, fake_backend):
        before = fixture_video.stat().st_mtime_ns
        estimate_pose_sequence(fixture_video, fake_backend())
        assert fixture_video.stat().st_mtime_ns == before


class TestProgress:
    def test_reports_progress_and_finishes_at_one(self, fixture_video, fake_backend):
        updates = []
        estimate_pose_sequence(
            fixture_video, fake_backend(), progress=lambda f, m: updates.append((f, m))
        )

        assert updates
        assert updates[-1][0] == pytest.approx(1.0)
        fractions = [fraction for fraction, _ in updates]
        assert fractions == sorted(fractions)
        assert all(0.0 <= fraction <= 1.0 for fraction in fractions)

    def test_progress_updates_stay_bounded(self, fixture_video, fake_backend):
        # The cost being avoided is a 3000-frame clip firing 3000 UI updates.
        # The bound is ~1% steps, so short clips legitimately report every
        # frame; what matters is that the count never scales past ~100.
        updates = []
        estimate_pose_sequence(
            fixture_video, fake_backend(), progress=lambda f, m: updates.append(f)
        )
        assert len(updates) <= 101

    @pytest.mark.parametrize("frame_count", [120, 1000, 3000, 30000])
    def test_long_clips_are_throttled(self, frame_count):
        # Exercises the throttle itself rather than generating a
        # multi-thousand-frame fixture, which would dominate suite runtime.
        reported = [
            index
            for index in range(frame_count)
            if should_report_progress(index, frame_count)
        ]
        assert len(reported) <= 101
        assert reported[0] == 0
        assert reported[-1] == frame_count - 1

    def test_throttle_handles_empty_and_single_frame_clips(self):
        assert not should_report_progress(0, 0)
        assert should_report_progress(0, 1)


class TestUndetectedFrames:
    def test_missing_poses_are_marked_not_interpolated(
        self, fixture_video, fake_backend
    ):
        sequence = estimate_pose_sequence(
            fixture_video, fake_backend(fail_frames=(10, 11, 12))
        )

        assert list(sequence.failed_frames) == [10, 11, 12]
        assert sequence.detected_count == 57
        # The whole point: no values were invented for those frames.
        import numpy as np

        assert np.isnan(sequence.landmarks[10:13]).all()

    def test_run_continues_past_a_missing_pose(self, fixture_video, fake_backend):
        # A golfer stepping out of frame must not discard the rest of the swing.
        sequence = estimate_pose_sequence(fixture_video, fake_backend(fail_frames=(5,)))
        assert sequence.detected[6]
        assert sequence.detected_count == 59

    def test_scattered_backend_errors_are_tolerated(self, fixture_video, fake_backend):
        sequence = estimate_pose_sequence(
            fixture_video, fake_backend(error_frames=(3, 20, 44))
        )
        assert sequence.detected_count == 57
        assert list(sequence.failed_frames) == [3, 20, 44]

    def test_sustained_errors_abort_with_an_explanation(
        self, fixture_video, fake_backend
    ):
        # A long unbroken run of errors is structural, not per-frame; grinding
        # through thousands of identical failures helps nobody.
        backend = fake_backend(error_frames=tuple(range(60)))
        with pytest.raises(PoseBackendError) as info:
            estimate_pose_sequence(fixture_video, backend, max_consecutive_errors=5)

        assert "consecutive failures" in str(info.value)
        assert len(backend.calls) < 20


class TestCancellation:
    def test_cancel_stops_early_and_keeps_partial_work(
        self, fixture_video, fake_backend
    ):
        state = {"count": 0}

        def should_cancel():
            state["count"] += 1
            return state["count"] > 10

        backend = fake_backend()
        with pytest.raises(PoseInferenceCancelled) as info:
            estimate_pose_sequence(fixture_video, backend, should_cancel=should_cancel)

        error = info.value
        assert error.frames_done == 10
        assert error.partial.frame_count == 60
        # Frames never reached stay honestly undetected.
        assert error.partial.detected_count == 10
        assert not error.partial.detected[59]

    def test_cancelling_immediately_does_no_work(self, fixture_video, fake_backend):
        backend = fake_backend()
        with pytest.raises(PoseInferenceCancelled):
            estimate_pose_sequence(
                fixture_video, backend, should_cancel=lambda: True
            )
        assert backend.calls == []

    def test_not_cancelling_runs_to_completion(self, fixture_video, fake_backend):
        sequence = estimate_pose_sequence(
            fixture_video, fake_backend(), should_cancel=lambda: False
        )
        assert sequence.detected_count == 60


class TestBadInput:
    def test_missing_video_raises(self, tmp_path, fake_backend):
        from golf_lab.video.frame_reader import FrameReadError

        with pytest.raises(FrameReadError):
            estimate_pose_sequence(tmp_path / "nope.mp4", fake_backend())
