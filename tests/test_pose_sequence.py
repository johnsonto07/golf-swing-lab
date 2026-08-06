"""PoseSequence: the on-disk contract and the failed-frame guarantees."""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.pose.landmarks import (
    KEY_SWING_LANDMARKS,
    LANDMARK_NAMES,
    LEFT_WRIST,
    NUM_LANDMARKS,
    POSE_CONNECTIONS,
    landmark_name,
)
from golf_lab.pose.sequence import PoseSequence, PoseSequenceError


class TestLandmarkTopology:
    def test_names_cover_every_landmark(self):
        assert len(LANDMARK_NAMES) == NUM_LANDMARKS
        assert len(set(LANDMARK_NAMES)) == NUM_LANDMARKS

    def test_connections_reference_valid_landmarks(self):
        for start, end in POSE_CONNECTIONS:
            assert 0 <= start < NUM_LANDMARKS
            assert 0 <= end < NUM_LANDMARKS
            assert start != end

    def test_no_duplicate_connections(self):
        normalized = {tuple(sorted(edge)) for edge in POSE_CONNECTIONS}
        assert len(normalized) == len(POSE_CONNECTIONS)

    def test_key_swing_landmarks_exclude_the_face(self):
        # The headline confidence number must not be propped up by face
        # landmarks that say nothing about a swing.
        for index in KEY_SWING_LANDMARKS:
            assert "eye" not in LANDMARK_NAMES[index]
            assert "mouth" not in LANDMARK_NAMES[index]
            assert "ear" not in LANDMARK_NAMES[index]

    def test_landmark_name_rejects_out_of_range(self):
        assert landmark_name(LEFT_WRIST) == "left_wrist"
        with pytest.raises(IndexError):
            landmark_name(NUM_LANDMARKS)


class TestEmptySequence:
    def test_starts_fully_undetected_with_nan(self):
        sequence = PoseSequence.empty(10, fps=30.0, frame_width=320, frame_height=240)
        assert sequence.frame_count == 10
        assert sequence.detected_count == 0
        assert sequence.detection_rate == 0.0
        # NaN, not zero: zero is a legitimate coordinate.
        assert np.isnan(sequence.landmarks).all()

    def test_rejects_negative_frame_count(self):
        with pytest.raises(PoseSequenceError):
            PoseSequence.empty(-1)

    def test_zero_frames_is_allowed(self):
        sequence = PoseSequence.empty(0)
        assert sequence.frame_count == 0
        assert sequence.detection_rate == 0.0


class TestSetAndFail:
    def test_set_frame_marks_detected(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=5)
        assert sequence.detected_count == 5
        assert sequence.detection_rate == 1.0
        assert not np.isnan(sequence.landmarks).any()

    def test_mark_failed_restores_nan_not_zero(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=5)
        sequence.mark_failed(2)

        assert not sequence.detected[2]
        assert np.isnan(sequence.landmarks[2]).all()
        assert sequence.visibility[2].sum() == 0.0
        assert sequence.detected_count == 4

    def test_failed_frames_are_reported(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=10, failed=(3, 4, 7))
        assert list(sequence.failed_frames) == [3, 4, 7]
        assert sequence.detection_rate == pytest.approx(0.7)

    def test_out_of_range_writes_are_rejected(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=3)
        with pytest.raises(PoseSequenceError):
            sequence.mark_failed(5)
        with pytest.raises(PoseSequenceError):
            sequence.set_frame(5, np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))

    def test_wrong_landmark_shape_is_rejected(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=3)
        with pytest.raises(PoseSequenceError):
            sequence.set_frame(0, np.zeros((10, 3), dtype=np.float32))

    def test_mismatched_arrays_are_rejected(self):
        with pytest.raises(PoseSequenceError):
            PoseSequence(
                landmarks=np.zeros((5, NUM_LANDMARKS, 3), dtype=np.float32),
                world_landmarks=np.zeros((5, NUM_LANDMARKS, 3), dtype=np.float32),
                visibility=np.zeros((5, NUM_LANDMARKS), dtype=np.float32),
                presence=np.zeros((5, NUM_LANDMARKS), dtype=np.float32),
                detected=np.zeros(4, dtype=bool),  # disagrees with the rest
            )


class TestPixelCoordinates:
    def test_scales_normalized_coordinates(self):
        sequence = PoseSequence.empty(1, frame_width=200, frame_height=100)
        landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        landmarks[:, 0] = 0.5
        landmarks[:, 1] = 0.25
        sequence.set_frame(0, landmarks)

        points = sequence.pixel_coordinates(0)
        assert points is not None
        assert points[0][0] == pytest.approx(100.0)
        assert points[0][1] == pytest.approx(25.0)

    def test_undetected_frame_returns_none_not_nans(self, pose_sequence_factory):
        # Callers must be forced to handle "no pose" rather than silently
        # drawing NaNs.
        sequence = pose_sequence_factory(frame_count=3, failed=(1,))
        assert sequence.pixel_coordinates(1) is None

    def test_unknown_dimensions_raise(self):
        sequence = PoseSequence.empty(1)
        sequence.set_frame(0, np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
        with pytest.raises(PoseSequenceError):
            sequence.pixel_coordinates(0)


class TestConfidence:
    def test_undetected_frame_has_zero_confidence(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=4, failed=(2,))
        assert sequence.confidence_for_frame(2) == 0.0
        assert sequence.confidence_for_frame(0) == pytest.approx(0.8)

    def test_mean_confidence_excludes_undetected_frames(self, pose_sequence_factory):
        # Undetected frames are already reported by detection_rate; counting
        # them as zero confidence here would double-penalise and conflate
        # "never seen" with "seen poorly".
        sequence = pose_sequence_factory(frame_count=10, failed=(0, 1, 2, 3, 4))
        assert sequence.mean_confidence() == pytest.approx(0.8)
        assert sequence.detection_rate == pytest.approx(0.5)

    def test_mean_confidence_of_empty_sequence_is_zero(self):
        assert PoseSequence.empty(5).mean_confidence() == 0.0


class TestLongestGap:
    def test_finds_longest_run(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=12, failed=(1, 5, 6, 7, 10))
        start, length = sequence.longest_gap()
        assert (start, length) == (5, 3)

    def test_no_gap_when_all_detected(self, pose_sequence_factory):
        assert pose_sequence_factory(frame_count=5).longest_gap() == (0, 0)

    def test_gap_running_to_the_end(self, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=6, failed=(4, 5))
        assert sequence.longest_gap() == (4, 2)


class TestPersistence:
    def test_round_trip_preserves_everything(self, tmp_path, pose_sequence_factory):
        original = pose_sequence_factory(frame_count=15, failed=(3, 9))
        original.metadata["backend"] = "fake/test-backend"
        original.smoothing = "savgol(window=7, polyorder=2)"

        path = original.save(tmp_path / "pose_raw.npz")
        loaded = PoseSequence.load(path)

        assert loaded.frame_count == original.frame_count
        assert loaded.fps == original.fps
        assert loaded.frame_width == original.frame_width
        assert loaded.smoothing == original.smoothing
        assert loaded.metadata["backend"] == "fake/test-backend"
        np.testing.assert_array_equal(loaded.detected, original.detected)
        np.testing.assert_allclose(
            loaded.visibility, original.visibility, rtol=1e-6
        )

    def test_nan_survives_the_round_trip(self, tmp_path, pose_sequence_factory):
        # If NaN came back as 0.0, a failed frame would silently become a
        # skeleton collapsed into the image corner.
        original = pose_sequence_factory(frame_count=6, failed=(2,))
        loaded = PoseSequence.load(original.save(tmp_path / "pose.npz"))

        assert np.isnan(loaded.landmarks[2]).all()
        assert not loaded.detected[2]
        assert loaded.pixel_coordinates(2) is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(PoseSequenceError):
            PoseSequence.load(tmp_path / "nope.npz")

    def test_corrupt_file_raises_actionable_error(self, tmp_path):
        path = tmp_path / "broken.npz"
        path.write_bytes(b"this is not an npz archive")
        with pytest.raises(PoseSequenceError) as info:
            PoseSequence.load(path)
        assert "re-running pose estimation" in str(info.value).lower()

    def test_future_format_version_is_refused(self, tmp_path, pose_sequence_factory):
        # Reading a newer layout with older code would misinterpret the arrays
        # rather than fail, which is worse than refusing.
        original = pose_sequence_factory(frame_count=4)
        path = original.save(tmp_path / "pose.npz")

        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        payload["format_version"] = np.array(999)
        np.savez_compressed(path, **payload)

        with pytest.raises(PoseSequenceError) as info:
            PoseSequence.load(path)
        assert "999" in str(info.value)

    def test_save_is_atomic_leaving_no_temp_file(self, tmp_path, pose_sequence_factory):
        path = pose_sequence_factory(frame_count=3).save(tmp_path / "pose.npz")
        assert path.exists()
        assert not list(tmp_path.glob("*.tmp"))
