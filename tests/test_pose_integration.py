"""Integration tests against the real MediaPipe backend and a real model file.

Everything else in the pose suite runs through a fake backend, which is what
keeps the suite fast and dependency-free. That leaves exactly one thing
unverified: whether ``MediaPipePoseBackend`` actually talks to MediaPipe
correctly — landmark count, array shapes, coordinate normalization, and the
VIDEO-mode timestamp contract.

These tests close that gap. They **skip** cleanly when MediaPipe is not
installed or no model has been downloaded, so they never block a contributor
who has neither.

The subject is a drawn figure rather than real golf footage, deliberately: no
private video enters the repository, and a synthetic subject makes the test
deterministic. It is enough to prove the plumbing is right. It is *not* a claim
about accuracy on real swings, which only real footage can establish.
"""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.config import MODELS_DIR
from golf_lab.pose.landmarks import NUM_LANDMARKS
from golf_lab.pose.mediapipe_backend import mediapipe_available
from golf_lab.pose.model_manager import get_spec, is_downloaded, model_path

pytestmark = [
    pytest.mark.skipif(
        not mediapipe_available(), reason="MediaPipe is not installed"
    ),
    pytest.mark.skipif(
        not any(is_downloaded(get_spec(key), MODELS_DIR) for key in ("lite", "full", "heavy")),
        reason="No pose model has been downloaded",
    ),
]


def _available_spec():
    for key in ("full", "lite", "heavy"):
        spec = get_spec(key)
        if is_downloaded(spec, MODELS_DIR):
            return spec
    pytest.skip("No pose model available")


def _draw_figure(phase: float) -> np.ndarray:
    """A crude but human-proportioned figure, drawn in BGR.

    MediaPipe reliably finds a pose in this, which is what makes it usable as
    a deterministic stand-in for a person.
    """
    import cv2

    image = np.full((480, 640, 3), 210, dtype=np.uint8)
    cx, cy = 320, 120
    skin = (120, 140, 190)
    shirt = (90, 70, 60)

    sway = int(20 * np.sin(phase * 2 * np.pi))
    swing = int(60 * np.sin(phase * 2 * np.pi))

    cv2.circle(image, (cx + sway // 2, cy), 34, skin, -1)
    cv2.rectangle(
        image, (cx - 45 + sway // 2, cy + 30), (cx + 45 + sway // 2, cy + 190), shirt, -1
    )
    cv2.line(image, (cx - 45 + sway // 2, cy + 55), (cx - 110, cy + 130 + swing), skin, 24)
    cv2.line(image, (cx + 45 + sway // 2, cy + 55), (cx + 110, cy + 130 - swing), skin, 24)
    cv2.line(image, (cx - 22, cy + 190), (cx - 45, cy + 330), (70, 60, 110), 30)
    cv2.line(image, (cx + 22, cy + 190), (cx + 45, cy + 330), (70, 60, 110), 30)
    cv2.ellipse(image, (cx - 50, cy + 340), (34, 14), 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(image, (cx + 50, cy + 340), (34, 14), 0, 0, 360, (40, 40, 40), -1)
    return image


@pytest.fixture(scope="module")
def real_backend():
    from golf_lab.pose.mediapipe_backend import MediaPipePoseBackend

    spec = _available_spec()
    backend = MediaPipePoseBackend(model_path=model_path(spec, MODELS_DIR))
    yield backend
    backend.close()


@pytest.fixture(scope="module")
def real_detection(real_backend):
    """One genuine detection from the real model."""
    import cv2

    for index in range(5):
        frame = _draw_figure(index / 30.0)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = real_backend.detect(rgb, index * 33)
        if result is not None:
            return result
    pytest.skip("The real model found no pose in the synthetic figure")


class TestRealBackendContract:
    def test_backend_reports_itself(self, real_backend):
        assert real_backend.name.startswith("mediapipe/")
        assert real_backend.device == "cpu"

    def test_returns_the_expected_landmark_count(self, real_detection):
        # If MediaPipe ever changed topology, every stored .npz and every
        # index constant in landmarks.py would silently mean something else.
        assert real_detection.landmarks.shape == (NUM_LANDMARKS, 3)

    def test_world_landmarks_match_the_same_topology(self, real_detection):
        assert real_detection.world_landmarks is not None
        assert real_detection.world_landmarks.shape == (NUM_LANDMARKS, 3)

    def test_confidence_arrays_are_per_landmark(self, real_detection):
        assert real_detection.visibility.shape == (NUM_LANDMARKS,)
        assert real_detection.presence.shape == (NUM_LANDMARKS,)

    def test_coordinates_are_normalized_and_finite(self, real_detection):
        # The overlay multiplies x/y by frame dimensions, so anything outside
        # roughly 0..1 would draw off-image.
        assert np.isfinite(real_detection.landmarks).all()
        assert -0.5 <= float(real_detection.landmarks[:, 0].min())
        assert float(real_detection.landmarks[:, 0].max()) <= 1.5
        assert -0.5 <= float(real_detection.landmarks[:, 1].min())
        assert float(real_detection.landmarks[:, 1].max()) <= 1.5

    def test_visibility_is_a_probability(self, real_detection):
        assert 0.0 <= float(real_detection.visibility.min())
        assert float(real_detection.visibility.max()) <= 1.0

    def test_arrays_are_float32(self, real_detection):
        # PoseSequence stores float32; a float64 result would silently upcast
        # and double every stored file.
        assert real_detection.landmarks.dtype == np.float32
        assert real_detection.visibility.dtype == np.float32


class TestRealBackendOnAVideo:
    @pytest.fixture(scope="class")
    def figure_video(self, tmp_path_factory):
        import cv2

        directory = tmp_path_factory.mktemp("figure_clip")
        path = directory / "figure.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480)
        )
        for index in range(30):
            writer.write(_draw_figure(index / 30.0))
        writer.release()
        return path

    def test_full_pipeline_produces_a_usable_sequence(self, figure_video):
        from golf_lab.pose.inference import estimate_pose_sequence
        from golf_lab.pose.mediapipe_backend import MediaPipePoseBackend
        from golf_lab.pose.smoothing import SmoothingSettings, smooth_sequence

        spec = _available_spec()
        backend = MediaPipePoseBackend(model_path=model_path(spec, MODELS_DIR))
        try:
            sequence = estimate_pose_sequence(figure_video, backend)
        finally:
            backend.close()

        assert sequence.frame_count == 30
        assert sequence.detection_rate > 0.8, (
            "the real model should find the synthetic figure in most frames"
        )
        assert sequence.mean_confidence() > 0.5

        smoothed = smooth_sequence(sequence, SmoothingSettings(window_length=7))
        assert smoothed.detected_count == sequence.detected_count
        # Raw must survive smoothing untouched.
        assert not np.array_equal(sequence.landmarks, smoothed.landmarks)

    def test_round_trip_through_storage_is_exact(self, figure_video, tmp_path):
        from golf_lab.pose.inference import estimate_pose_sequence
        from golf_lab.pose.mediapipe_backend import MediaPipePoseBackend
        from golf_lab.pose.sequence import PoseSequence

        spec = _available_spec()
        backend = MediaPipePoseBackend(model_path=model_path(spec, MODELS_DIR))
        try:
            sequence = estimate_pose_sequence(figure_video, backend)
        finally:
            backend.close()

        path = sequence.save(tmp_path / "pose_raw.npz")
        loaded = PoseSequence.load(path)

        np.testing.assert_array_equal(loaded.detected, sequence.detected)
        np.testing.assert_allclose(
            loaded.landmarks, sequence.landmarks, rtol=1e-6, equal_nan=True
        )

    def test_overlay_draws_on_a_real_detection(self, figure_video):
        import cv2

        from golf_lab.pose.inference import estimate_pose_sequence
        from golf_lab.pose.mediapipe_backend import MediaPipePoseBackend
        from golf_lab.pose.overlay import draw_sequence_frame
        from golf_lab.video.frame_reader import FrameReader

        spec = _available_spec()
        backend = MediaPipePoseBackend(model_path=model_path(spec, MODELS_DIR))
        try:
            sequence = estimate_pose_sequence(figure_video, backend)
        finally:
            backend.close()

        index = int(np.flatnonzero(sequence.detected)[0])
        with FrameReader(figure_video) as reader:
            frame = reader.read_frame(index)
            image, drawn = draw_sequence_frame(frame, sequence, index)

        assert drawn
        assert not np.array_equal(image, frame), "overlay drew nothing"
        assert image.shape == frame.shape
