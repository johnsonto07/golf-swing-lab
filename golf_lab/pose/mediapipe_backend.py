"""MediaPipe Pose Landmarker backend.

``mediapipe`` is imported lazily inside the constructor, not at module import.
The package is a large optional dependency, and every other pose module — the
sequence container, smoothing, the overlay — must stay importable (and
testable) on a machine that has never installed it.

Runs in VIDEO mode rather than IMAGE mode. VIDEO mode carries tracking state
between frames, which both speeds up inference and keeps landmark identity
stable through a fast downswing; IMAGE mode re-detects from scratch every
frame and jitters badly. VIDEO mode requires monotonically increasing
timestamps, which the inference loop guarantees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from golf_lab.logging_config import get_logger
from golf_lab.pose.backend import PoseBackendError, PoseFrameResult
from golf_lab.pose.landmarks import NUM_LANDMARKS

logger = get_logger(__name__)


class MediaPipePoseBackend:
    """Wraps a MediaPipe ``PoseLandmarker`` running on CPU by default."""

    def __init__(
        self,
        model_path: Path,
        use_gpu: bool = False,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise PoseBackendError(
                f"Pose model file not found: {model_path}. Download it from the "
                "Swing Analysis page first."
            )

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise PoseBackendError(
                "MediaPipe is not installed, so pose estimation is unavailable.\n\n"
                'Install it with:  pip install -e ".[pose]"\n\n'
                "Everything else in the app works without it."
            ) from exc

        self._mp_vision = mp_vision
        self.name = f"mediapipe/{model_path.stem}"
        self.model_path = model_path
        self.device = "gpu" if use_gpu else "cpu"

        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if use_gpu
            else mp_python.BaseOptions.Delegate.CPU
        )
        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path), delegate=delegate
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,  # one golfer; extra detections are a distraction
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        try:
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - MediaPipe raises bare RuntimeError
            raise PoseBackendError(
                f"Could not initialise the pose model from {model_path.name}. "
                "The file may be corrupt or incomplete — try re-downloading it."
                + (
                    "\n\nGPU delegation failed; try again with GPU turned off."
                    if use_gpu
                    else ""
                )
                + f"\n\n{exc}"
            ) from exc

        self._last_timestamp_ms = -1
        logger.info("Pose backend ready: %s on %s", self.name, self.device)

    # -- detection --------------------------------------------------------
    def detect(
        self, frame_rgb: np.ndarray, timestamp_ms: int
    ) -> Optional[PoseFrameResult]:
        """Detect one pose, or return None if there is none in this frame."""
        import mediapipe as mp

        # VIDEO mode rejects a non-increasing timestamp outright. Rounding two
        # frames of a high-fps clip to the same millisecond is entirely
        # possible (240 fps is ~4.2 ms), so nudge forward instead of failing.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        try:
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:  # noqa: BLE001 - MediaPipe raises bare RuntimeError
            raise PoseBackendError(
                f"Pose detection failed at {timestamp_ms} ms: {exc}"
            ) from exc

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        if len(landmarks) != NUM_LANDMARKS:
            logger.warning(
                "Expected %d landmarks but got %d; treating frame as undetected.",
                NUM_LANDMARKS,
                len(landmarks),
            )
            return None

        coordinates = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
        )
        visibility = np.array(
            [getattr(lm, "visibility", 0.0) or 0.0 for lm in landmarks],
            dtype=np.float32,
        )
        presence = np.array(
            [getattr(lm, "presence", 0.0) or 0.0 for lm in landmarks],
            dtype=np.float32,
        )

        world = None
        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks[0]
            if len(world_landmarks) == NUM_LANDMARKS:
                world = np.array(
                    [[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32
                )

        return PoseFrameResult(
            landmarks=coordinates,
            world_landmarks=world,
            visibility=visibility,
            presence=presence,
        )

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception as exc:  # noqa: BLE001 - closing must never raise
                logger.debug("Ignoring error while closing pose backend: %s", exc)
            self._landmarker = None  # type: ignore[assignment]

    def __enter__(self) -> "MediaPipePoseBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def mediapipe_available() -> bool:
    """Whether the optional ``mediapipe`` dependency can be imported."""
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return True


def mediapipe_version() -> Optional[str]:
    try:
        import mediapipe

        return str(mediapipe.__version__)
    except (ImportError, AttributeError):
        return None
