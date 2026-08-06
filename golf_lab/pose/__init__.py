"""Pose estimation (Milestone 2).

Import order matters here: nothing at package import time may pull in
``mediapipe``. The heavy dependency is confined to
``golf_lab.pose.mediapipe_backend``, which imports it lazily inside the
constructor, so the rest of the app — and the whole test suite — works on a
machine that has neither MediaPipe nor a downloaded model.
"""

from golf_lab.pose.backend import PoseBackend, PoseBackendError, PoseFrameResult
from golf_lab.pose.inference import (
    PoseInferenceCancelled,
    estimate_pose_sequence,
)
from golf_lab.pose.model_manager import (
    DEFAULT_MODEL_KEY,
    POSE_MODELS,
    PoseModelError,
    PoseModelSpec,
    available_specs,
    download_model,
    ensure_model,
    get_spec,
    is_downloaded,
    model_path,
    verify_model,
)
from golf_lab.pose.overlay import OverlayStyle, draw_pose_on_frame, draw_sequence_frame
from golf_lab.pose.sequence import PoseSequence, PoseSequenceError
from golf_lab.pose.smoothing import SmoothingSettings, smooth_sequence

__all__ = [
    "DEFAULT_MODEL_KEY",
    "POSE_MODELS",
    "OverlayStyle",
    "PoseBackend",
    "PoseBackendError",
    "PoseFrameResult",
    "PoseInferenceCancelled",
    "PoseModelError",
    "PoseModelSpec",
    "PoseSequence",
    "PoseSequenceError",
    "SmoothingSettings",
    "available_specs",
    "download_model",
    "draw_pose_on_frame",
    "draw_sequence_frame",
    "ensure_model",
    "estimate_pose_sequence",
    "get_spec",
    "is_downloaded",
    "model_path",
    "smooth_sequence",
    "verify_model",
]
