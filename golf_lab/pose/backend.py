"""The pose-backend interface, kept separate from any implementation.

Two reasons this abstraction exists rather than calling MediaPipe directly
from the inference loop:

1. The whole inference pipeline — progress, cancellation, failed-frame
   bookkeeping, storage — can then be tested with a deterministic fake, with
   no model file, no network, and no MediaPipe import.
2. Milestone 9 anticipates trying other detectors. Swapping one in should not
   mean rewriting the loop that drives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from golf_lab.pose.landmarks import NUM_LANDMARKS


class PoseBackendError(RuntimeError):
    """Raised when a backend cannot be created or fails irrecoverably."""


@dataclass(frozen=True)
class PoseFrameResult:
    """One frame's detection.

    ``landmarks`` are normalized to the frame (x and y in 0..1) so a result
    stays meaningful if the image is later scaled. ``world_landmarks`` are
    MediaPipe's roughly-metric, hip-origin coordinates.
    """

    landmarks: np.ndarray  # (33, 3)
    world_landmarks: Optional[np.ndarray] = None  # (33, 3)
    visibility: Optional[np.ndarray] = None  # (33,)
    presence: Optional[np.ndarray] = None  # (33,)

    def __post_init__(self) -> None:
        if self.landmarks.shape != (NUM_LANDMARKS, 3):
            raise PoseBackendError(
                f"A backend returned landmarks of shape {self.landmarks.shape}; "
                f"expected ({NUM_LANDMARKS}, 3)."
            )


@runtime_checkable
class PoseBackend(Protocol):
    """Detects one pose per frame, in video (temporally aware) mode.

    Implementations must return ``None`` for a frame with no detectable pose
    rather than raising or inventing coordinates — a missing pose is ordinary
    (the golfer walks out of frame) and is recorded as a failed frame.
    """

    name: str

    def detect(
        self, frame_rgb: np.ndarray, timestamp_ms: int
    ) -> Optional[PoseFrameResult]:
        ...

    def close(self) -> None:
        ...
