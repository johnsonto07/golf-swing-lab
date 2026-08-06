"""Temporal smoothing of a pose sequence.

Landmark estimates jitter frame to frame. Smoothing makes both the overlay and
any later velocity measurement usable — a numerical derivative of raw landmark
positions is mostly noise.

Two rules this module will not break:

**Raw is never overwritten.** ``smooth_sequence`` returns a new sequence. Both
are stored per swing (``pose_raw.npz`` and ``pose_smoothed.npz``), so any
measurement can be traced back to what was actually observed.

**Smoothing never crosses a gap.** Each unbroken run of detected frames is
filtered independently. Filtering across an undetected stretch would invent
plausible-looking motion through frames where nothing was seen, and the result
would be indistinguishable from real data downstream. Undetected frames stay
undetected, and a run of detected frames too short to filter is copied through
unchanged rather than being dropped or padded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from golf_lab.logging_config import get_logger
from golf_lab.pose.sequence import PoseSequence, PoseSequenceError

logger = get_logger(__name__)

SMOOTHING_METHODS = ("savgol", "moving_average", "none")

DEFAULT_WINDOW_LENGTH = 7
DEFAULT_POLYORDER = 2


@dataclass(frozen=True)
class SmoothingSettings:
    """How to smooth. Stored alongside the result so it is reproducible."""

    method: str = "savgol"
    window_length: int = DEFAULT_WINDOW_LENGTH
    polyorder: int = DEFAULT_POLYORDER

    def __post_init__(self) -> None:
        if self.method not in SMOOTHING_METHODS:
            raise PoseSequenceError(
                f"Unknown smoothing method '{self.method}'. "
                f"Available: {', '.join(SMOOTHING_METHODS)}."
            )
        if self.window_length < 1:
            raise PoseSequenceError("window_length must be at least 1.")
        if self.method == "savgol" and self.polyorder >= self.window_length:
            raise PoseSequenceError(
                f"polyorder ({self.polyorder}) must be less than window_length "
                f"({self.window_length}) for Savitzky-Golay smoothing."
            )

    def describe(self) -> str:
        if self.method == "none":
            return "none"
        if self.method == "savgol":
            return f"savgol(window={self.window_length}, polyorder={self.polyorder})"
        return f"moving_average(window={self.window_length})"


def detected_segments(detected: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous runs of detected frames as inclusive-exclusive ``(start, stop)``.

    These are the only stretches over which smoothing is meaningful.
    """
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, is_detected in enumerate(detected):
        if is_detected and start is None:
            start = index
        elif not is_detected and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(detected)))
    return segments


def _odd_window_for(length: int, requested: int) -> int:
    """Largest usable odd window that fits inside a segment of ``length``.

    Savitzky-Golay needs an odd window no longer than the data. Shrinking to
    fit means a short segment is smoothed gently instead of being skipped.
    """
    window = min(requested, length)
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def _smooth_block(block: np.ndarray, settings: SmoothingSettings) -> np.ndarray:
    """Smooth one contiguous block along axis 0 (time)."""
    length = block.shape[0]
    if settings.method == "none" or length < 3:
        return block

    window = _odd_window_for(length, settings.window_length)
    if window < 3:
        return block

    if settings.method == "savgol":
        from scipy.signal import savgol_filter

        polyorder = min(settings.polyorder, window - 1)
        return savgol_filter(
            block, window_length=window, polyorder=polyorder, axis=0, mode="interp"
        ).astype(np.float32)

    # Moving average, edge-padded so the ends are not pulled toward zero.
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    flat = block.reshape(length, -1)
    smoothed = np.empty_like(flat, dtype=np.float64)
    for column in range(flat.shape[1]):
        padded = np.pad(flat[:, column], pad, mode="edge")
        smoothed[:, column] = np.convolve(padded, kernel, mode="valid")
    return smoothed.reshape(block.shape).astype(np.float32)


def smooth_sequence(
    raw: PoseSequence, settings: Optional[SmoothingSettings] = None
) -> PoseSequence:
    """Return a smoothed copy of ``raw``. ``raw`` is not modified."""
    settings = settings or SmoothingSettings()

    smoothed = PoseSequence(
        landmarks=raw.landmarks.copy(),
        world_landmarks=raw.world_landmarks.copy(),
        visibility=raw.visibility.copy(),
        presence=raw.presence.copy(),
        detected=raw.detected.copy(),
        fps=raw.fps,
        frame_width=raw.frame_width,
        frame_height=raw.frame_height,
        smoothing=settings.describe(),
        metadata=dict(raw.metadata),
    )

    if settings.method == "none" or raw.frame_count == 0:
        return smoothed

    segments = detected_segments(raw.detected)
    for start, stop in segments:
        smoothed.landmarks[start:stop] = _smooth_block(
            raw.landmarks[start:stop], settings
        )
        smoothed.world_landmarks[start:stop] = _smooth_block(
            raw.world_landmarks[start:stop], settings
        )

    smoothed.metadata["smoothing_segments"] = str(len(segments))
    logger.info(
        "Smoothed %d detected segment(s) with %s",
        len(segments),
        settings.describe(),
    )
    return smoothed
