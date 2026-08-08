"""Exact-frame reading with a sequential-access fast path.

The Video Lab must satisfy a strict acceptance criterion: the frame number
shown in the UI is the frame that was actually decoded, and stepping forward
by one moves by exactly one frame. Random `CAP_PROP_POS_FRAMES` seeking is not
reliable enough on its own for that, so this reader keeps its own position
pointer and grabs forward for small jumps, only seeking for large ones.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type

import numpy as np

from golf_lab.logging_config import get_logger
from golf_lab.video.frame_cache import FrameCache, frame_cache_key

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from golf_lab.video.timeline import SourceTimeline

logger = get_logger(__name__)

# Grabbing forward is cheaper than a keyframe seek + decode for short hops,
# which is exactly what the prev/next frame buttons produce.
SEQUENTIAL_GRAB_LIMIT = 30


class FrameReadError(RuntimeError):
    """Raised when a requested frame could not be decoded."""


def timestamp_for_frame(frame_index: int, fps: float) -> float:
    """Seconds at which a 0-based frame index is presented (CFR assumption)."""
    if fps <= 0:
        return 0.0
    return frame_index / fps


def frame_for_timestamp(seconds: float, fps: float, frame_count: int) -> int:
    """Inverse of :func:`timestamp_for_frame`, clamped to the valid range."""
    if fps <= 0:
        return 0
    index = int(round(seconds * fps))
    upper = max(frame_count - 1, 0)
    return max(0, min(index, upper))


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm for display."""
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


class FrameReader:
    """Random-access frame reader over a single video file.

    Usage::

        with FrameReader(preview_path) as reader:
            frame = reader.read_frame(120)   # BGR numpy array

    Opens the file read-only; never writes to it.
    """

    def __init__(
        self,
        video_path: Path,
        cache_size: int = 24,
        timeline: Optional["SourceTimeline"] = None,
    ) -> None:
        """``timeline``, when supplied, is the authority on timing.

        Without it the reader falls back to ``index / fps``, which is correct
        only for constant-rate media and is the assumption that produced ~9%
        wrong timestamps on a file whose container misreported its rate. The
        fallback is kept for callers that have no measured timeline, and it is
        distinguishable through :attr:`timing_is_measured`.
        """
        import cv2

        self.timeline = timeline
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FrameReadError(f"Video not found: {self.video_path}")

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise FrameReadError(
                f"OpenCV could not open {self.video_path.name}. The codec may be "
                "unsupported. Use the generated preview file instead of the original."
            )

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._next_index = 0  # index of the frame the next read() will return
        self._cache = FrameCache(max_items=cache_size)

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None  # type: ignore[assignment]
        self._cache.clear()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # -- helpers --------------------------------------------------------
    @property
    def frame_count_is_measured(self) -> bool:
        """Whether the frame count came from decoded frames rather than OpenCV.

        OpenCV reports ``CAP_PROP_FRAME_COUNT``, which is the container's
        claim — the same number that said 484 for a 438-frame file.
        """
        return self.timeline is not None and self.timeline.frame_count > 0

    @property
    def last_index(self) -> int:
        if self.frame_count_is_measured:
            assert self.timeline is not None
            return max(self.timeline.frame_count - 1, 0)
        return max(self.frame_count - 1, 0)

    def clamp(self, frame_index: int) -> int:
        return max(0, min(int(frame_index), self.last_index))

    @property
    def timing_is_measured(self) -> bool:
        """Whether timestamps come from the media rather than a nominal rate."""
        return (
            self.timeline is not None
            and self.timeline.confidence.supports_durations
        )

    def timestamp_for_frame(self, frame_index: int) -> float:
        """Presentation time of a frame, measured where possible.

        Falls back to ``index / fps`` only when no measured timeline is
        available; :attr:`timing_is_measured` distinguishes the two so the UI
        never presents an assumed time as a measured one.
        """
        if self.timeline is not None:
            measured = self.timeline.source_seconds(frame_index)
            if measured is not None:
                return measured
        return timestamp_for_frame(frame_index, self.fps)

    def frame_for_timestamp(self, seconds: float) -> int:
        """Nearest frame to a timestamp, using measured timing where available."""
        if self.timeline is not None:
            index = self.timeline.preview_index_for_source_seconds(seconds)
            if index is not None:
                return index
        return frame_for_timestamp(seconds, self.fps, self.frame_count)

    # -- reading --------------------------------------------------------
    def read_frame(self, frame_index: int) -> np.ndarray:
        """Return the BGR frame at ``frame_index`` (0-based).

        Raises FrameReadError rather than silently returning a neighbouring
        frame, so a decode problem is visible instead of quietly wrong.
        """
        target = self.clamp(frame_index)

        key = frame_cache_key(self.video_path, target, kind="bgr")
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        delta = target - self._next_index
        if delta < 0 or delta > SEQUENTIAL_GRAB_LIMIT:
            self._seek(target)
        else:
            # Fast path: discard the frames between here and the target
            # without decoding them fully.
            for _ in range(delta):
                if not self._capture.grab():
                    self._seek(target)
                    break
                self._next_index += 1

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise FrameReadError(
                f"Could not decode frame {target} of {self.video_path.name}. "
                f"The file reports {self.frame_count} frames; it may be truncated."
            )
        self._next_index += 1

        self._cache.put(key, frame)
        return frame

    def read_frame_rgb(self, frame_index: int) -> np.ndarray:
        """Same as :meth:`read_frame` but in RGB order, ready for display."""
        frame = self.read_frame(frame_index)
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def _seek(self, frame_index: int) -> None:
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        self._next_index = frame_index

    def save_frame(self, frame_index: int, destination: Path) -> Path:
        """Write a single frame to disk as PNG or JPEG (by extension)."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = self.read_frame(frame_index)
        # cv2.imwrite mishandles some non-ASCII paths on Windows, so encode to
        # a buffer and write the bytes with pathlib instead.
        suffix = destination.suffix.lower() or ".png"
        ok, buffer = self._cv2.imencode(suffix, frame)
        if not ok:
            raise FrameReadError(f"Could not encode frame {frame_index} as {suffix}.")
        destination.write_bytes(buffer.tobytes())
        logger.info("Saved frame %d to %s", frame_index, destination.name)
        return destination
