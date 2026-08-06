"""Drive a pose backend across every frame of a video.

Deliberately knows nothing about MediaPipe or Streamlit. It takes a backend
and a video path, and returns a :class:`PoseSequence`. That is what makes the
whole pipeline — progress, cancellation, failure bookkeeping — testable with a
fake backend and a two-second synthetic clip.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from golf_lab.logging_config import get_logger
from golf_lab.pose.backend import PoseBackend, PoseBackendError
from golf_lab.pose.sequence import PoseSequence
from golf_lab.video.frame_reader import FrameReader

logger = get_logger(__name__)

# (fraction_complete, message) -> None
ProgressCallback = Callable[[float, str], None]
# Returns True to stop early. Polled once per frame.
CancelCallback = Callable[[], bool]


def should_report_progress(index: int, frame_count: int) -> bool:
    """Whether frame ``index`` should trigger a progress update.

    Roughly 1% steps. A 3000-frame clip firing 3000 Streamlit updates spends
    real time redrawing instead of decoding, so the update count is capped at
    ~101 rather than tracking frame count. Short clips fall out of this
    reporting every frame, which is already cheap.

    The step rounds *up* deliberately: integer division would give a step of 1
    for anything under 200 frames, so a 199-frame clip would still fire 199
    updates and the cap would not actually hold.

    The last frame always reports so the bar reliably reaches 100%.
    """
    if frame_count <= 0:
        return False
    step = max(1, -(-frame_count // 100))  # ceiling division
    return index % step == 0 or index == frame_count - 1


class PoseInferenceCancelled(RuntimeError):
    """Raised when the caller asked to stop before the run finished."""

    def __init__(self, partial: PoseSequence, frames_done: int) -> None:
        super().__init__(
            f"Pose estimation cancelled after {frames_done} of "
            f"{partial.frame_count} frames."
        )
        self.partial = partial
        self.frames_done = frames_done


def estimate_pose_sequence(
    video_path: Path,
    backend: PoseBackend,
    progress: Optional[ProgressCallback] = None,
    should_cancel: Optional[CancelCallback] = None,
    max_consecutive_errors: int = 30,
) -> PoseSequence:
    """Run ``backend`` over every frame of ``video_path``.

    Frames the backend cannot resolve are marked failed and the run continues:
    a golfer stepping briefly out of frame should not throw away the rest of
    the swing. A long unbroken run of *errors* (as opposed to honest
    non-detections) is different — that means something is structurally wrong,
    so the run stops instead of grinding through thousands of identical
    failures.

    The video is opened read-only and never modified.
    """
    report = progress or (lambda fraction, message: None)
    video_path = Path(video_path)

    with FrameReader(video_path) as reader:
        frame_count = reader.frame_count
        if frame_count <= 0:
            raise PoseBackendError(
                f"{video_path.name} reports no frames, so there is nothing to "
                "analyse. The preview may be corrupt; try re-importing the swing."
            )

        sequence = PoseSequence.empty(
            frame_count=frame_count,
            fps=reader.fps,
            frame_width=reader.width,
            frame_height=reader.height,
        )
        sequence.metadata["backend"] = getattr(backend, "name", "unknown")
        sequence.metadata["source_video"] = video_path.name

        started = time.monotonic()
        consecutive_errors = 0
        last_error: Optional[Exception] = None

        for index in range(frame_count):
            if should_cancel is not None and should_cancel():
                logger.info("Pose estimation cancelled at frame %d", index)
                sequence.metadata["cancelled_at_frame"] = str(index)
                raise PoseInferenceCancelled(sequence, index)

            try:
                frame_rgb = reader.read_frame_rgb(index)
            except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the run
                sequence.mark_failed(index)
                consecutive_errors += 1
                last_error = exc
                logger.warning("Could not read frame %d: %s", index, exc)
                if consecutive_errors >= max_consecutive_errors:
                    break
                continue

            # Timestamps must increase monotonically for VIDEO-mode tracking.
            timestamp_ms = int(round(reader.timestamp_for_frame(index) * 1000))

            try:
                result = backend.detect(frame_rgb, timestamp_ms)
                consecutive_errors = 0
            except PoseBackendError as exc:
                sequence.mark_failed(index)
                consecutive_errors += 1
                last_error = exc
                logger.warning("Pose backend failed on frame %d: %s", index, exc)
                if consecutive_errors >= max_consecutive_errors:
                    break
                continue

            if result is None:
                # An honest "no golfer visible here", not an error.
                sequence.mark_failed(index)
            else:
                sequence.set_frame(
                    index,
                    landmarks=result.landmarks,
                    world_landmarks=result.world_landmarks,
                    visibility=result.visibility,
                    presence=result.presence,
                )

            if should_report_progress(index, frame_count):
                report(
                    (index + 1) / frame_count,
                    f"Analysing frame {index + 1} of {frame_count}",
                )

        if consecutive_errors >= max_consecutive_errors:
            raise PoseBackendError(
                f"Pose estimation stopped after {max_consecutive_errors} "
                "consecutive failures, which means something is wrong with the "
                "video or the model rather than with individual frames.\n\n"
                f"Last error: {last_error}"
            )

        elapsed = time.monotonic() - started
        sequence.metadata["elapsed_seconds"] = f"{elapsed:.1f}"
        sequence.metadata["device"] = str(getattr(backend, "device", "cpu"))
        report(1.0, "Pose estimation complete")
        logger.info(
            "Pose estimation finished: %d/%d frames detected in %.1fs",
            sequence.detected_count,
            frame_count,
            elapsed,
        )
        return sequence
