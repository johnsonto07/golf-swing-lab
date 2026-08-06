"""Preview (proxy) and thumbnail generation.

Why a proxy exists at all: interactive frame scrubbing on a 4K 120fps phone
clip is slow, and some phone codecs (HEVC / MOV) will not play in a browser
<video> element at all. The preview is a browser-safe, downscaled, upright
H.264 MP4 used for *all* interactive work. The immutable original is used
only for final export.

Two invariants this module tries hard to preserve:
  1. Frame indices in the preview map 1:1 to frame indices in the original,
     so a frame number shown in the UI is meaningful for the original too.
  2. Rotation is baked into the pixels, so anything reading the preview
     (OpenCV, the browser) sees an upright image without needing to
     re-interpret container metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from golf_lab.config import PREVIEW_MAX_WIDTH, THUMBNAIL_JPEG_QUALITY
from golf_lab.logging_config import get_logger
from golf_lab.models.video import VideoMetadata
from golf_lab.video.ffmpeg import FFmpegCommandError, find_ffmpeg, run_command

logger = get_logger(__name__)


class PreviewGenerationError(RuntimeError):
    """Raised when a browser-compatible preview could not be produced."""


def _scale_filter(metadata: VideoMetadata, max_width: int) -> Optional[str]:
    """Downscale only if the display width exceeds the cap.

    ``-2`` keeps the aspect ratio while forcing an even height, which H.264
    with yuv420p chroma subsampling requires.
    """
    if metadata.width <= max_width:
        return None
    return f"scale={max_width}:-2:flags=bicubic"


def build_preview_command(
    ffmpeg_binary: str,
    source: Path,
    destination: Path,
    metadata: VideoMetadata,
    max_width: int = PREVIEW_MAX_WIDTH,
    frame_sync_flag: str = "-fps_mode",
) -> List[str]:
    """Assemble the ffmpeg command for preview generation.

    Exposed separately from ``generate_preview`` so it can be unit tested
    without invoking FFmpeg.
    """
    filters: List[str] = []
    scale = _scale_filter(metadata, max_width)
    if scale:
        filters.append(scale)
    # Guarantee even dimensions even when no scaling was applied (odd-sized
    # source frames otherwise fail to encode with yuv420p).
    filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")

    sync_value = "passthrough" if frame_sync_flag == "-fps_mode" else "0"

    command = [
        ffmpeg_binary,
        "-y",
        "-nostdin",  # never block waiting on stdin when run from a UI process
        "-hide_banner",
        "-loglevel", "error",
        # ffmpeg auto-applies the container display matrix while decoding, so
        # the rotation is baked into the output pixels.
        "-i", str(source),
        "-map", "0:v:0",
        "-vf", ",".join(filters),
        frame_sync_flag, sync_value,  # never drop/duplicate frames
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",        # required for browser playback
        "-profile:v", "high",
        "-movflags", "+faststart",
        "-an",                        # preview is silent; audio comes from the original at export
        "-metadata:s:v:0", "rotate=0",
        str(destination),
    ]
    return command


def generate_preview(
    source: Path,
    destination: Path,
    metadata: VideoMetadata,
    max_width: int = PREVIEW_MAX_WIDTH,
) -> Path:
    """Create an upright, browser-playable H.264 preview.

    The source file is only ever read, never modified.
    """
    tools = find_ffmpeg(required=True)
    assert tools is not None

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # `-fps_mode` replaced `-vsync` in FFmpeg 5.x. Try the modern flag first
    # and fall back so this works on both old and new installs.
    attempts = ("-fps_mode", "-vsync")
    last_error: Optional[Exception] = None
    for flag in attempts:
        command = build_preview_command(
            tools.ffmpeg, source, destination, metadata, max_width, flag
        )
        try:
            run_command(command)
            logger.info("Generated preview: %s", destination.name)
            return destination
        except FFmpegCommandError as exc:
            last_error = exc
            logger.debug("Preview attempt with %s failed; trying fallback.", flag)
            continue

    raise PreviewGenerationError(
        "Could not generate a browser-compatible preview. This usually means "
        "the source codec is unsupported by your FFmpeg build.\n\n"
        f"{last_error}"
    ) from last_error


def generate_thumbnail(
    source: Path,
    destination: Path,
    timestamp_seconds: float = 0.0,
) -> Path:
    """Save a single JPEG still from ``source`` at the given timestamp.

    ``source`` should normally be the already-upright preview so the thumbnail
    inherits correct orientation.
    """
    tools = find_ffmpeg(required=True)
    assert tools is not None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        tools.ffmpeg,
        "-y",
        "-nostdin",  # never block waiting on stdin when run from a UI process
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{max(0.0, timestamp_seconds):.3f}",
        "-i", str(source),
        "-frames:v", "1",
        "-q:v", str(THUMBNAIL_JPEG_QUALITY),
        str(destination),
    ]
    try:
        run_command(command)
    except FFmpegCommandError as exc:
        if timestamp_seconds > 0:
            logger.debug("Thumbnail seek failed; retrying at t=0.")
            return generate_thumbnail(source, destination, 0.0)
        raise PreviewGenerationError(f"Could not generate thumbnail.\n\n{exc}") from exc

    # Seeking past the end of a clip makes FFmpeg exit 0 while writing nothing,
    # so success has to be judged by the file, not by the exit code.
    if not destination.exists() or destination.stat().st_size == 0:
        if timestamp_seconds > 0:
            logger.debug(
                "Thumbnail at t=%.3fs produced no output; retrying at t=0.",
                timestamp_seconds,
            )
            return generate_thumbnail(source, destination, 0.0)
        raise PreviewGenerationError(
            f"FFmpeg reported success but wrote no thumbnail for {source.name}."
        )

    logger.info("Generated thumbnail: %s", destination.name)
    return destination
