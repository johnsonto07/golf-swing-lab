"""Video metadata extraction with correct phone-orientation handling.

ffprobe is the primary source because it exposes the container rotation
metadata that phones write. OpenCV is used as a fallback so the app degrades
gracefully rather than refusing to open a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from golf_lab.logging_config import get_logger
from golf_lab.models.video import VideoMetadata
from golf_lab.video.ffmpeg import (
    FFmpegCommandError,
    FFmpegNotFoundError,
    probe_json,
)

logger = get_logger(__name__)


class VideoMetadataError(RuntimeError):
    """Raised when a file cannot be understood as a video at all."""


def parse_frame_rate(value: Optional[str]) -> float:
    """Parse an ffprobe rational frame rate such as '120000/1001'."""
    if not value:
        return 0.0
    value = value.strip()
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return 0.0
        if den == 0:
            return 0.0
        return num / den
    try:
        return float(value)
    except ValueError:
        return 0.0


def extract_rotation_degrees(video_stream: dict) -> int:
    """Return clockwise display rotation in degrees, normalized to 0/90/180/270.

    Phones express rotation in two different ways depending on the OS and
    ffmpeg version:
      * a ``rotate`` tag on the stream (older style, clockwise), and/or
      * a Display Matrix entry in ``side_data_list`` whose ``rotation`` is
        counter-clockwise (so -90 means "rotate 90 clockwise for display").
    """
    rotation: Optional[float] = None

    tags = video_stream.get("tags") or {}
    if "rotate" in tags:
        try:
            rotation = float(tags["rotate"])
        except (TypeError, ValueError):
            rotation = None

    if rotation is None:
        for side_data in video_stream.get("side_data_list") or []:
            if "rotation" in side_data:
                try:
                    # Display-matrix rotation is counter-clockwise; negate it
                    # to get the clockwise rotation needed for display.
                    rotation = -float(side_data["rotation"])
                except (TypeError, ValueError):
                    continue
                break

    if rotation is None:
        return 0

    normalized = int(round(rotation)) % 360
    # Snap to the four right angles; arbitrary angles are not supported and
    # are extremely rare in phone footage.
    snapped = min((0, 90, 180, 270), key=lambda angle: abs(angle - normalized))
    if normalized > 315:
        snapped = 0
    return snapped


def _probe_with_ffprobe(video_path: Path) -> VideoMetadata:
    data = probe_json(video_path)

    streams = data.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise VideoMetadataError(
            f"No video stream found in {video_path.name}. "
            "The file may be audio-only or corrupt."
        )

    vs = video_streams[0]
    fmt = data.get("format") or {}

    coded_width = int(vs.get("width") or 0)
    coded_height = int(vs.get("height") or 0)
    if coded_width <= 0 or coded_height <= 0:
        raise VideoMetadataError(
            f"Could not read frame dimensions from {video_path.name}."
        )

    rotation = extract_rotation_degrees(vs)
    if rotation in (90, 270):
        display_width, display_height = coded_height, coded_width
    else:
        display_width, display_height = coded_width, coded_height

    avg_frame_rate = parse_frame_rate(vs.get("avg_frame_rate"))
    r_frame_rate = parse_frame_rate(vs.get("r_frame_rate"))
    # avg wins because it reflects the frames actually present; r_frame_rate
    # is only the rate the container advertises. They disagree on VFR sources,
    # which is exactly what `is_variable_frame_rate` reports on.
    fps = avg_frame_rate or r_frame_rate

    duration = 0.0
    for candidate in (vs.get("duration"), fmt.get("duration")):
        if candidate:
            try:
                duration = float(candidate)
                break
            except (TypeError, ValueError):
                continue

    frame_count = 0
    estimated = False
    for key in ("nb_frames", "nb_read_frames"):
        raw = vs.get(key)
        if raw:
            try:
                frame_count = int(raw)
                break
            except (TypeError, ValueError):
                continue
    if frame_count <= 0:
        # Fall back to duration * fps. Flagged as estimated so the UI can say
        # so rather than implying a precise count.
        frame_count = int(round(duration * fps)) if duration and fps else 0
        estimated = True

    if duration <= 0 and fps > 0 and frame_count > 0:
        duration = frame_count / fps

    return VideoMetadata(
        path=str(video_path),
        container_format=(fmt.get("format_name") or None),
        codec_name=vs.get("codec_name"),
        audio_codec_name=(audio_streams[0].get("codec_name") if audio_streams else None),
        has_audio=bool(audio_streams),
        coded_width=coded_width,
        coded_height=coded_height,
        width=display_width,
        height=display_height,
        rotation_degrees=rotation,
        fps=float(fps),
        frame_count=frame_count,
        duration_seconds=float(duration),
        frame_count_is_estimated=estimated,
        avg_frame_rate=avg_frame_rate or None,
        r_frame_rate=r_frame_rate or None,
        probe_source="ffprobe",
    )


def _probe_with_opencv(video_path: Path) -> VideoMetadata:
    """Fallback probe. Cannot see container rotation metadata reliably."""
    import cv2  # imported lazily so metadata parsing helpers stay importable

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoMetadataError(
            f"Neither FFmpeg nor OpenCV could open {video_path.name}. "
            "The codec may be unsupported; try re-exporting the clip as H.264 MP4."
        )
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise VideoMetadataError(
            f"OpenCV opened {video_path.name} but reported invalid dimensions."
        )

    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    return VideoMetadata(
        path=str(video_path),
        coded_width=width,
        coded_height=height,
        width=width,
        height=height,
        rotation_degrees=0,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        frame_count_is_estimated=True,
        probe_source="opencv",
    )


def extract_metadata(video_path: Path) -> VideoMetadata:
    """Probe a video file, preferring ffprobe and falling back to OpenCV.

    The fallback is deliberately narrow. If ffprobe is *installed* and still
    rejects the file, that is a definitive answer — the file is not readable
    media — and we surface it immediately. Handing such a file to OpenCV
    instead makes it grind through every demuxer for tens of seconds before
    reaching the same conclusion. OpenCV is only consulted when ffprobe is
    genuinely unavailable.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoMetadataError(f"File does not exist: {video_path}")
    if video_path.stat().st_size == 0:
        raise VideoMetadataError(f"File is empty: {video_path.name}")

    try:
        return _probe_with_ffprobe(video_path)
    except FFmpegNotFoundError:
        logger.warning(
            "ffprobe is not installed; falling back to OpenCV for %s. "
            "Rotation metadata cannot be detected this way.",
            video_path.name,
        )
        return _probe_with_opencv(video_path)
    except FFmpegCommandError as exc:
        raise VideoMetadataError(
            f"FFmpeg could not read '{video_path.name}'. It may be corrupt, "
            "incompletely copied, or in a codec your FFmpeg build does not "
            f"support.\n\n{exc}"
        ) from exc
