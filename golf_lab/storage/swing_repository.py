"""Reading and writing swing records, and the video-ingestion pipeline.

This module owns Pipeline A (video ingestion) end to end:

    validate -> probe -> copy original (immutable) -> preview -> thumbnail
             -> persist metadata.json

The original file is copied once and then made read-only. Nothing in the
codebase ever writes to it again; exports always go to ``exports/``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Callable, List, Optional

from golf_lab.config import (
    ANALYSIS_VERSION,
    APP_VERSION,
    SUPPORTED_VIDEO_EXTENSIONS,
    SWINGS_DIR,
    ensure_data_dirs,
)
from golf_lab.logging_config import get_logger
from golf_lab.models.video import SwingContext, SwingRecord, SwingStatus
from golf_lab.storage.file_repository import (
    create_swing_dir,
    list_swing_ids,
    metadata_path,
    new_swing_id,
    sanitize_filename,
    swing_dir,
)
from golf_lab.video.metadata import VideoMetadataError, extract_metadata
from golf_lab.video.preview import PreviewGenerationError, generate_preview, generate_thumbnail

logger = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]


class SwingImportError(RuntimeError):
    """Raised when a swing could not be imported. Message is user-facing."""


def _noop_progress(fraction: float, message: str) -> None:
    logger.info("[import %3d%%] %s", int(fraction * 100), message)


def validate_upload(filename: str, size_bytes: int) -> None:
    """Reject obviously unusable uploads before any expensive work happens."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise SwingImportError(
            f"'{filename}' has an unsupported extension ({suffix or 'none'}). "
            f"Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}."
        )
    if size_bytes <= 0:
        raise SwingImportError(f"'{filename}' is empty (0 bytes).")


def _make_read_only(path: Path) -> None:
    """Best effort: clear the write bits on the imported original.

    A failure here is not fatal — it is a guard rail, not a security control —
    so it is logged rather than raised.
    """
    try:
        current = path.stat().st_mode
        path.chmod(current & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except OSError as exc:
        logger.warning("Could not mark original read-only (%s): %s", path.name, exc)


def import_swing(
    source_path: Path,
    original_filename: str,
    context: SwingContext,
    root: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
) -> SwingRecord:
    """Run Pipeline A for one uploaded video and return the saved record.

    ``source_path`` is the temporary file the UI wrote the upload to. It is
    copied, never moved, and never modified.
    """
    report = progress or _noop_progress
    ensure_data_dirs()
    root = root or SWINGS_DIR
    root.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_path)
    report(0.05, "Validating upload")
    validate_upload(original_filename, source_path.stat().st_size if source_path.exists() else 0)

    swing_id = new_swing_id()
    directory = create_swing_dir(swing_id, root)
    logger.info("Importing '%s' as swing %s", original_filename, swing_id)

    try:
        # --- copy the original, then freeze it -------------------------
        report(0.15, "Copying original video")
        safe_name = sanitize_filename(original_filename)
        extension = Path(safe_name).suffix.lower() or ".mp4"
        original_dest = directory / f"original{extension}"
        shutil.copyfile(source_path, original_dest)
        _make_read_only(original_dest)

        # --- probe -----------------------------------------------------
        report(0.35, "Reading video metadata")
        try:
            video_metadata = extract_metadata(original_dest)
        except VideoMetadataError as exc:
            raise SwingImportError(str(exc)) from exc

        if video_metadata.fps <= 0:
            raise SwingImportError(
                f"Could not determine a frame rate for '{original_filename}'. "
                "Frame-accurate stepping needs a valid fps. Try re-exporting the "
                "clip as constant-frame-rate H.264 MP4."
            )

        # --- preview + thumbnail ---------------------------------------
        report(0.5, "Generating browser preview (this is the slow step)")
        preview_path = directory / "preview.mp4"
        try:
            generate_preview(original_dest, preview_path, video_metadata)
        except PreviewGenerationError as exc:
            raise SwingImportError(str(exc)) from exc

        report(0.85, "Generating thumbnail")
        thumbnail_path = directory / "thumbnail.jpg"
        thumb_time = min(0.5, max(0.0, video_metadata.duration_seconds / 2))
        try:
            generate_thumbnail(preview_path, thumbnail_path, thumb_time)
        except PreviewGenerationError as exc:
            logger.warning("Thumbnail generation failed for %s: %s", swing_id, exc)
            thumbnail_path = None  # type: ignore[assignment]

        # --- verify the preview kept the frame count -------------------
        status = SwingStatus.READY
        status_detail = ""
        preview_meta = extract_metadata(preview_path)
        if (
            not video_metadata.frame_count_is_estimated
            and preview_meta.frame_count
            and abs(preview_meta.frame_count - video_metadata.frame_count) > 1
        ):
            status = SwingStatus.NEEDS_REVIEW
            status_detail = (
                f"Preview has {preview_meta.frame_count} frames but the original "
                f"reports {video_metadata.frame_count}. Frame numbers may be offset; "
                "this usually means the source is variable-frame-rate."
            )
            logger.warning("%s: %s", swing_id, status_detail)

        record = SwingRecord(
            swing_id=swing_id,
            original_filename=original_filename,
            original_relpath=original_dest.name,
            preview_relpath=preview_path.name,
            thumbnail_relpath=thumbnail_path.name if thumbnail_path else None,
            video=video_metadata,
            context=context,
            status=status,
            status_detail=status_detail,
            app_version=APP_VERSION,
            analysis_version=ANALYSIS_VERSION,
        )
        report(0.95, "Saving metadata")
        save_record(record, root)
        report(1.0, "Import complete")
        return record

    except Exception:
        # Leave no half-built swing directory behind.
        _cleanup_failed_import(directory)
        raise


def _cleanup_failed_import(directory: Path) -> None:
    def _force_writable(func, path, _exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass

    try:
        shutil.rmtree(directory, onerror=_force_writable)
        logger.info("Removed incomplete swing directory %s", directory.name)
    except OSError as exc:
        logger.warning("Could not clean up %s: %s", directory, exc)


def save_record(record: SwingRecord, root: Optional[Path] = None) -> Path:
    """Write metadata.json atomically so a crash can't truncate it."""
    path = metadata_path(record.swing_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json")
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def load_record(swing_id: str, root: Optional[Path] = None) -> SwingRecord:
    path = metadata_path(swing_id, root)
    if not path.exists():
        raise SwingImportError(f"No saved swing found with id '{swing_id}'.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return SwingRecord.model_validate(data)


def list_records(root: Optional[Path] = None) -> List[SwingRecord]:
    """Load every saved swing, skipping (and logging) unreadable ones."""
    records: List[SwingRecord] = []
    for swing_id in list_swing_ids(root):
        try:
            records.append(load_record(swing_id, root))
        except Exception as exc:  # noqa: BLE001 - one bad file must not hide the rest
            logger.warning("Skipping unreadable swing %s: %s", swing_id, exc)
    return records


def resolve_path(record: SwingRecord, relpath: Optional[str], root: Optional[Path] = None) -> Optional[Path]:
    """Turn a stored relative path into an absolute one, or None if missing."""
    if not relpath:
        return None
    path = swing_dir(record.swing_id, root) / relpath
    return path if path.exists() else None


def preview_or_original_path(record: SwingRecord, root: Optional[Path] = None) -> Path:
    """Path to use for interactive work: the preview if it exists."""
    preview = resolve_path(record, record.preview_relpath, root)
    if preview:
        return preview
    original = resolve_path(record, record.original_relpath, root)
    if original:
        return original
    raise SwingImportError(
        f"Swing {record.swing_id} has no readable video file on disk."
    )
