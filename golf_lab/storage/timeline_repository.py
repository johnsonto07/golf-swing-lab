"""Per-swing storage for the measured source timeline.

Written to ``timeline.json`` beside the pose and analysis artifacts. Measuring
it means one ffprobe pass over every frame, which is cheap but not free on a
long clip, so it is measured once at import and read thereafter.

Staleness is decided by the fingerprints of the media it was measured from. If
either the original or the preview is replaced, the stored timing describes a
file that no longer exists and must not be used — the same rule the pose layer
applies, for the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from golf_lab.logging_config import get_logger
from golf_lab.storage.file_repository import swing_dir
from golf_lab.video.frame_cache import file_fingerprint
from golf_lab.video.timeline import (
    EXTRACTOR_VERSION,
    TIMELINE_SCHEMA_VERSION,
    SourceTimeline,
    TimelineError,
    build_timeline,
)

logger = get_logger(__name__)

TIMELINE_FILENAME = "timeline.json"


def timeline_path(swing_id: str, root: Optional[Path] = None) -> Path:
    return swing_dir(swing_id, root) / TIMELINE_FILENAME


def has_timeline(swing_id: str, root: Optional[Path] = None) -> bool:
    return timeline_path(swing_id, root).exists()


def save_timeline(
    swing_id: str, timeline: SourceTimeline, root: Optional[Path] = None
) -> Path:
    """Write the timeline atomically."""
    path = timeline_path(swing_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(timeline.to_dict(), indent=2), encoding="utf-8")
    temp_path.replace(path)
    logger.info(
        "Saved timeline for %s: %d frames, %s, %s",
        swing_id,
        timeline.frame_count,
        timeline.confidence.value,
        timeline.rate_classification.value,
    )
    return path


def load_timeline(
    swing_id: str, root: Optional[Path] = None
) -> Optional[SourceTimeline]:
    """Read the stored timeline, or None if absent or unreadable.

    Returning None rather than raising is deliberate: a missing timeline
    degrades timing to "unavailable", which the metric layer already refuses
    on. A corrupt one must not take the whole page down.
    """
    path = timeline_path(swing_id, root)
    if not path.exists():
        return None
    try:
        return SourceTimeline.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TimelineError, ValueError, KeyError) as exc:
        logger.warning("Unreadable timeline for %s: %s", swing_id, exc)
        return None


def delete_timeline(swing_id: str, root: Optional[Path] = None) -> None:
    try:
        timeline_path(swing_id, root).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete timeline for %s: %s", swing_id, exc)


def measure_and_save(
    swing_id: str,
    preview_path: Path,
    nominal_fps: float,
    container_frame_count: Optional[int] = None,
    original_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> Optional[SourceTimeline]:
    """Measure the preview's timing and persist it.

    The preview is measured rather than the original because it is what every
    interactive feature reads, and because ``-fps_mode passthrough`` carries
    the original timestamps into it unchanged — verified on a genuinely
    variable-frame-rate file at zero drift. Both fingerprints are recorded so
    replacing either invalidates the result.

    Returns None on failure rather than raising: a swing without measured
    timing is usable for everything except duration claims, and import should
    not fail because timestamps could not be read.
    """
    try:
        timeline = build_timeline(
            preview_path,
            nominal_fps=nominal_fps,
            container_frame_count=container_frame_count,
            source_fingerprint=(
                file_fingerprint(original_path) if original_path and original_path.exists() else ""
            ),
            preview_fingerprint=file_fingerprint(preview_path),
        )
    except (TimelineError, OSError) as exc:
        logger.warning("Could not measure timeline for %s: %s", swing_id, exc)
        return None

    save_timeline(swing_id, timeline, root)
    return timeline


def staleness_reasons(
    timeline: Optional[SourceTimeline],
    preview_path: Optional[Path] = None,
    original_path: Optional[Path] = None,
) -> List[str]:
    """Why a stored timeline should not be trusted, as user-facing sentences."""
    if timeline is None:
        return ["No measured timeline has been stored for this swing."]

    reasons: List[str] = []
    if timeline.schema_version != TIMELINE_SCHEMA_VERSION:
        reasons.append(
            f"It was stored with timeline schema v{timeline.schema_version}; "
            f"this build reads v{TIMELINE_SCHEMA_VERSION}."
        )
    if timeline.extractor_version != EXTRACTOR_VERSION:
        reasons.append(
            f"It was measured with extractor version "
            f"{timeline.extractor_version}; this build uses {EXTRACTOR_VERSION}."
        )

    for path, stored, label in (
        (preview_path, timeline.preview_fingerprint, "preview"),
        (original_path, timeline.source_fingerprint, "original video"),
    ):
        if path is None or not stored:
            continue
        try:
            if file_fingerprint(path) != stored:
                reasons.append(
                    f"The {label} has changed since this timing was measured."
                )
        except OSError:
            reasons.append(f"The {label} for this swing could not be read.")
    return reasons


def is_stale(
    timeline: Optional[SourceTimeline],
    preview_path: Optional[Path] = None,
    original_path: Optional[Path] = None,
) -> bool:
    return bool(staleness_reasons(timeline, preview_path, original_path))
