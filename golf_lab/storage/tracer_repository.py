"""Per-swing storage for the ball tracer.

Written to ``tracer.json`` beside the pose, timeline, and analysis artifacts,
and deliberately separate from all of them. A tracer is the only artifact in
this project that is *authored* rather than computed: re-running pose
inference or phase detection must never touch it, because nothing can
regenerate a drawing the user made by hand.

That asymmetry drives the staleness rule too. Derived artifacts are discarded
when they go stale. A stale tracer is reported and kept, because the
coordinates still represent real work even when the frame they were placed on
has been replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from golf_lab.logging_config import get_logger
from golf_lab.storage.file_repository import swing_dir
from golf_lab.tracer.model import TRACER_SCHEMA_VERSION, TracerSpec

logger = get_logger(__name__)

TRACER_FILENAME = "tracer.json"


def tracer_path(swing_id: str, root: Optional[Path] = None) -> Path:
    return swing_dir(swing_id, root) / TRACER_FILENAME


def has_tracer(swing_id: str, root: Optional[Path] = None) -> bool:
    return tracer_path(swing_id, root).exists()


def save_tracer(spec: TracerSpec, root: Optional[Path] = None) -> Path:
    """Write the tracer atomically.

    The temp-file-then-replace dance matters more here than elsewhere: this is
    hand-authored data, so a partial write during a crash would destroy
    something the user cannot get back by re-running anything.
    """
    path = tracer_path(spec.swing_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    temp_path.replace(path)
    logger.info(
        "Saved tracer for %s: impact frame %s, %d points (%d confirmed)",
        spec.swing_id,
        spec.impact_frame,
        len(spec.points),
        len(spec.confirmed_points),
    )
    return path


def load_tracer(swing_id: str, root: Optional[Path] = None) -> Optional[TracerSpec]:
    """Read the stored tracer, or None if there is none or it is unreadable."""
    path = tracer_path(swing_id, root)
    if not path.exists():
        return None
    try:
        return TracerSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Unreadable tracer for %s: %s", swing_id, exc)
        return None


def delete_tracer(swing_id: str, root: Optional[Path] = None) -> None:
    """Remove the tracer. Only ever called on explicit user action."""
    try:
        tracer_path(swing_id, root).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete tracer for %s: %s", swing_id, exc)


def staleness_reasons(
    spec: Optional[TracerSpec], preview_fingerprint: str = ""
) -> List[str]:
    """Why a stored tracer may no longer line up with the video.

    Returns user-facing sentences, empty when the tracer is trustworthy. The
    caller decides what to do about them — unlike derived analysis, the answer
    here is never "silently recompute".
    """
    if spec is None:
        return []

    reasons: List[str] = []

    if spec.schema_version != TRACER_SCHEMA_VERSION:
        reasons.append(
            f"It was stored with tracer schema v{spec.schema_version}; this "
            f"build uses v{TRACER_SCHEMA_VERSION}."
        )
    if (
        preview_fingerprint
        and spec.preview_fingerprint
        and spec.preview_fingerprint != preview_fingerprint
    ):
        reasons.append(
            "The preview video it was drawn on has been regenerated, so its "
            "ball positions may no longer line up with what you see."
        )
    return reasons


def is_stale(spec: Optional[TracerSpec], preview_fingerprint: str = "") -> bool:
    return bool(staleness_reasons(spec, preview_fingerprint))
