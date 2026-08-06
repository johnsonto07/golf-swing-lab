"""Filesystem layout helpers for per-swing directories.

All path construction goes through here so the on-disk layout is defined in
exactly one place:

    data/swings/<swing_id>/
        original.<ext>
        preview.mp4
        thumbnail.jpg
        metadata.json
        exports/
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from golf_lab.config import SWINGS_DIR

# Characters Windows forbids in filenames, plus control characters.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")

# Names Windows reserves regardless of extension.
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename(name: str, fallback: str = "video") -> str:
    """Make an arbitrary uploaded filename safe to write on Windows.

    Handles the cases called out in the acceptance criteria — spaces and
    parentheses — plus reserved device names and non-ASCII characters. The
    *original* filename is still preserved verbatim in metadata.json; this
    only affects the name used on disk.
    """
    name = unicodedata.normalize("NFKC", name).strip()

    # Split before substituting so that trailing spaces in the stem
    # ("clip   .mp4") disappear instead of becoming underscores.
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""

    def _clean(part: str) -> str:
        part = part.strip()
        part = part.replace("(", "_").replace(")", "_")
        part = _UNSAFE_CHARS.sub("_", part)
        part = _WHITESPACE.sub("_", part)
        return part.strip("._ ")

    stem = _clean(stem)
    suffix = _clean(suffix)

    if stem.upper() in _RESERVED_NAMES:
        stem = f"{stem}_file"
    if not stem:
        stem = fallback

    stem = stem[:80]  # keep total path length well under Windows' MAX_PATH
    return f"{stem}.{suffix}" if suffix else stem


def new_swing_id() -> str:
    """Sortable, collision-resistant local identifier."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def swing_dir(swing_id: str, root: Path | None = None) -> Path:
    return (root or SWINGS_DIR) / swing_id


def metadata_path(swing_id: str, root: Path | None = None) -> Path:
    return swing_dir(swing_id, root) / "metadata.json"


def exports_dir(swing_id: str, root: Path | None = None) -> Path:
    return swing_dir(swing_id, root) / "exports"


def create_swing_dir(swing_id: str, root: Path | None = None) -> Path:
    directory = swing_dir(swing_id, root)
    directory.mkdir(parents=True, exist_ok=False)
    exports_dir(swing_id, root).mkdir(parents=True, exist_ok=True)
    return directory


def list_swing_ids(root: Path | None = None) -> List[str]:
    """Swing ids that have a metadata.json, newest first."""
    base = root or SWINGS_DIR
    if not base.exists():
        return []
    ids = [
        entry.name
        for entry in base.iterdir()
        if entry.is_dir() and (entry / "metadata.json").exists()
    ]
    return sorted(ids, reverse=True)
