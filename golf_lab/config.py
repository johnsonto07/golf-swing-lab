"""Central configuration and filesystem paths for Golf Swing Lab.

Nothing in this module contacts the network or reads secrets into memory
beyond checking whether an environment variable is *set* (never its value).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Versioning -------------------------------------------------------
# Bump APP_VERSION for user-facing releases. Bump ANALYSIS_VERSION whenever
# a change could alter previously-computed measurements, so old results in
# swing history can be flagged as stale rather than silently trusted.
APP_VERSION = "0.1.0"
ANALYSIS_VERSION = "1"

# --- Filesystem layout --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SWINGS_DIR = DATA_DIR / "swings"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = REPO_ROOT / "models"


def ensure_data_dirs() -> None:
    """Create the local data directories if they don't exist yet.

    Safe to call repeatedly. Does not touch any user-provided video.
    """
    for path in (DATA_DIR, SWINGS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --- Video Lab settings --------------------------------------------------
# Preview videos are generated at a capped width to keep interactive
# scrubbing fast; the immutable original is always used for final export.
PREVIEW_MAX_WIDTH = 960
THUMBNAIL_JPEG_QUALITY = 3  # ffmpeg -q:v scale, 2 (best) - 31 (worst)

# Extensions accepted by the Video Lab uploader.
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v")


# --- Optional cloud coaching (not used before Milestone 7) --------------
def openai_integration_enabled() -> bool:
    """Whether an OPENAI_API_KEY is present in the environment.

    Only ever reports presence/absence as a boolean. Never logs, prints, or
    returns the key value itself.
    """
    return bool(os.environ.get("OPENAI_API_KEY"))
