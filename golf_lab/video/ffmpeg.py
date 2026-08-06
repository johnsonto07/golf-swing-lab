"""Thin, typed wrapper around the ffmpeg/ffprobe command-line tools.

Everything that shells out to FFmpeg goes through here so that binary
discovery, error reporting, and Windows path quoting are handled in one place.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from golf_lab.logging_config import get_logger

logger = get_logger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg/ffprobe binaries are not on PATH."""


class FFmpegCommandError(RuntimeError):
    """Raised when an ffmpeg/ffprobe invocation exits non-zero.

    Carries the tail of stderr so the UI can show a real diagnosis instead of
    a generic failure.
    """

    def __init__(self, command: Sequence[str], returncode: int, stderr: str):
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        super().__init__(
            f"FFmpeg command failed (exit {returncode}).\n"
            f"Command: {' '.join(command)}\n"
            f"stderr tail:\n{tail}"
        )


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: str
    ffprobe: str
    version: str


@lru_cache(maxsize=1)
def _locate_ffmpeg() -> Optional[FFmpegTools]:
    """Discover the FFmpeg binaries once per process.

    Cached because every probe, preview, and export otherwise re-runs two
    PATH lookups plus ``ffmpeg -version``, which is measurable when a page
    reads many frames. Call ``refresh_ffmpeg()`` after installing FFmpeg
    without restarting the app.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        return None

    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        version_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
    except (OSError, subprocess.SubprocessError):
        version_line = "unknown"

    return FFmpegTools(ffmpeg=ffmpeg_path, ffprobe=ffprobe_path, version=version_line)


def find_ffmpeg(required: bool = True) -> Optional[FFmpegTools]:
    """Locate ffmpeg and ffprobe on PATH.

    Returns None instead of raising when ``required`` is False, so the
    diagnostics page can report a missing install rather than crash.
    """
    tools = _locate_ffmpeg()
    if tools is None and required:
        raise FFmpegNotFoundError(
            "ffmpeg and ffprobe were not found on PATH. Install FFmpeg and "
            "make sure the 'bin' folder is on your PATH, then restart the app. "
            "On Windows: winget install Gyan.FFmpeg"
        )
    return tools


def refresh_ffmpeg() -> Optional[FFmpegTools]:
    """Re-run discovery, e.g. after the user installs FFmpeg mid-session."""
    _locate_ffmpeg.cache_clear()
    return _locate_ffmpeg()


def run_command(command: Sequence[str], timeout: Optional[float] = None) -> str:
    """Run a command, raising FFmpegCommandError with stderr on failure."""
    logger.debug("Running command: %s", " ".join(command))
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(str(exc)) from exc

    if result.returncode != 0:
        raise FFmpegCommandError(command, result.returncode, result.stderr or "")
    return result.stdout


def probe_json(video_path: Path, timeout: float = 60.0) -> dict:
    """Return the full ffprobe JSON description of a media file."""
    tools = find_ffmpeg(required=True)
    assert tools is not None
    stdout = run_command(
        [
            tools.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ],
        timeout=timeout,
    )
    return json.loads(stdout)
