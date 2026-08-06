"""Shared pytest fixtures.

The test fixture video is *generated* with FFmpeg rather than committed. That
keeps private golf footage out of the repository entirely and makes the
fixture reproducible on any machine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from golf_lab.video.ffmpeg import find_ffmpeg  # noqa: E402

FIXTURE_FPS = 30
FIXTURE_FRAMES = 60
FIXTURE_WIDTH = 320
FIXTURE_HEIGHT = 240


def _ffmpeg_or_skip():
    tools = find_ffmpeg(required=False)
    if tools is None:
        pytest.skip("FFmpeg is not installed; skipping tests that require it.")
    return tools


@pytest.fixture(scope="session")
def ffmpeg_tools():
    return _ffmpeg_or_skip()


def _build_fixture_video(
    ffmpeg_binary: str,
    destination: Path,
    rotate_metadata: Optional[int] = None,
) -> Path:
    """Render a tiny synthetic clip with a burned-in frame counter.

    The visible counter lets tests assert that the frame the reader returns is
    the frame that was requested, rather than a neighbouring one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = FIXTURE_FRAMES / FIXTURE_FPS

    command = [
        ffmpeg_binary, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi",
        "-i",
        f"testsrc=size={FIXTURE_WIDTH}x{FIXTURE_HEIGHT}:rate={FIXTURE_FPS}:duration={duration}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
    ]

    if rotate_metadata is None:
        command.append(str(destination))
        subprocess.run(command, check=True, capture_output=True)
        return destination

    # Rotation metadata does not survive being set during an encode from a
    # lavfi source, so write the clip first and stamp the rotation on with a
    # stream copy. This is also closer to how phones actually store it: the
    # pixels are landscape and a display matrix tells players to turn them.
    intermediate = destination.with_name(f"_pre_{destination.name}")
    command.append(str(intermediate))
    subprocess.run(command, check=True, capture_output=True)

    subprocess.run(
        [
            ffmpeg_binary, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(intermediate),
            "-c", "copy",
            "-metadata:s:v:0", f"rotate={rotate_metadata}",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    intermediate.unlink(missing_ok=True)
    return destination


@pytest.fixture(scope="session")
def fixture_video(ffmpeg_tools, tmp_path_factory) -> Path:
    """A small 320x240, 30fps, 60-frame H.264 MP4 with an audio track."""
    directory = tmp_path_factory.mktemp("fixtures")
    return _build_fixture_video(ffmpeg_tools.ffmpeg, directory / "synthetic_swing.mp4")


@pytest.fixture(scope="session")
def fixture_video_rotated(ffmpeg_tools, tmp_path_factory) -> Path:
    """Same clip, tagged as needing 90 degrees of display rotation."""
    directory = tmp_path_factory.mktemp("fixtures_rotated")
    return _build_fixture_video(
        ffmpeg_tools.ffmpeg, directory / "synthetic_portrait.mp4", rotate_metadata=90
    )


@pytest.fixture(scope="session")
def fixture_video_awkward_name(ffmpeg_tools, tmp_path_factory) -> Path:
    """Filename with spaces and parentheses, per the acceptance criteria."""
    directory = tmp_path_factory.mktemp("fixtures_awkward")
    return _build_fixture_video(
        ffmpeg_tools.ffmpeg, directory / "My Swing (driver) 2.mp4"
    )


@pytest.fixture()
def swing_root(tmp_path) -> Path:
    """An isolated data/swings root so tests never touch real user data."""
    root = tmp_path / "swings"
    root.mkdir(parents=True, exist_ok=True)
    return root
