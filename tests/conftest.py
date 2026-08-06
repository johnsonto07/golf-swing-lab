"""Shared pytest fixtures.

The test fixture video is *generated* with FFmpeg rather than committed. That
keeps private golf footage out of the repository entirely and makes the
fixture reproducible on any machine.
"""

from __future__ import annotations

import json
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

    # `-display_rotation` is an *input* option and writes a real Display Matrix.
    # The legacy `-metadata:s:v:0 rotate=` tag is the fallback for FFmpeg < 6;
    # from FFmpeg 7 on it is silently ignored, which produced an unrotated
    # fixture that made the rotation tests pass vacuously.
    #
    # The sign is deliberate: FFmpeg takes counter-clockwise degrees here, and
    # a phone held in portrait writes rotation=-90, meaning "turn 90 clockwise
    # to display". Passing -rotate_metadata reproduces that exactly.
    stamping_attempts = (
        [
            ffmpeg_binary, "-y", "-hide_banner", "-loglevel", "error",
            "-display_rotation", str(-rotate_metadata),
            "-i", str(intermediate),
            "-c", "copy",
            str(destination),
        ],
        [
            ffmpeg_binary, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(intermediate),
            "-c", "copy",
            "-metadata:s:v:0", f"rotate={rotate_metadata}",
            str(destination),
        ],
    )

    for attempt in stamping_attempts:
        result = subprocess.run(attempt, capture_output=True)
        if result.returncode == 0 and _probed_rotation(ffmpeg_binary, destination):
            intermediate.unlink(missing_ok=True)
            return destination

    intermediate.unlink(missing_ok=True)
    raise RuntimeError(
        "Could not build a rotated fixture video: no rotation metadata survived. "
        "Rotation tests would otherwise pass vacuously against an upright clip. "
        f"FFmpeg binary: {ffmpeg_binary}"
    )


def _probed_rotation(ffmpeg_binary: str, video: Path) -> bool:
    """Return True if ``video`` actually carries non-zero rotation metadata.

    Verifying rather than trusting the stamping command is the whole point: a
    fixture that quietly lost its rotation turns every rotation assertion into
    a no-op.
    """
    ffprobe_binary = str(Path(ffmpeg_binary).with_name("ffprobe" + Path(ffmpeg_binary).suffix))
    result = subprocess.run(
        [
            ffprobe_binary, "-v", "error",
            "-select_streams", "v:0",
            "-show_streams",
            "-of", "json",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        streams = json.loads(result.stdout).get("streams") or []
    except json.JSONDecodeError:
        return False
    if not streams:
        return False

    stream = streams[0]
    if (stream.get("tags") or {}).get("rotate"):
        return True
    return any(
        side_data.get("rotation")
        for side_data in (stream.get("side_data_list") or [])
    )


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


# --- pose fixtures ------------------------------------------------------
class FakePoseBackend:
    """A deterministic stand-in for a real pose backend.

    Lets the inference loop, storage, staleness, and overlay be tested without
    MediaPipe installed, without a downloaded model, and without the
    frame-to-frame nondeterminism of a real detector. ``fail_frames`` drives
    the undetected-frame paths, which are otherwise awkward to provoke on a
    synthetic clip where a detector would either always or never find a pose.
    """

    def __init__(self, fail_frames=(), error_frames=(), jitter=0.0, seed=0):
        self.name = "fake/test-backend"
        self.device = "cpu"
        self.fail_frames = set(fail_frames)
        self.error_frames = set(error_frames)
        self.jitter = jitter
        self.calls = []
        self.timestamps = []
        self.closed = False
        self._rng = __import__("numpy").random.default_rng(seed)

    def detect(self, frame_rgb, timestamp_ms):
        import numpy as np

        from golf_lab.pose.backend import PoseBackendError, PoseFrameResult
        from golf_lab.pose.landmarks import NUM_LANDMARKS

        index = len(self.calls)
        self.calls.append(index)
        self.timestamps.append(timestamp_ms)

        if index in self.error_frames:
            raise PoseBackendError(f"synthetic backend failure on frame {index}")
        if index in self.fail_frames:
            return None

        # A body that drifts steadily across the frame, so smoothing has a
        # real trend to preserve and jitter to remove.
        base = 0.25 + 0.5 * (index / 100.0)
        landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        landmarks[:, 0] = base
        landmarks[:, 1] = np.linspace(0.1, 0.9, NUM_LANDMARKS)
        landmarks[:, 2] = 0.0
        if self.jitter:
            landmarks[:, :2] += self._rng.normal(0, self.jitter, (NUM_LANDMARKS, 2))

        return PoseFrameResult(
            landmarks=landmarks,
            world_landmarks=landmarks.copy(),
            visibility=np.full(NUM_LANDMARKS, 0.9, dtype=np.float32),
            presence=np.full(NUM_LANDMARKS, 0.95, dtype=np.float32),
        )

    def close(self):
        self.closed = True


@pytest.fixture()
def fake_backend():
    """Factory so each test can choose its own failure pattern."""
    return FakePoseBackend


@pytest.fixture()
def swing_pose_factory():
    """Build a deterministic synthetic *swing* as a PoseSequence.

    Milestone 3 detectors are pure functions of landmarks, so they are tested
    against generated arrays rather than video: no MediaPipe, no model, no
    decoding, and — more importantly — a swing whose address and top frames are
    known exactly, so a test can assert the detector found the right frame
    rather than merely a plausible one.

    The generated swing has the rhythm that matters: a still address, a slow
    backswing, a fast downswing, and a finish with the hands high again (which
    is what makes "highest hands" alone an insufficient rule for the top).
    """

    def _build(
        address_frames: int = 12,
        backswing_frames: int = 30,
        downswing_frames: int = 8,
        follow_frames: int = 25,
        hand_visibility: float = 0.95,
        body_visibility: float = 0.95,
        failed_frames: tuple = (),
        jitter: float = 0.0,
        seed: int = 7,
        fps: float = 30.0,
        width: int = 720,
        height: int = 1280,
    ):
        import numpy as np

        from golf_lab.pose import landmarks as lmk
        from golf_lab.pose.sequence import PoseSequence

        rng = np.random.default_rng(seed)
        total = address_frames + backswing_frames + downswing_frames + follow_frames
        sequence = PoseSequence.empty(total, fps=fps, frame_width=width, frame_height=height)

        # Swing angle: 0 at address, -1 at top, +1 at finish.
        angle = np.concatenate(
            [
                np.zeros(address_frames),
                -np.sin(np.linspace(0, np.pi / 2, backswing_frames)),
                -np.cos(np.linspace(0, np.pi / 2, downswing_frames)),
                np.sin(np.linspace(0, np.pi / 2, follow_frames)),
            ]
        )

        shoulder_y, hip_y = 0.36, 0.55
        half_shoulder = 0.09
        hand_radius = 0.20

        for index in range(total):
            if index in set(failed_frames):
                sequence.mark_failed(index)
                continue

            a = float(angle[index])
            points = np.zeros((lmk.NUM_LANDMARKS, 3), dtype=np.float32)

            # Torso is essentially still; the hands do the moving.
            points[lmk.LEFT_SHOULDER] = (0.5 - half_shoulder, shoulder_y, 0.0)
            points[lmk.RIGHT_SHOULDER] = (0.5 + half_shoulder, shoulder_y, 0.0)
            points[lmk.LEFT_HIP] = (0.5 - 0.06, hip_y, 0.0)
            points[lmk.RIGHT_HIP] = (0.5 + 0.06, hip_y, 0.0)
            points[lmk.NOSE] = (0.5 + 0.02 * a, shoulder_y - 0.10, 0.0)
            points[lmk.LEFT_KNEE] = (0.5 - 0.06, 0.72, 0.0)
            points[lmk.RIGHT_KNEE] = (0.5 + 0.06, 0.72, 0.0)
            points[lmk.LEFT_ANKLE] = (0.5 - 0.07, 0.88, 0.0)
            points[lmk.RIGHT_ANKLE] = (0.5 + 0.07, 0.88, 0.0)

            # Hands swing on an arc: down at address, high at top and finish.
            hand_x = 0.5 + hand_radius * np.sin(a * np.pi * 0.55)
            hand_y = hip_y + hand_radius * np.cos(a * np.pi * 0.55)
            points[lmk.LEFT_WRIST] = (hand_x - 0.01, hand_y, 0.0)
            points[lmk.RIGHT_WRIST] = (hand_x + 0.01, hand_y, 0.0)
            points[lmk.LEFT_ELBOW] = (
                (points[lmk.LEFT_SHOULDER][0] + hand_x) / 2,
                (shoulder_y + hand_y) / 2,
                0.0,
            )
            points[lmk.RIGHT_ELBOW] = (
                (points[lmk.RIGHT_SHOULDER][0] + hand_x) / 2,
                (shoulder_y + hand_y) / 2,
                0.0,
            )

            if jitter:
                points[:, :2] += rng.normal(0, jitter, (lmk.NUM_LANDMARKS, 2))

            visibility = np.full(lmk.NUM_LANDMARKS, body_visibility, dtype=np.float32)
            visibility[[lmk.LEFT_WRIST, lmk.RIGHT_WRIST]] = hand_visibility

            sequence.set_frame(
                index,
                landmarks=points,
                world_landmarks=points.copy(),
                visibility=visibility,
                presence=np.full(lmk.NUM_LANDMARKS, 0.95, dtype=np.float32),
            )

        # Expected phases, so tests assert against truth rather than output.
        sequence.metadata["expected_address"] = str(address_frames - 1)
        sequence.metadata["expected_top"] = str(address_frames + backswing_frames - 1)
        return sequence

    return _build


@pytest.fixture()
def pose_sequence_factory():
    """Build a PoseSequence with a chosen set of undetected frames."""

    def _build(frame_count=30, failed=(), fps=30.0, width=320, height=240, jitter=0.0, seed=1):
        import numpy as np

        from golf_lab.pose.landmarks import NUM_LANDMARKS
        from golf_lab.pose.sequence import PoseSequence

        rng = np.random.default_rng(seed)
        sequence = PoseSequence.empty(frame_count, fps=fps, frame_width=width, frame_height=height)
        for index in range(frame_count):
            if index in set(failed):
                sequence.mark_failed(index)
                continue
            landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            landmarks[:, 0] = 0.2 + 0.6 * (index / max(frame_count - 1, 1))
            landmarks[:, 1] = np.linspace(0.1, 0.9, NUM_LANDMARKS)
            if jitter:
                landmarks[:, :2] += rng.normal(0, jitter, (NUM_LANDMARKS, 2))
            sequence.set_frame(
                index,
                landmarks=landmarks,
                world_landmarks=landmarks.copy(),
                visibility=np.full(NUM_LANDMARKS, 0.8, dtype=np.float32),
                presence=np.full(NUM_LANDMARKS, 0.9, dtype=np.float32),
            )
        return sequence

    return _build
