"""The per-swing pose result: landmarks over time, plus what failed.

Design commitment, restated here because it is easy to erode later: **a frame
where pose estimation failed is recorded as failed.** Its landmark values are
NaN and its ``detected`` flag is False. Nothing in this module fills those
gaps in. Interpolating a missing wrist would make a swing look smooth and
measurable when the truth is that the wrist was never seen, and every
downstream number would inherit that fiction silently.

Smoothing produces a *separate* sequence (see ``pose.smoothing``); the raw
result is always kept alongside it so any measurement can be traced back to
what was actually observed.

Storage format is a compressed ``.npz`` with a ``format_version`` so a future
change can be detected rather than misread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from golf_lab.pose.landmarks import KEY_SWING_LANDMARKS, NUM_LANDMARKS

# Bump when the array layout changes in a way older code would misread.
POSE_FORMAT_VERSION = 1


class PoseSequenceError(RuntimeError):
    """Raised when a pose sequence cannot be built, read, or written."""


@dataclass
class PoseSequence:
    """Landmarks for every frame of one video.

    Arrays are frame-major so ``sequence.landmarks[i]`` is the whole skeleton
    at frame ``i``:

    ==================  ======================  ===========================
    Attribute           Shape                   Meaning
    ==================  ======================  ===========================
    ``landmarks``       (frames, 33, 3)         normalized image coords;
                                                x/y in 0..1 of the *frame*,
                                                z is relative depth
    ``world_landmarks`` (frames, 33, 3)         roughly metric, hip-centred
    ``visibility``      (frames, 33)            landmark is not occluded
    ``presence``        (frames, 33)            landmark is in frame at all
    ``detected``        (frames,)               bool: pose found on this frame
    ==================  ======================  ===========================

    Undetected frames hold NaN, never zeros — zero is a valid coordinate and
    would quietly place a joint in the top-left corner of the image.
    """

    landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    detected: np.ndarray

    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    smoothing: str = "none"
    metadata: Dict[str, str] = field(default_factory=dict)

    # -- construction ---------------------------------------------------
    @classmethod
    def empty(
        cls,
        frame_count: int,
        fps: float = 0.0,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> "PoseSequence":
        """An all-undetected sequence, ready to be filled in frame by frame.

        Starting from "nothing was detected" rather than "everything is zero"
        means an aborted or partial run degrades into honestly-missing data.
        """
        if frame_count < 0:
            raise PoseSequenceError(f"frame_count must be >= 0, got {frame_count}")
        return cls(
            landmarks=np.full((frame_count, NUM_LANDMARKS, 3), np.nan, dtype=np.float32),
            world_landmarks=np.full((frame_count, NUM_LANDMARKS, 3), np.nan, dtype=np.float32),
            visibility=np.zeros((frame_count, NUM_LANDMARKS), dtype=np.float32),
            presence=np.zeros((frame_count, NUM_LANDMARKS), dtype=np.float32),
            detected=np.zeros(frame_count, dtype=bool),
            fps=float(fps),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        n = len(self.detected)
        expected = {
            "landmarks": (n, NUM_LANDMARKS, 3),
            "world_landmarks": (n, NUM_LANDMARKS, 3),
            "visibility": (n, NUM_LANDMARKS),
            "presence": (n, NUM_LANDMARKS),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise PoseSequenceError(
                    f"{name} has shape {actual}, expected {shape}. "
                    "The arrays in a PoseSequence must agree on frame count."
                )

    # -- basic properties -----------------------------------------------
    @property
    def frame_count(self) -> int:
        return int(len(self.detected))

    @property
    def detected_count(self) -> int:
        return int(np.count_nonzero(self.detected))

    @property
    def failed_frames(self) -> np.ndarray:
        """Indices of frames where no pose was found, ascending."""
        return np.flatnonzero(~self.detected)

    @property
    def detection_rate(self) -> float:
        """Fraction of frames with a pose, 0.0-1.0."""
        if self.frame_count == 0:
            return 0.0
        return self.detected_count / self.frame_count

    def __len__(self) -> int:
        return self.frame_count

    # -- writing ---------------------------------------------------------
    def set_frame(
        self,
        index: int,
        landmarks: np.ndarray,
        world_landmarks: Optional[np.ndarray] = None,
        visibility: Optional[np.ndarray] = None,
        presence: Optional[np.ndarray] = None,
    ) -> None:
        """Record a successful detection for one frame."""
        if not 0 <= index < self.frame_count:
            raise PoseSequenceError(
                f"Frame {index} is outside 0..{self.frame_count - 1}"
            )
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (NUM_LANDMARKS, 3):
            raise PoseSequenceError(
                f"landmarks for frame {index} have shape {landmarks.shape}, "
                f"expected ({NUM_LANDMARKS}, 3)"
            )

        self.landmarks[index] = landmarks
        if world_landmarks is not None:
            self.world_landmarks[index] = np.asarray(world_landmarks, dtype=np.float32)
        if visibility is not None:
            self.visibility[index] = np.asarray(visibility, dtype=np.float32)
        if presence is not None:
            self.presence[index] = np.asarray(presence, dtype=np.float32)
        self.detected[index] = True

    def mark_failed(self, index: int) -> None:
        """Record that no pose was found on this frame.

        Explicit rather than implicit: calling this is how a backend says "I
        looked and found nothing", which is different from "not processed".
        """
        if not 0 <= index < self.frame_count:
            raise PoseSequenceError(
                f"Frame {index} is outside 0..{self.frame_count - 1}"
            )
        self.landmarks[index] = np.nan
        self.world_landmarks[index] = np.nan
        self.visibility[index] = 0.0
        self.presence[index] = 0.0
        self.detected[index] = False

    # -- reading ---------------------------------------------------------
    def pixel_coordinates(
        self,
        index: int,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Landmarks for one frame as (33, 2) pixel coordinates.

        Returns ``None`` for an undetected frame rather than an array of NaNs,
        so callers must handle the "no pose here" case explicitly instead of
        accidentally drawing garbage.
        """
        if not 0 <= index < self.frame_count:
            raise PoseSequenceError(
                f"Frame {index} is outside 0..{self.frame_count - 1}"
            )
        if not self.detected[index]:
            return None

        width = int(width if width is not None else self.frame_width)
        height = int(height if height is not None else self.frame_height)
        if width <= 0 or height <= 0:
            raise PoseSequenceError(
                "Frame dimensions are unknown, so normalized landmarks cannot "
                "be converted to pixels. Pass width and height explicitly."
            )

        xy = self.landmarks[index, :, :2].astype(np.float64)
        return np.column_stack((xy[:, 0] * width, xy[:, 1] * height))

    def confidence_for_frame(
        self,
        index: int,
        landmark_indices: Sequence[int] = KEY_SWING_LANDMARKS,
    ) -> float:
        """Mean visibility of the swing-relevant joints on one frame.

        Restricted to :data:`KEY_SWING_LANDMARKS` by default so a frame is not
        scored as confident because it located ten face points while losing
        both wrists.
        """
        if not 0 <= index < self.frame_count:
            raise PoseSequenceError(
                f"Frame {index} is outside 0..{self.frame_count - 1}"
            )
        if not self.detected[index]:
            return 0.0
        values = self.visibility[index, list(landmark_indices)]
        return float(np.mean(values)) if values.size else 0.0

    def mean_confidence(
        self, landmark_indices: Sequence[int] = KEY_SWING_LANDMARKS
    ) -> float:
        """Mean key-joint visibility across *detected* frames only.

        Undetected frames are excluded rather than counted as zero; they are
        already reported separately by :attr:`detection_rate`, and folding
        them in here would conflate "never seen" with "seen poorly".
        """
        if self.detected_count == 0:
            return 0.0
        values = self.visibility[np.asarray(self.detected), :][:, list(landmark_indices)]
        return float(np.mean(values)) if values.size else 0.0

    def longest_gap(self) -> Tuple[int, int]:
        """Longest run of consecutive undetected frames as (start, length).

        A long gap in the middle of a downswing is far more damaging than the
        same number of scattered misses, so the UI reports it on its own.
        """
        best_start, best_length = 0, 0
        current_start, current_length = 0, 0
        for index, ok in enumerate(self.detected):
            if ok:
                current_length = 0
                continue
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_start, best_length = current_start, current_length
        return best_start, best_length

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> Path:
        """Write the sequence to a compressed ``.npz``, atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")

        # Written as a 2-row array of strings so the npz stays self-describing
        # without dragging in pickle (allow_pickle is a footgun on load).
        metadata_keys = np.array(list(self.metadata.keys()), dtype=np.str_)
        metadata_values = np.array(list(self.metadata.values()), dtype=np.str_)

        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                format_version=np.array(POSE_FORMAT_VERSION),
                landmarks=self.landmarks,
                world_landmarks=self.world_landmarks,
                visibility=self.visibility,
                presence=self.presence,
                detected=self.detected,
                fps=np.array(self.fps),
                frame_width=np.array(self.frame_width),
                frame_height=np.array(self.frame_height),
                smoothing=np.array(self.smoothing, dtype=np.str_),
                metadata_keys=metadata_keys,
                metadata_values=metadata_values,
            )
        temp_path.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        path = Path(path)
        if not path.exists():
            raise PoseSequenceError(f"No pose file at {path}")

        try:
            # allow_pickle stays False: these files are data, and a pickled
            # array in one would be arbitrary code execution on load.
            with np.load(path, allow_pickle=False) as data:
                version = int(data["format_version"])
                if version != POSE_FORMAT_VERSION:
                    raise PoseSequenceError(
                        f"{path.name} was written in pose format v{version}, but "
                        f"this build reads v{POSE_FORMAT_VERSION}. Re-run pose "
                        "estimation for this swing."
                    )
                keys = [str(k) for k in data["metadata_keys"]]
                values = [str(v) for v in data["metadata_values"]]
                return cls(
                    landmarks=data["landmarks"],
                    world_landmarks=data["world_landmarks"],
                    visibility=data["visibility"],
                    presence=data["presence"],
                    detected=data["detected"].astype(bool),
                    fps=float(data["fps"]),
                    frame_width=int(data["frame_width"]),
                    frame_height=int(data["frame_height"]),
                    smoothing=str(data["smoothing"]),
                    metadata=dict(zip(keys, values)),
                )
        except PoseSequenceError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise PoseSequenceError(
                f"Could not read pose data from {path.name}: {exc}. "
                "Re-running pose estimation for this swing will rebuild it."
            ) from exc
