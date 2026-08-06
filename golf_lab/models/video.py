"""Typed data models for videos and swing records.

These are the serialization contract for `metadata.json` inside each swing
directory. Adding fields is fine; renaming or removing them is a breaking
change that should bump `analysis_version` in the stored record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CameraView(str, Enum):
    """Which camera angle the swing was recorded from.

    Metrics are gated on this: hand depth means nothing face-on, lateral hip
    sway means nothing down-the-line.
    """

    FACE_ON = "face_on"
    DOWN_THE_LINE = "down_the_line"
    OTHER = "other"
    UNKNOWN = "unknown"


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class ShotShape(str, Enum):
    """User-reported observed shape. Never inferred automatically."""

    STRAIGHT = "straight"
    FADE = "fade"
    DRAW = "draw"
    SLICE = "slice"
    HOOK = "hook"
    PUSH = "push"
    PULL = "pull"
    UNKNOWN = "unknown"


class SwingStatus(str, Enum):
    """Coarse processing state shown in the UI."""

    NOT_PROCESSED = "not_processed"
    PROCESSING = "processing"
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    LOW_CONFIDENCE = "low_confidence"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    """Technical properties of a video file, as probed by ffprobe/OpenCV.

    `width`/`height` are DISPLAY dimensions, i.e. after applying any rotation
    from the container's display matrix. `coded_width`/`coded_height` are the
    raw stored dimensions before rotation.
    """

    path: str
    container_format: Optional[str] = None
    codec_name: Optional[str] = None
    audio_codec_name: Optional[str] = None
    has_audio: bool = False

    coded_width: int
    coded_height: int
    width: int
    height: int
    rotation_degrees: int = 0

    fps: float
    frame_count: int
    duration_seconds: float
    frame_count_is_estimated: bool = False

    probe_source: str = "unknown"  # "ffprobe" or "opencv"

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    def timestamp_for_frame(self, frame_index: int) -> float:
        """Presentation time in seconds of a 0-based frame index.

        Uses the constant-frame-rate assumption, which holds for the phone
        footage this app targets. Variable-frame-rate video will drift; that
        limitation is documented in docs/LIMITATIONS.md.
        """
        if self.fps <= 0:
            return 0.0
        return frame_index / self.fps

    def frame_for_timestamp(self, seconds: float) -> int:
        if self.fps <= 0:
            return 0
        index = int(round(seconds * self.fps))
        return max(0, min(index, max(self.frame_count - 1, 0)))


class SwingContext(BaseModel):
    """What the user tells us about the shot. Never guessed by the app."""

    club: str = "Unknown"
    camera_view: CameraView = CameraView.UNKNOWN
    handedness: Handedness = Handedness.RIGHT
    shot_shape: ShotShape = ShotShape.UNKNOWN
    typical_miss: str = ""
    carry_yards: Optional[float] = None
    notes: str = ""


class SwingRecord(BaseModel):
    """The complete persisted description of one imported swing."""

    swing_id: str
    original_filename: str
    imported_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Paths are stored relative to the swing directory so the data folder
    # remains portable if it is moved or copied to another machine.
    original_relpath: str
    preview_relpath: Optional[str] = None
    thumbnail_relpath: Optional[str] = None

    video: VideoMetadata
    context: SwingContext = Field(default_factory=SwingContext)

    status: SwingStatus = SwingStatus.NOT_PROCESSED
    status_detail: str = ""

    app_version: str
    analysis_version: str

    model_config = {"use_enum_values": False}
