"""Persisting pose results per swing, and knowing when they are stale.

Layout added to each swing directory:

    data/swings/<swing_id>/
        pose_raw.npz        exactly what the backend produced
        pose_smoothed.npz   filtered copy, regenerable from raw
        pose_info.json      provenance: model, versions, fingerprint, counts

Both sequences are kept. Raw is the evidence; smoothed is a convenience. If
they ever disagree about what was detected, raw wins.

Staleness is decided by comparing the stored fingerprint and versions against
the current ones. The alternative — assuming a cached result still applies —
is how you end up showing a skeleton computed from a video the user has since
replaced, which the roadmap explicitly calls out as unacceptable ("stale
results are flagged rather than trusted").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from golf_lab.config import ANALYSIS_VERSION, APP_VERSION
from golf_lab.logging_config import get_logger
from golf_lab.models.video import SwingRecord
from golf_lab.pose.sequence import PoseSequence, PoseSequenceError
from golf_lab.storage.file_repository import swing_dir
from golf_lab.video.frame_cache import file_fingerprint

logger = get_logger(__name__)

RAW_FILENAME = "pose_raw.npz"
SMOOTHED_FILENAME = "pose_smoothed.npz"
INFO_FILENAME = "pose_info.json"


@dataclass
class PoseAnalysisInfo:
    """Provenance for one stored pose analysis."""

    swing_id: str
    model_key: str
    model_filename: str
    model_sha256: str
    backend: str
    device: str
    mediapipe_version: str

    video_fingerprint: str
    frame_count: int
    detected_count: int
    detection_rate: float
    mean_confidence: float
    longest_gap_frames: int

    smoothing: str
    analysis_version: str = ANALYSIS_VERSION
    app_version: str = APP_VERSION
    pose_format_version: int = 1
    created_at: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        if not data.get("created_at"):
            data["created_at"] = datetime.now(timezone.utc).isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PoseAnalysisInfo":
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)


# --- paths ---------------------------------------------------------------
def raw_path(swing_id: str, root: Optional[Path] = None) -> Path:
    return swing_dir(swing_id, root) / RAW_FILENAME


def smoothed_path(swing_id: str, root: Optional[Path] = None) -> Path:
    return swing_dir(swing_id, root) / SMOOTHED_FILENAME


def info_path(swing_id: str, root: Optional[Path] = None) -> Path:
    return swing_dir(swing_id, root) / INFO_FILENAME


def has_pose_analysis(swing_id: str, root: Optional[Path] = None) -> bool:
    return raw_path(swing_id, root).exists() and info_path(swing_id, root).exists()


# --- writing -------------------------------------------------------------
def save_pose_analysis(
    swing_id: str,
    raw: PoseSequence,
    smoothed: Optional[PoseSequence],
    info: PoseAnalysisInfo,
    root: Optional[Path] = None,
) -> Path:
    """Write raw, smoothed, and provenance for one swing.

    ``pose_info.json`` is written *last*: it is the marker that a complete
    analysis exists, so a crash midway leaves an obviously-incomplete state
    rather than provenance pointing at arrays that were never finished.
    """
    directory = swing_dir(swing_id, root)
    directory.mkdir(parents=True, exist_ok=True)

    raw.save(raw_path(swing_id, root))
    if smoothed is not None:
        smoothed.save(smoothed_path(swing_id, root))

    path = info_path(swing_id, root)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    temp_path.replace(path)

    logger.info(
        "Saved pose analysis for %s (%d/%d frames detected)",
        swing_id,
        info.detected_count,
        info.frame_count,
    )
    return path


def build_info(
    swing_id: str,
    raw: PoseSequence,
    smoothed: Optional[PoseSequence],
    video_path: Path,
    model_key: str,
    model_filename: str,
    model_sha256: str,
    backend: str,
    device: str,
    mediapipe_version: str,
) -> PoseAnalysisInfo:
    """Assemble provenance from a finished run."""
    _, gap_length = raw.longest_gap()
    try:
        elapsed = float(raw.metadata.get("elapsed_seconds", 0.0))
    except (TypeError, ValueError):
        elapsed = 0.0

    return PoseAnalysisInfo(
        swing_id=swing_id,
        model_key=model_key,
        model_filename=model_filename,
        model_sha256=model_sha256,
        backend=backend,
        device=device,
        mediapipe_version=mediapipe_version,
        video_fingerprint=file_fingerprint(video_path),
        frame_count=raw.frame_count,
        detected_count=raw.detected_count,
        detection_rate=raw.detection_rate,
        mean_confidence=raw.mean_confidence(),
        longest_gap_frames=int(gap_length),
        smoothing=smoothed.smoothing if smoothed is not None else "none",
        created_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed,
    )


# --- reading -------------------------------------------------------------
def load_info(swing_id: str, root: Optional[Path] = None) -> Optional[PoseAnalysisInfo]:
    path = info_path(swing_id, root)
    if not path.exists():
        return None
    try:
        return PoseAnalysisInfo.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Unreadable pose info for %s: %s", swing_id, exc)
        return None


def load_raw(swing_id: str, root: Optional[Path] = None) -> Optional[PoseSequence]:
    path = raw_path(swing_id, root)
    if not path.exists():
        return None
    try:
        return PoseSequence.load(path)
    except PoseSequenceError as exc:
        logger.warning("Could not load raw pose for %s: %s", swing_id, exc)
        return None


def load_smoothed(swing_id: str, root: Optional[Path] = None) -> Optional[PoseSequence]:
    path = smoothed_path(swing_id, root)
    if not path.exists():
        return None
    try:
        return PoseSequence.load(path)
    except PoseSequenceError as exc:
        logger.warning("Could not load smoothed pose for %s: %s", swing_id, exc)
        return None


def delete_pose_analysis(swing_id: str, root: Optional[Path] = None) -> None:
    """Remove a stored analysis. The video itself is never touched."""
    for path in (
        raw_path(swing_id, root),
        smoothed_path(swing_id, root),
        info_path(swing_id, root),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path, exc)


# --- staleness -----------------------------------------------------------
def staleness_reasons(
    info: Optional[PoseAnalysisInfo],
    video_path: Path,
    record: Optional[SwingRecord] = None,
) -> List[str]:
    """Why a stored analysis should not be trusted, as user-facing sentences.

    An empty list means the cached result is current. Each reason is phrased
    for display, because this is exactly the sort of thing that gets shown to
    the user rather than logged and forgotten.
    """
    if info is None:
        return ["No pose analysis has been run for this swing yet."]

    reasons: List[str] = []

    try:
        current_fingerprint = file_fingerprint(video_path)
    except OSError:
        return ["The video file for this swing could not be read."]

    if info.video_fingerprint and info.video_fingerprint != current_fingerprint:
        reasons.append(
            "The video file has changed since this analysis was computed."
        )
    if info.analysis_version != ANALYSIS_VERSION:
        reasons.append(
            f"It was computed with analysis version {info.analysis_version}; "
            f"this build uses version {ANALYSIS_VERSION}."
        )
    # Deliberately NOT compared: info.frame_count against
    # record.video.frame_count. Pose runs on the *preview*, and for a
    # variable-frame-rate source the preview legitimately has a different
    # frame count from the original — FFmpeg normalizes VFR to CFR, so a
    # 484-frame VFR original becomes a 438-frame preview. Comparing the two
    # marked every VFR swing permanently stale the instant it was analysed,
    # with no way to clear it, which trains the user to ignore the warning
    # exactly where it matters most.
    #
    # The fingerprint above already covers the real case: it is computed on
    # the file that was actually analysed, so any change to that file — a
    # re-import, a regenerated preview — invalidates the analysis.
    #
    # The frame-count mismatch between original and preview is real and is
    # already reported, once, where it belongs: as the swing's `needs_review`
    # status at import time.
    return reasons


def is_stale(
    info: Optional[PoseAnalysisInfo],
    video_path: Path,
    record: Optional[SwingRecord] = None,
) -> bool:
    return bool(staleness_reasons(info, video_path, record))
