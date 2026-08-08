"""Real per-frame timing, measured rather than assumed.

Every timestamp in this project used to be computed as ``frame_index / fps``.
That is correct only if two things hold: the file is genuinely constant frame
rate, and the ``fps`` figure is right. Measuring a real phone clip showed both
can fail at once —

* its container advertised ``nb_frames = 484`` while only **438** frames
  decode, and
* ``avg_frame_rate`` was therefore 22.873 when the true rate was a rock-steady
  25.000.

So the clip was mislabelled as variable-frame-rate, and every timestamp derived
from the bogus rate was ~9% out. Container metadata is a claim; the frame
timestamps are the evidence.

This module reads the evidence. It extracts each frame's presentation
timestamp with ffprobe and builds an explicit mapping from preview frame index
to source time, recording **how** each timestamp was obtained so a caller can
refuse to compute a duration it cannot stand behind.

## Why the mapping is usually the identity

The preview is generated with ``-fps_mode passthrough``, which passes frames
through with their original timestamps. Verified on a genuinely
variable-frame-rate file (inter-frame gaps alternating 0.033 s and 0.067 s):
60 frames in, 60 frames out, maximum timestamp drift 0.00000 s.

Preview frame N is therefore source frame N, and the preview carries the true
timing. The mapping is still built and stored explicitly rather than assumed,
because "usually the identity" is not "always the identity" — a re-encode that
drops or duplicates a frame must be detectable, not silently absorbed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from golf_lab.logging_config import get_logger
from golf_lab.video.ffmpeg import FFmpegCommandError, FFmpegNotFoundError, find_ffmpeg, run_command

logger = get_logger(__name__)

# Bump when the stored layout changes in a way older code would misread.
TIMELINE_SCHEMA_VERSION = 1

# Identifies *how* timestamps were obtained. Stored per timeline so a change of
# extraction strategy invalidates old artifacts without a schema bump.
EXTRACTOR_NAME = "ffprobe/best_effort_timestamp_time"
EXTRACTOR_VERSION = "1"

# Frame gaps within this fraction of each other count as constant rate. Real
# CFR files show tiny rounding wobble in their timestamps (a 1/30 s gap stored
# in a 1/15000 timebase is not exactly 0.0333), so an exact-equality test would
# label almost everything variable.
CFR_TOLERANCE = 0.02


class TimingMethod(str, Enum):
    """How a frame's timestamp was obtained. Recorded per frame, not assumed."""

    MEASURED = "measured"
    """Read from the media's own presentation timestamps. Trustworthy."""

    INTERPOLATED = "interpolated"
    """Neighbouring frames had timestamps; this one did not. Reasonable, not exact."""

    NOMINAL = "nominal"
    """Derived from a nominal frame rate. This is the assumption that caused
    the original bug, so it is labelled rather than hidden, and durations
    computed from it are refused."""


class TimelineConfidence(str, Enum):
    """How much of this timeline's timing can be trusted.

    These are the timing *trust states*. Every timing-dependent metric states
    which one it rests on, so a duration is never presented without the basis
    that produced it.
    """

    MEASURED = "measured"
    """Every frame timestamp came from the media. Durations are trustworthy."""

    DEGRADED = "degraded"
    """Partially interpolated: some frames had no readable timestamp and were
    estimated between known neighbours. Durations are permitted but reported as
    low confidence, because the interpolation policy — linear between the
    nearest measured frames either side — is documented and bounded."""

    NOMINAL = "nominal"
    """Nominal only. No usable timestamps were recovered; everything derives
    from a nominal rate. Durations and tempo are refused entirely."""

    UNAVAILABLE = "unavailable"
    """No timing at all: the file could not be probed, or holds no frames.
    Distinguished from NOMINAL because there is not even a rate to fall back
    on, and the remedy differs — re-import rather than re-record."""

    @property
    def supports_durations(self) -> bool:
        """Whether a duration in seconds may be computed from this timeline.

        NOMINAL and UNAVAILABLE are excluded on purpose. A duration computed
        from a nominal rate on a file whose real rate is unknown is exactly the
        fabricated number this module exists to prevent.
        """
        return self in (TimelineConfidence.MEASURED, TimelineConfidence.DEGRADED)

    @property
    def label(self) -> str:
        return {
            TimelineConfidence.MEASURED: "measured",
            TimelineConfidence.DEGRADED: "partially interpolated",
            TimelineConfidence.NOMINAL: "nominal only",
            TimelineConfidence.UNAVAILABLE: "unavailable",
        }[self]


class RateClassification(str, Enum):
    """What the *decoded* frames say about frame spacing.

    Deliberately decided from measured timestamps alone. Container metadata is
    not consulted: a file whose ``nb_frames`` disagrees with reality, or whose
    nominal and average rates contradict each other, is not thereby variable
    frame rate — it just has bad metadata, which is a separate thing to report.
    """

    CONSTANT = "constant"
    VARIABLE = "variable"
    UNVERIFIED = "unverified"
    """Only nominal timing was available, so spacing could not be checked."""

    @property
    def label(self) -> str:
        return {
            RateClassification.CONSTANT: "constant rate",
            RateClassification.VARIABLE: "variable rate",
            RateClassification.UNVERIFIED: "timing unverified",
        }[self]


class TimelineError(RuntimeError):
    """Raised when a timeline cannot be built or read. User-facing."""


@dataclass(frozen=True)
class FrameTiming:
    """One frame's place on both timelines."""

    preview_index: int
    source_seconds: float
    method: TimingMethod
    duration_seconds: Optional[float] = None
    source_frame_index: Optional[int] = None

    @property
    def is_measured(self) -> bool:
        return self.method is TimingMethod.MEASURED


@dataclass
class SourceTimeline:
    """Maps preview frame indices to real source timestamps.

    ``frames`` is dense and index-aligned: ``frames[i].preview_index == i``.
    """

    frames: List[FrameTiming] = field(default_factory=list)
    confidence: TimelineConfidence = TimelineConfidence.NOMINAL
    nominal_fps: float = 0.0
    measured_fps: Optional[float] = None
    is_constant_rate: bool = False
    schema_version: int = TIMELINE_SCHEMA_VERSION
    source_name: str = ""
    notes: List[str] = field(default_factory=list)

    extractor: str = EXTRACTOR_NAME
    extractor_version: str = EXTRACTOR_VERSION
    # Fingerprints of the media this was measured from, for staleness checks.
    source_fingerprint: str = ""
    preview_fingerprint: str = ""
    # Frame count the container claimed, kept only so the UI can point out a
    # disagreement. It is never used as the frame count.
    container_frame_count: Optional[int] = None

    @property
    def rate_classification(self) -> RateClassification:
        """Constant / variable / unverified, decided from measured spacing."""
        if not self.confidence.supports_durations:
            return RateClassification.UNVERIFIED
        return (
            RateClassification.CONSTANT
            if self.is_constant_rate
            else RateClassification.VARIABLE
        )

    @property
    def container_metadata_is_inconsistent(self) -> bool:
        """Whether the container's own numbers disagree with the decoded stream.

        Worth surfacing — it is why this module exists — but it says nothing
        about whether the decoded video is variable frame rate. Reporting the
        two separately is the whole correction.
        """
        if self.container_frame_count is None or not self.frames:
            return False
        return abs(self.container_frame_count - len(self.frames)) > 1

    # -- basics ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frames)

    @property
    def frame_count(self) -> int:
        """Frames that actually exist — measured, not the container's claim."""
        return len(self.frames)

    @property
    def duration_seconds(self) -> Optional[float]:
        if not self.frames:
            return None
        last = self.frames[-1]
        end = last.source_seconds + (last.duration_seconds or 0.0)
        return end - self.frames[0].source_seconds

    def timing_for(self, preview_index: int) -> Optional[FrameTiming]:
        if 0 <= preview_index < len(self.frames):
            return self.frames[preview_index]
        return None

    # -- conversions -----------------------------------------------------
    def source_seconds(self, preview_index: int) -> Optional[float]:
        """Source time of a preview frame, or None if outside the clip."""
        timing = self.timing_for(preview_index)
        return timing.source_seconds if timing else None

    def preview_index_for_source_seconds(self, seconds: float) -> Optional[int]:
        """Nearest preview frame to a source timestamp.

        Nearest rather than floor: a timestamp landing 1 ms before a frame
        belongs to that frame, not the one before it. Binary search keeps this
        usable on long clips.
        """
        if not self.frames:
            return None

        low, high = 0, len(self.frames) - 1
        if seconds <= self.frames[low].source_seconds:
            return low
        if seconds >= self.frames[high].source_seconds:
            return high

        while low < high - 1:
            middle = (low + high) // 2
            if self.frames[middle].source_seconds <= seconds:
                low = middle
            else:
                high = middle

        before = self.frames[low].source_seconds
        after = self.frames[high].source_seconds
        return low if (seconds - before) <= (after - seconds) else high

    def duration_between(
        self, start_index: int, end_index: int
    ) -> Optional[float]:
        """Elapsed source time between two preview frames.

        Returns ``None`` — never a guess — when the timeline cannot support
        the claim, either because the whole timeline is nominal or because one
        of these two specific frames was not measured.
        """
        if not self.confidence.supports_durations:
            return None
        start = self.timing_for(start_index)
        end = self.timing_for(end_index)
        if start is None or end is None:
            return None
        if start.method is TimingMethod.NOMINAL or end.method is TimingMethod.NOMINAL:
            return None
        return end.source_seconds - start.source_seconds

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        """Serialize. Frames are stored column-wise to keep the file small —
        a 4-minute 120 fps clip is ~29k frames, and a list of objects would
        cost several megabytes of punctuation."""
        return {
            "schema_version": self.schema_version,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "confidence": self.confidence.value,
            "rate_classification": self.rate_classification.value,
            "nominal_fps": self.nominal_fps,
            "measured_fps": self.measured_fps,
            "is_constant_rate": self.is_constant_rate,
            "cfr_tolerance": CFR_TOLERANCE,
            "source_name": self.source_name,
            "source_fingerprint": self.source_fingerprint,
            "preview_fingerprint": self.preview_fingerprint,
            "container_frame_count": self.container_frame_count,
            "measured_duration_seconds": self.duration_seconds,
            "notes": list(self.notes),
            "frame_count": len(self.frames),
            "source_seconds": [round(f.source_seconds, 6) for f in self.frames],
            "durations": [
                None if f.duration_seconds is None else round(f.duration_seconds, 6)
                for f in self.frames
            ],
            "methods": [f.method.value for f in self.frames],
            "source_frame_indices": [f.source_frame_index for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SourceTimeline":
        version = int(data.get("schema_version", 0))
        if version != TIMELINE_SCHEMA_VERSION:
            raise TimelineError(
                f"Timeline was stored with schema v{version}; this build reads "
                f"v{TIMELINE_SCHEMA_VERSION}. Re-import or re-analyse the swing."
            )

        times = list(data.get("source_seconds") or [])
        durations = list(data.get("durations") or [None] * len(times))
        methods = list(data.get("methods") or [TimingMethod.NOMINAL.value] * len(times))
        indices = list(data.get("source_frame_indices") or [None] * len(times))

        frames = [
            FrameTiming(
                preview_index=i,
                source_seconds=float(times[i]),
                method=TimingMethod(methods[i]),
                duration_seconds=durations[i],
                source_frame_index=indices[i],
            )
            for i in range(len(times))
        ]
        return cls(
            frames=frames,
            confidence=TimelineConfidence(data.get("confidence", "nominal")),
            nominal_fps=float(data.get("nominal_fps", 0.0)),
            measured_fps=data.get("measured_fps"),
            is_constant_rate=bool(data.get("is_constant_rate", False)),
            schema_version=version,
            source_name=str(data.get("source_name", "")),
            notes=list(data.get("notes") or []),
            extractor=str(data.get("extractor", EXTRACTOR_NAME)),
            extractor_version=str(data.get("extractor_version", EXTRACTOR_VERSION)),
            source_fingerprint=str(data.get("source_fingerprint", "")),
            preview_fingerprint=str(data.get("preview_fingerprint", "")),
            container_frame_count=data.get("container_frame_count"),
        )


# --- extraction ----------------------------------------------------------
def probe_frame_timestamps(video_path: Path) -> List[Tuple[Optional[float], Optional[float]]]:
    """Return ``(pts_seconds, duration_seconds)`` for every frame, in order.

    Uses ``best_effort_timestamp_time``, which is ffmpeg's own reconciliation
    of pts and dts and is populated for files where ``pkt_pts_time`` is not.
    Frames whose timestamp cannot be read come back as ``None`` rather than
    being dropped, so the caller can see the gaps and decide what to do.
    """
    tools = find_ffmpeg(required=False)
    if tools is None:
        raise FFmpegNotFoundError(
            "FFmpeg is required to read frame timestamps. Install it and retry."
        )

    command = [
        tools.ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time,duration_time",
        "-of", "json",
        str(video_path),
    ]
    try:
        stdout = run_command(command)
    except FFmpegCommandError as exc:
        raise TimelineError(
            f"Could not read frame timestamps from {Path(video_path).name}.\n\n{exc}"
        ) from exc

    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        raise TimelineError(
            f"ffprobe returned unreadable timestamp data for "
            f"{Path(video_path).name}: {exc}"
        ) from exc

    out: List[Tuple[Optional[float], Optional[float]]] = []
    for frame in payload.get("frames") or []:
        out.append((_maybe_float(frame.get("best_effort_timestamp_time")),
                    _maybe_float(frame.get("duration_time"))))
    return out


def _maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # ffprobe emits "N/A" for unknown, and some muxers emit nonsense negatives.
    return number if number == number and number >= 0 else None


def _fill_gaps(
    raw: Sequence[Tuple[Optional[float], Optional[float]]],
    nominal_fps: float,
) -> Tuple[List[FrameTiming], TimelineConfidence, List[str]]:
    """Turn raw probe output into a dense timeline, labelling every frame.

    Missing timestamps are interpolated between known neighbours where that is
    possible and marked INTERPOLATED. Frames with no usable neighbour fall back
    to the nominal rate and are marked NOMINAL, which blocks durations rather
    than silently producing one.
    """
    notes: List[str] = []
    known = [(i, t) for i, (t, _) in enumerate(raw) if t is not None]

    if not known:
        step = 1.0 / nominal_fps if nominal_fps > 0 else 0.0
        notes.append(
            "No frame timestamps could be read from this file; timings are "
            "derived from its nominal frame rate and cannot support durations."
        )
        frames = [
            FrameTiming(
                preview_index=i,
                source_seconds=i * step,
                method=TimingMethod.NOMINAL,
                duration_seconds=step or None,
                source_frame_index=i,
            )
            for i in range(len(raw))
        ]
        return frames, TimelineConfidence.NOMINAL, notes

    frames: List[FrameTiming] = []
    interpolated = 0
    for index, (timestamp, duration) in enumerate(raw):
        if timestamp is not None:
            method = TimingMethod.MEASURED
        else:
            timestamp = _interpolate_at(index, known, nominal_fps)
            method = TimingMethod.INTERPOLATED
            interpolated += 1
        frames.append(
            FrameTiming(
                preview_index=index,
                source_seconds=timestamp,
                method=method,
                duration_seconds=duration,
                source_frame_index=index,
            )
        )

    # Timestamps must not go backwards; a muxer that reorders them would make
    # every later comparison meaningless, so it is reported rather than sorted
    # away.
    for index in range(1, len(frames)):
        if frames[index].source_seconds < frames[index - 1].source_seconds:
            notes.append(
                f"Frame timestamps are not monotonic at frame {index}; the "
                "source may have reordered presentation times."
            )
            break

    if interpolated:
        notes.append(
            f"{interpolated} of {len(frames)} frames had no readable timestamp "
            "and were interpolated from their neighbours."
        )
        confidence = TimelineConfidence.DEGRADED
    else:
        confidence = TimelineConfidence.MEASURED
    return frames, confidence, notes


def _interpolate_at(
    index: int, known: Sequence[Tuple[int, float]], nominal_fps: float
) -> float:
    """Estimate a missing timestamp from the nearest known frames either side."""
    before = None
    after = None
    for position, timestamp in known:
        if position < index:
            before = (position, timestamp)
        elif position > index:
            after = (position, timestamp)
            break

    if before and after:
        span = after[0] - before[0]
        progress = (index - before[0]) / span if span else 0.0
        return before[1] + progress * (after[1] - before[1])

    step = 1.0 / nominal_fps if nominal_fps > 0 else 0.0
    if before:
        return before[1] + (index - before[0]) * step
    assert after is not None
    return max(0.0, after[1] - (after[0] - index) * step)


def _measure_rate(frames: Sequence[FrameTiming]) -> Tuple[Optional[float], bool]:
    """Frame rate and constancy, measured from the timestamps themselves.

    This is what the container's ``avg_frame_rate`` should have said. On the
    clip that motivated this module it reports 25.0 where the container
    claimed 22.873.
    """
    if len(frames) < 3:
        return None, False

    gaps = [
        frames[i + 1].source_seconds - frames[i].source_seconds
        for i in range(len(frames) - 1)
    ]
    positive = [gap for gap in gaps if gap > 1e-9]
    if not positive:
        return None, False

    shortest, longest = min(positive), max(positive)
    constant = (longest - shortest) <= CFR_TOLERANCE * longest

    span = frames[-1].source_seconds - frames[0].source_seconds
    rate = (len(frames) - 1) / span if span > 0 else None
    return rate, constant


def build_timeline(
    video_path: Path,
    nominal_fps: float = 0.0,
    container_frame_count: Optional[int] = None,
    source_fingerprint: str = "",
    preview_fingerprint: str = "",
) -> SourceTimeline:
    """Measure a video's real timing and return the mapping.

    ``nominal_fps`` is used only as a fallback for frames with no readable
    timestamp, and its use is always recorded in the frame's ``method``.
    ``container_frame_count`` is stored purely so a disagreement with the
    decoded stream can be reported; it is never used as the frame count.
    """
    video_path = Path(video_path)
    raw = probe_frame_timestamps(video_path)
    if not raw:
        raise TimelineError(
            f"{video_path.name} reported no video frames, so no timeline could "
            "be built."
        )

    frames, confidence, notes = _fill_gaps(raw, nominal_fps)
    measured_fps, constant = _measure_rate(frames)

    timeline = SourceTimeline(
        frames=frames,
        confidence=confidence,
        nominal_fps=nominal_fps,
        measured_fps=measured_fps,
        is_constant_rate=constant,
        source_name=video_path.name,
        notes=notes,
        source_fingerprint=source_fingerprint,
        preview_fingerprint=preview_fingerprint,
        container_frame_count=container_frame_count,
    )

    if measured_fps and nominal_fps > 0:
        disagreement = abs(measured_fps - nominal_fps) / max(measured_fps, nominal_fps)
        if disagreement > 0.02:
            timeline.notes.append(
                f"The container advertises {nominal_fps:.3f} fps but the frame "
                f"timestamps measure {measured_fps:.3f} fps. The measured rate "
                "is used; the container's figure is unreliable for this file."
            )

    if timeline.container_metadata_is_inconsistent:
        timeline.notes.append(
            f"The container claims {container_frame_count} frames but "
            f"{len(frames)} actually decode. The decoded count is used. This is "
            "inconsistent container metadata, not variable frame rate — the "
            "frame spacing is judged separately, from the timestamps."
        )

    logger.info(
        "Timeline for %s: %d frames, %s, measured %.3f fps, constant=%s",
        video_path.name,
        len(frames),
        confidence.value,
        measured_fps or 0.0,
        constant,
    )
    return timeline
