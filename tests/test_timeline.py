"""Source-timeline mapping: measured timing, and honest refusal without it.

The bug this module exists to prevent is specific and was observed on real
footage: a container advertising 484 frames at 22.873 fps for a file that
actually holds 438 frames at a rock-steady 25.000. Every timestamp derived
from the container was ~9% wrong, and the clip was wrongly flagged as
variable-frame-rate.

So these tests care about two things above all — that timings come from the
frames rather than the container, and that a timeline which *cannot* measure
refuses to produce durations instead of inventing them.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from golf_lab.video.timeline import (
    CFR_TOLERANCE,
    TIMELINE_SCHEMA_VERSION,
    FrameTiming,
    SourceTimeline,
    TimelineConfidence,
    TimelineError,
    TimingMethod,
    _fill_gaps,
    _measure_rate,
    build_timeline,
    probe_frame_timestamps,
)


def _timeline(times, methods=None, durations=None, **kwargs):
    methods = methods or [TimingMethod.MEASURED] * len(times)
    durations = durations or [None] * len(times)
    frames = [
        FrameTiming(
            preview_index=i,
            source_seconds=t,
            method=methods[i],
            duration_seconds=durations[i],
            source_frame_index=i,
        )
        for i, t in enumerate(times)
    ]
    kwargs.setdefault("confidence", TimelineConfidence.MEASURED)
    return SourceTimeline(frames=frames, **kwargs)


# --- fixtures generating real media -------------------------------------
@pytest.fixture(scope="module")
def cfr_video(tmp_path_factory):
    """A genuinely constant-frame-rate clip."""
    from golf_lab.video.ffmpeg import find_ffmpeg

    tools = find_ffmpeg(required=False)
    if tools is None:
        pytest.skip("FFmpeg not installed")
    path = tmp_path_factory.mktemp("cfr") / "cfr.mp4"
    subprocess.run(
        [
            tools.ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def vfr_video(tmp_path_factory):
    """A genuinely variable-frame-rate clip: gaps alternate 1/30 s and 2/30 s.

    Built with a non-linear setpts expression rather than by dropping frames,
    because the `select` filter renumbers timestamps uniformly and would
    silently produce a CFR file — which would make this fixture test nothing.
    """
    from golf_lab.video.ffmpeg import find_ffmpeg

    tools = find_ffmpeg(required=False)
    if tools is None:
        pytest.skip("FFmpeg not installed")
    path = tmp_path_factory.mktemp("vfr") / "vfr.mp4"
    subprocess.run(
        [
            tools.ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
            "-vf", "setpts='(N+0.4*N*N/60)/30/TB'",
            "-fps_mode", "passthrough",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


class TestProbing:
    def test_reads_a_timestamp_for_every_frame(self, cfr_video):
        raw = probe_frame_timestamps(cfr_video)
        assert len(raw) == 60
        assert all(t is not None for t, _ in raw)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((TimelineError, Exception)):
            probe_frame_timestamps(tmp_path / "nope.mp4")


class TestConstantFrameRate:
    def test_measures_the_true_rate(self, cfr_video):
        timeline = build_timeline(cfr_video, nominal_fps=30.0)
        assert timeline.frame_count == 60
        assert timeline.confidence is TimelineConfidence.MEASURED
        assert timeline.is_constant_rate
        assert timeline.measured_fps == pytest.approx(30.0, abs=0.05)

    def test_frame_count_comes_from_frames_not_the_container(self, cfr_video):
        # The whole point: 484-vs-438 was a container lie.
        raw = probe_frame_timestamps(cfr_video)
        assert build_timeline(cfr_video, 30.0).frame_count == len(raw)

    def test_timestamps_are_evenly_spaced(self, cfr_video):
        timeline = build_timeline(cfr_video, nominal_fps=30.0)
        gaps = [
            timeline.frames[i + 1].source_seconds - timeline.frames[i].source_seconds
            for i in range(len(timeline) - 1)
        ]
        assert max(gaps) - min(gaps) < 0.005

    def test_every_frame_is_measured(self, cfr_video):
        timeline = build_timeline(cfr_video, nominal_fps=30.0)
        assert all(f.is_measured for f in timeline.frames)


class TestVariableFrameRate:
    def test_detected_as_variable(self, vfr_video):
        timeline = build_timeline(vfr_video, nominal_fps=30.0)
        assert timeline.confidence is TimelineConfidence.MEASURED
        assert not timeline.is_constant_rate

    def test_gaps_really_do_vary(self, vfr_video):
        timeline = build_timeline(vfr_video, nominal_fps=30.0)
        gaps = [
            round(timeline.frames[i + 1].source_seconds - timeline.frames[i].source_seconds, 5)
            for i in range(len(timeline) - 1)
        ]
        assert len(set(gaps)) > 1, "fixture is not actually variable-frame-rate"

    def test_durations_use_real_timestamps_not_a_nominal_rate(self, vfr_video):
        # index/fps would give a uniform answer; the real answer is not uniform.
        timeline = build_timeline(vfr_video, nominal_fps=30.0)
        first_half = timeline.duration_between(0, 10)
        second_half = timeline.duration_between(len(timeline) - 11, len(timeline) - 1)

        assert first_half is not None and second_half is not None
        assert first_half != pytest.approx(second_half, rel=0.05), (
            "equal spans imply a nominal rate was used instead of real timestamps"
        )

    def test_durations_are_still_allowed_when_measured(self, vfr_video):
        # VFR is not itself a reason to refuse: measured is measured.
        timeline = build_timeline(vfr_video, nominal_fps=30.0)
        assert timeline.confidence.supports_durations
        assert timeline.duration_between(0, len(timeline) - 1) is not None


class TestMissingAndIrregularTimestamps:
    def test_isolated_gaps_are_interpolated_and_labelled(self):
        raw = [(0.0, None), (None, None), (0.2, None), (0.3, None)]
        frames, confidence, notes = _fill_gaps(raw, nominal_fps=10.0)

        assert confidence is TimelineConfidence.DEGRADED
        assert frames[1].method is TimingMethod.INTERPOLATED
        assert frames[1].source_seconds == pytest.approx(0.1)
        assert any("interpolated" in n for n in notes)

    def test_degraded_timelines_still_support_durations(self):
        raw = [(0.0, None), (None, None), (0.2, None)]
        _, confidence, _ = _fill_gaps(raw, nominal_fps=10.0)
        assert confidence.supports_durations

    def test_no_timestamps_at_all_falls_back_and_says_so(self):
        raw = [(None, None)] * 5
        frames, confidence, notes = _fill_gaps(raw, nominal_fps=25.0)

        assert confidence is TimelineConfidence.NOMINAL
        assert all(f.method is TimingMethod.NOMINAL for f in frames)
        assert any("nominal frame rate" in n for n in notes)

    def test_nominal_timelines_refuse_durations(self):
        # This is the core guarantee. A duration from a nominal rate on a file
        # whose real rate is unknown is exactly the fabricated number this
        # module exists to prevent.
        raw = [(None, None)] * 5
        frames, confidence, _ = _fill_gaps(raw, nominal_fps=25.0)
        timeline = SourceTimeline(frames=frames, confidence=confidence)

        assert not confidence.supports_durations
        assert timeline.duration_between(0, 4) is None

    def test_a_single_nominal_frame_blocks_only_spans_touching_it(self):
        times = [0.0, 0.1, 0.2, 0.3]
        methods = [
            TimingMethod.MEASURED,
            TimingMethod.MEASURED,
            TimingMethod.NOMINAL,
            TimingMethod.MEASURED,
        ]
        timeline = _timeline(times, methods, confidence=TimelineConfidence.DEGRADED)

        assert timeline.duration_between(0, 1) == pytest.approx(0.1)
        assert timeline.duration_between(0, 2) is None
        assert timeline.duration_between(2, 3) is None

    def test_non_monotonic_timestamps_are_reported(self):
        raw = [(0.0, None), (0.2, None), (0.1, None), (0.3, None)]
        _, _, notes = _fill_gaps(raw, nominal_fps=10.0)
        assert any("monotonic" in n for n in notes)

    def test_negative_and_unparseable_timestamps_are_treated_as_missing(self):
        from golf_lab.video.timeline import _maybe_float

        assert _maybe_float("N/A") is None
        assert _maybe_float(None) is None
        assert _maybe_float("-1.5") is None
        assert _maybe_float("0.25") == pytest.approx(0.25)


class TestRateMeasurement:
    def test_constant_rate_recognised(self):
        frames = _timeline([i / 30.0 for i in range(30)]).frames
        rate, constant = _measure_rate(frames)
        assert constant
        assert rate == pytest.approx(30.0, abs=0.01)

    def test_variable_rate_recognised(self):
        times = [0.0]
        for i in range(1, 20):
            times.append(times[-1] + (0.02 if i % 2 else 0.06))
        rate, constant = _measure_rate(_timeline(times).frames)
        assert not constant

    def test_small_rounding_wobble_is_still_constant(self):
        # Real CFR files store 1/30 s in an integer timebase, so gaps wobble
        # slightly. An exact-equality test would call everything variable.
        times, t = [0.0], 0.0
        for i in range(30):
            t += 1 / 30.0 + (0.0001 if i % 2 else -0.0001)
            times.append(t)
        _, constant = _measure_rate(_timeline(times).frames)
        assert constant

    def test_too_few_frames_measures_nothing(self):
        assert _measure_rate(_timeline([0.0, 0.1]).frames) == (None, False)


class TestConversions:
    def test_source_seconds_round_trips_through_index(self, cfr_video):
        timeline = build_timeline(cfr_video, nominal_fps=30.0)
        for index in (0, 7, 30, len(timeline) - 1):
            seconds = timeline.source_seconds(index)
            assert timeline.preview_index_for_source_seconds(seconds) == index

    def test_lookup_picks_the_nearest_frame_not_the_earlier_one(self):
        timeline = _timeline([0.0, 0.1, 0.2, 0.3])
        # 0.099 is 1 ms before frame 1 and 99 ms after frame 0.
        assert timeline.preview_index_for_source_seconds(0.099) == 1
        assert timeline.preview_index_for_source_seconds(0.051) == 1
        assert timeline.preview_index_for_source_seconds(0.049) == 0

    def test_lookup_clamps_outside_the_clip(self):
        timeline = _timeline([0.0, 0.1, 0.2])
        assert timeline.preview_index_for_source_seconds(-5.0) == 0
        assert timeline.preview_index_for_source_seconds(99.0) == 2

    def test_out_of_range_index_returns_none(self):
        timeline = _timeline([0.0, 0.1])
        assert timeline.source_seconds(5) is None
        assert timeline.timing_for(-1) is None

    def test_empty_timeline_is_safe(self):
        timeline = SourceTimeline()
        assert timeline.frame_count == 0
        assert timeline.duration_seconds is None
        assert timeline.preview_index_for_source_seconds(1.0) is None
        assert timeline.duration_between(0, 1) is None


class TestSerialization:
    def test_round_trip_preserves_everything(self, cfr_video):
        original = build_timeline(cfr_video, nominal_fps=30.0)
        restored = SourceTimeline.from_dict(json.loads(json.dumps(original.to_dict())))

        assert restored.frame_count == original.frame_count
        assert restored.confidence is original.confidence
        assert restored.is_constant_rate == original.is_constant_rate
        assert restored.measured_fps == pytest.approx(original.measured_fps)
        for a, b in zip(restored.frames, original.frames):
            assert a.preview_index == b.preview_index
            assert a.source_seconds == pytest.approx(b.source_seconds, abs=1e-6)
            assert a.method is b.method

    def test_methods_survive_the_round_trip(self):
        timeline = _timeline(
            [0.0, 0.1, 0.2],
            methods=[TimingMethod.MEASURED, TimingMethod.INTERPOLATED, TimingMethod.NOMINAL],
            confidence=TimelineConfidence.DEGRADED,
        )
        restored = SourceTimeline.from_dict(timeline.to_dict())
        assert [f.method for f in restored.frames] == [
            TimingMethod.MEASURED,
            TimingMethod.INTERPOLATED,
            TimingMethod.NOMINAL,
        ]

    def test_refusal_to_compute_durations_survives_storage(self):
        raw = [(None, None)] * 4
        frames, confidence, _ = _fill_gaps(raw, nominal_fps=25.0)
        restored = SourceTimeline.from_dict(
            SourceTimeline(frames=frames, confidence=confidence).to_dict()
        )
        assert restored.duration_between(0, 3) is None

    def test_future_schema_version_is_refused(self, cfr_video):
        payload = build_timeline(cfr_video, nominal_fps=30.0).to_dict()
        payload["schema_version"] = TIMELINE_SCHEMA_VERSION + 99

        with pytest.raises(TimelineError, match="schema"):
            SourceTimeline.from_dict(payload)

    def test_serialized_form_is_compact(self, cfr_video):
        # 29k frames on a long clip: a list of objects would be megabytes of
        # punctuation, so frames are stored column-wise.
        payload = build_timeline(cfr_video, nominal_fps=30.0).to_dict()
        assert isinstance(payload["source_seconds"], list)
        assert len(payload["source_seconds"]) == payload["frame_count"]


class TestContainerMetadataIsNotTrusted:
    def test_measured_rate_wins_and_the_disagreement_is_recorded(self, cfr_video):
        # Simulates the real failure: container claims 22.873, frames say 30.
        timeline = build_timeline(cfr_video, nominal_fps=22.873)

        assert timeline.measured_fps == pytest.approx(30.0, abs=0.05)
        assert any("container advertises" in n for n in timeline.notes)

    def test_agreeing_metadata_produces_no_warning(self, cfr_video):
        timeline = build_timeline(cfr_video, nominal_fps=30.0)
        assert not any("container advertises" in n for n in timeline.notes)
