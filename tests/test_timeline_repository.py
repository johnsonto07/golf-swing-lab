"""Persisted timelines, staleness, and the corrected VFR classification.

The regression these guard is specific and was shipped: a clip whose container
claimed 484 frames at 22.873 fps, but which decodes 438 frames at a rock-steady
25.000, was labelled variable-frame-rate and had its timestamps computed from
the wrong rate. Bad metadata and variable frame rate are different conditions
with different remedies, and the tests below keep them apart.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from golf_lab.storage import timeline_repository as repo
from golf_lab.storage.file_repository import swing_dir
from golf_lab.video.frame_reader import FrameReader
from golf_lab.video.timeline import (
    EXTRACTOR_VERSION,
    TIMELINE_SCHEMA_VERSION,
    FrameTiming,
    RateClassification,
    SourceTimeline,
    TimelineConfidence,
    TimingMethod,
    build_timeline,
)

SWING_ID = "20260101_120000_timeline"


@pytest.fixture()
def cfr_clip(tmp_path):
    from golf_lab.video.ffmpeg import find_ffmpeg

    tools = find_ffmpeg(required=False)
    if tools is None:
        pytest.skip("FFmpeg not installed")
    path = tmp_path / "cfr.mp4"
    subprocess.run(
        [
            tools.ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture()
def vfr_clip(tmp_path):
    from golf_lab.video.ffmpeg import find_ffmpeg

    tools = find_ffmpeg(required=False)
    if tools is None:
        pytest.skip("FFmpeg not installed")
    path = tmp_path / "vfr.mp4"
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


class TestCorrectedClassification:
    def test_lying_container_does_not_make_a_clip_variable_rate(self, cfr_clip):
        """The exact shape of the user's clip: metadata wrong, video fine.

        Container claims far more frames than decode, and a nominal rate that
        disagrees with its own average — yet the frames are evenly spaced.
        Before this milestone that combination produced a VFR warning.
        """
        timeline = build_timeline(
            cfr_clip,
            nominal_fps=22.873,          # the container's bogus average
            container_frame_count=484,   # the container's bogus count
        )

        assert timeline.rate_classification is RateClassification.CONSTANT
        assert timeline.is_constant_rate
        assert timeline.frame_count == 60, "decoded count must win"
        assert timeline.measured_fps == pytest.approx(30.0, abs=0.05)

    def test_the_metadata_problem_is_still_reported_separately(self, cfr_clip):
        timeline = build_timeline(
            cfr_clip, nominal_fps=22.873, container_frame_count=484
        )
        assert timeline.container_metadata_is_inconsistent
        assert any("not variable frame rate" in n for n in timeline.notes)

    def test_genuine_variable_rate_is_still_detected(self, vfr_clip):
        timeline = build_timeline(vfr_clip, nominal_fps=30.0)
        assert timeline.rate_classification is RateClassification.VARIABLE
        assert not timeline.is_constant_rate

    def test_nominal_only_timing_is_unverified_not_constant(self):
        frames = [
            FrameTiming(preview_index=i, source_seconds=i / 30.0,
                        method=TimingMethod.NOMINAL)
            for i in range(10)
        ]
        timeline = SourceTimeline(frames=frames, confidence=TimelineConfidence.NOMINAL)
        # Without measurement, spacing is unknown — not "constant by default".
        assert timeline.rate_classification is RateClassification.UNVERIFIED

    def test_consistent_metadata_produces_no_inconsistency_flag(self, cfr_clip):
        timeline = build_timeline(cfr_clip, nominal_fps=30.0, container_frame_count=60)
        assert not timeline.container_metadata_is_inconsistent


class TestPreviewPreservesFrames:
    def test_preview_keeps_every_frame_and_timestamp(self, vfr_clip, tmp_path):
        """The invariant issue #1 wrongly assumed was broken.

        Runs the project's own preview generator on genuinely variable-rate
        input and compares timelines frame by frame.
        """
        from golf_lab.video.metadata import extract_metadata
        from golf_lab.video.preview import generate_preview

        meta = extract_metadata(vfr_clip)
        preview = generate_preview(vfr_clip, tmp_path / "preview.mp4", meta)

        source = build_timeline(vfr_clip, nominal_fps=meta.fps)
        generated = build_timeline(preview, nominal_fps=meta.fps)

        assert generated.frame_count == source.frame_count
        for a, b in zip(source.frames, generated.frames):
            assert a.source_seconds == pytest.approx(b.source_seconds, abs=1e-4)


class TestPersistence:
    def test_round_trip_through_disk(self, cfr_clip, swing_root):
        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        timeline = repo.measure_and_save(
            SWING_ID, cfr_clip, nominal_fps=30.0, container_frame_count=60,
            root=swing_root,
        )
        assert timeline is not None
        assert repo.has_timeline(SWING_ID, swing_root)

        loaded = repo.load_timeline(SWING_ID, swing_root)
        assert loaded is not None
        assert loaded.frame_count == timeline.frame_count
        assert loaded.rate_classification is timeline.rate_classification
        assert loaded.confidence is timeline.confidence
        assert loaded.measured_fps == pytest.approx(timeline.measured_fps)

    def test_stored_file_records_provenance(self, cfr_clip, swing_root):
        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        repo.measure_and_save(SWING_ID, cfr_clip, 30.0, 60, root=swing_root)

        data = json.loads(
            repo.timeline_path(SWING_ID, swing_root).read_text(encoding="utf-8")
        )
        assert data["schema_version"] == TIMELINE_SCHEMA_VERSION
        assert data["extractor_version"] == EXTRACTOR_VERSION
        assert data["preview_fingerprint"]
        assert data["rate_classification"] == "constant"
        assert data["measured_duration_seconds"] is not None
        assert len(data["methods"]) == data["frame_count"]

    def test_missing_and_corrupt_files_load_as_none(self, swing_root):
        assert repo.load_timeline("absent", swing_root) is None

        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        repo.timeline_path(SWING_ID, swing_root).write_text("{oops", encoding="utf-8")
        assert repo.load_timeline(SWING_ID, swing_root) is None

    def test_delete_removes_it(self, cfr_clip, swing_root):
        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        repo.measure_and_save(SWING_ID, cfr_clip, 30.0, root=swing_root)
        repo.delete_timeline(SWING_ID, swing_root)
        assert not repo.has_timeline(SWING_ID, swing_root)


class TestStaleness:
    def test_fresh_timeline_is_current(self, cfr_clip, swing_root):
        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        timeline = repo.measure_and_save(SWING_ID, cfr_clip, 30.0, root=swing_root)
        assert repo.staleness_reasons(timeline, preview_path=cfr_clip) == []

    def test_replaced_preview_makes_it_stale(self, cfr_clip, swing_root, tmp_path):
        import os
        import time

        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        timeline = repo.measure_and_save(SWING_ID, cfr_clip, 30.0, root=swing_root)

        time.sleep(1.1)  # fingerprint uses whole-second mtime
        cfr_clip.write_bytes(b"completely different content")
        os.utime(cfr_clip, None)

        reasons = repo.staleness_reasons(timeline, preview_path=cfr_clip)
        assert any("preview has changed" in r for r in reasons)

    def test_missing_timeline_is_stale(self):
        reasons = repo.staleness_reasons(None)
        assert reasons and "No measured timeline" in reasons[0]

    def test_extractor_version_change_invalidates(self, cfr_clip, swing_root):
        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        timeline = repo.measure_and_save(SWING_ID, cfr_clip, 30.0, root=swing_root)
        timeline.extractor_version = "0"
        assert any(
            "extractor version" in r for r in repo.staleness_reasons(timeline)
        )


class TestFrameReaderUsesMeasuredTiming:
    def test_timestamps_come_from_the_timeline(self, vfr_clip):
        timeline = build_timeline(vfr_clip, nominal_fps=30.0)
        with FrameReader(vfr_clip, timeline=timeline) as reader:
            assert reader.timing_is_measured
            for index in (0, 5, 20, 40):
                assert reader.timestamp_for_frame(index) == pytest.approx(
                    timeline.source_seconds(index), abs=1e-6
                )

    def test_measured_timestamps_differ_from_the_nominal_assumption(self, vfr_clip):
        # On variable-rate media index/fps is simply wrong; this pins that the
        # reader no longer returns it.
        timeline = build_timeline(vfr_clip, nominal_fps=30.0)
        with FrameReader(vfr_clip, timeline=timeline) as reader:
            late = len(timeline) - 1
            assert reader.timestamp_for_frame(late) != pytest.approx(
                late / 30.0, rel=0.02
            )

    def test_frame_count_uses_decoded_frames(self, cfr_clip):
        timeline = build_timeline(cfr_clip, nominal_fps=30.0)
        with FrameReader(cfr_clip, timeline=timeline) as reader:
            assert reader.frame_count_is_measured
            assert reader.last_index == timeline.frame_count - 1

    def test_without_a_timeline_it_falls_back_and_says_so(self, cfr_clip):
        with FrameReader(cfr_clip) as reader:
            assert not reader.timing_is_measured
            assert not reader.frame_count_is_measured
            assert reader.timestamp_for_frame(30) == pytest.approx(30 / reader.fps)

    def test_timestamp_to_frame_round_trips(self, vfr_clip):
        timeline = build_timeline(vfr_clip, nominal_fps=30.0)
        with FrameReader(vfr_clip, timeline=timeline) as reader:
            for index in (0, 11, 33):
                seconds = reader.timestamp_for_frame(index)
                assert reader.frame_for_timestamp(seconds) == index


class TestImportMeasuresTiming:
    def test_import_stores_a_timeline(self, fixture_video, swing_root):
        from golf_lab.models.video import SwingContext
        from golf_lab.storage import swing_repository

        record = swing_repository.import_swing(
            source_path=fixture_video,
            original_filename="measured.mp4",
            context=SwingContext(),
            root=swing_root,
        )
        timeline = repo.load_timeline(record.swing_id, swing_root)

        assert timeline is not None
        assert timeline.frame_count > 0
        assert timeline.confidence.supports_durations
        assert timeline.rate_classification is RateClassification.CONSTANT

    def test_frame_counts_agree_across_storage_and_timeline(
        self, fixture_video, swing_root
    ):
        from golf_lab.models.video import SwingContext
        from golf_lab.storage import swing_repository

        record = swing_repository.import_swing(
            source_path=fixture_video,
            original_filename="counts.mp4",
            context=SwingContext(),
            root=swing_root,
        )
        timeline = repo.load_timeline(record.swing_id, swing_root)
        video_path = swing_repository.preview_or_original_path(record, swing_root)

        with FrameReader(video_path, timeline=timeline) as reader:
            assert reader.last_index + 1 == timeline.frame_count

    def test_a_clean_clip_is_not_marked_needs_review(self, fixture_video, swing_root):
        from golf_lab.models.video import SwingContext, SwingStatus
        from golf_lab.storage import swing_repository

        record = swing_repository.import_swing(
            source_path=fixture_video,
            original_filename="clean.mp4",
            context=SwingContext(),
            root=swing_root,
        )
        assert record.status is SwingStatus.READY
        assert "variable-frame-rate" not in record.status_detail.lower()
