"""Variable-frame-rate detection and the preview/original timeline split.

Written against a real failure: a 4K HEVC phone clip whose 484 source frames
at 22.87 fps average became a 438-frame preview at a constant 25 fps. The app
had no way to express that those are two different timelines, so it reported
the original's frame count next to a slider bounded by the preview's.

See docs/KNOWN_ISSUES.md (GSL-1).
"""

from __future__ import annotations

import pytest

from golf_lab.config import ANALYSIS_VERSION, APP_VERSION
from golf_lab.models.video import SwingRecord, VideoMetadata


def _metadata(**overrides) -> VideoMetadata:
    base = dict(
        path="original.mov",
        coded_width=3840,
        coded_height=2160,
        width=2160,
        height=3840,
        rotation_degrees=90,
        fps=22.873,
        frame_count=484,
        duration_seconds=17.55,
        avg_frame_rate=22.873,
        r_frame_rate=25.0,
    )
    base.update(overrides)
    return VideoMetadata(**base)


def _record(video: VideoMetadata, preview: VideoMetadata | None) -> SwingRecord:
    return SwingRecord(
        swing_id="20260806_095517_371a7361",
        original_filename="golfswingslow.mov",
        original_relpath="original.mov",
        preview_relpath="preview.mp4",
        video=video,
        preview_video=preview,
        app_version=APP_VERSION,
        analysis_version=ANALYSIS_VERSION,
    )


class TestVariableFrameRateDetection:
    def test_detects_the_real_failing_clip(self):
        # The actual numbers from the clip that exposed this.
        assert _metadata().is_variable_frame_rate

    def test_constant_frame_rate_is_not_flagged(self):
        meta = _metadata(avg_frame_rate=30.0, r_frame_rate=30.0, fps=30.0)
        assert not meta.is_variable_frame_rate

    def test_ntsc_rates_are_not_flagged(self):
        # 60 vs 59.94 differ by 0.1% and are perfectly constant. A tighter
        # tolerance here would flag most ordinary footage as VFR.
        meta = _metadata(avg_frame_rate=59.94, r_frame_rate=60.0, fps=59.94)
        assert not meta.is_variable_frame_rate

    def test_missing_rates_are_not_flagged(self):
        # OpenCV fallback cannot supply these; absence must not be read as VFR.
        meta = _metadata(avg_frame_rate=None, r_frame_rate=None)
        assert not meta.is_variable_frame_rate

    def test_zero_rates_do_not_divide_by_zero(self):
        meta = _metadata(avg_frame_rate=0.0, r_frame_rate=0.0)
        assert not meta.is_variable_frame_rate


class TestTimelineIsApproximate:
    def test_vfr_source_makes_the_timeline_approximate(self):
        record = _record(_metadata(), _metadata(fps=25.0, frame_count=438))
        assert record.timeline_is_approximate

    def test_frame_count_mismatch_alone_is_enough(self):
        # Even if the rates look constant, losing frames in the proxy means
        # preview index no longer equals source index.
        cfr = _metadata(avg_frame_rate=30.0, r_frame_rate=30.0, fps=30.0)
        preview = _metadata(avg_frame_rate=30.0, r_frame_rate=30.0, fps=30.0, frame_count=400)
        assert _record(cfr, preview).timeline_is_approximate

    def test_clean_cfr_clip_is_exact(self):
        cfr = _metadata(avg_frame_rate=30.0, r_frame_rate=30.0, fps=30.0, frame_count=300)
        preview = _metadata(avg_frame_rate=30.0, r_frame_rate=30.0, fps=30.0, frame_count=300)
        assert not _record(cfr, preview).timeline_is_approximate

    def test_off_by_one_is_tolerated(self):
        cfr = _metadata(avg_frame_rate=30.0, r_frame_rate=30.0, fps=30.0, frame_count=300)
        preview = _metadata(avg_frame_rate=30.0, r_frame_rate=30.0, fps=30.0, frame_count=301)
        assert not _record(cfr, preview).timeline_is_approximate


class TestPreviewAccessors:
    def test_preview_values_are_reported_not_original_ones(self):
        # This is the mismatch users saw: "Frames: 484" beside a slider that
        # stopped at 437.
        record = _record(_metadata(), _metadata(fps=25.0, frame_count=438))
        assert record.preview_frame_count == 438
        assert record.preview_fps == 25.0
        assert record.video.frame_count == 484

    def test_falls_back_to_the_original_when_preview_is_missing(self):
        # Records written before preview_video existed must still work.
        record = _record(_metadata(), None)
        assert record.preview_frame_count == 484
        assert record.preview_fps == pytest.approx(22.873)
        assert not record.timeline_is_approximate or record.video.is_variable_frame_rate


class TestBackwardCompatibility:
    def test_old_record_without_preview_video_still_loads(self):
        payload = _record(_metadata(), None).model_dump(mode="json")
        payload.pop("preview_video", None)
        payload["video"].pop("avg_frame_rate", None)
        payload["video"].pop("r_frame_rate", None)

        restored = SwingRecord.model_validate(payload)
        assert restored.preview_video is None
        assert not restored.video.is_variable_frame_rate
        assert restored.preview_frame_count == 484

    def test_round_trip_preserves_preview_metadata(self):
        record = _record(_metadata(), _metadata(fps=25.0, frame_count=438))
        restored = SwingRecord.model_validate(record.model_dump(mode="json"))
        assert restored.preview_video is not None
        assert restored.preview_video.frame_count == 438
        assert restored.timeline_is_approximate


class TestImportRecordsBothTimelines:
    def test_import_stores_preview_metadata(self, fixture_video, swing_root):
        from golf_lab.models.video import SwingContext
        from golf_lab.storage import swing_repository

        record = swing_repository.import_swing(
            source_path=fixture_video,
            original_filename="cfr_clip.mp4",
            context=SwingContext(),
            root=swing_root,
        )

        assert record.preview_video is not None
        assert record.preview_video.frame_count > 0
        # The synthetic fixture is constant-frame-rate, so the two timelines
        # must agree and nothing should be flagged.
        assert not record.timeline_is_approximate
        assert record.preview_frame_count == record.video.frame_count

    def test_probe_populates_both_frame_rates(self, fixture_video):
        from golf_lab.video.metadata import extract_metadata

        meta = extract_metadata(fixture_video)
        assert meta.avg_frame_rate and meta.avg_frame_rate > 0
        assert meta.r_frame_rate and meta.r_frame_rate > 0
        assert not meta.is_variable_frame_rate
