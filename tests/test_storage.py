"""Storage paths, serialization, and the end-to-end import pipeline."""

from __future__ import annotations

import json

import pytest

from golf_lab.models.video import (
    CameraView,
    Handedness,
    ShotShape,
    SwingContext,
    SwingRecord,
    SwingStatus,
    VideoMetadata,
)
from golf_lab.storage import swing_repository
from golf_lab.storage.file_repository import (
    exports_dir,
    list_swing_ids,
    metadata_path,
    new_swing_id,
    sanitize_filename,
    swing_dir,
)
from golf_lab.storage.swing_repository import SwingImportError
from golf_lab.video.frame_reader import FrameReader


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("My Swing.mp4", "My_Swing.mp4"),
            ("My Swing (driver) 2.mp4", "My_Swing__driver__2.mp4"),
            ("a/b\\c:d*e.mov", "a_b_c_d_e.mov"),
            ("CON.mp4", "CON_file.mp4"),
            ("   spaced   .mp4", "spaced.mp4"),
        ],
    )
    def test_produces_windows_safe_names(self, raw, expected):
        assert sanitize_filename(raw) == expected

    def test_empty_name_uses_fallback(self):
        assert sanitize_filename("   ").startswith("video")

    def test_result_has_no_forbidden_characters(self):
        result = sanitize_filename('we:ir*d?"na<me>.mp4')
        assert not any(char in result for char in '<>:"/\\|?*')

    def test_truncates_very_long_names(self):
        result = sanitize_filename("x" * 300 + ".mp4")
        assert len(result) <= 90


class TestSwingPaths:
    def test_ids_are_unique_and_sortable(self):
        ids = [new_swing_id() for _ in range(5)]
        assert len(set(ids)) == 5
        assert all(len(sid) > 10 for sid in ids)

    def test_path_helpers(self, swing_root):
        assert swing_dir("abc", swing_root) == swing_root / "abc"
        assert metadata_path("abc", swing_root) == swing_root / "abc" / "metadata.json"
        assert exports_dir("abc", swing_root) == swing_root / "abc" / "exports"

    def test_list_ignores_dirs_without_metadata(self, swing_root):
        (swing_root / "incomplete").mkdir()
        assert list_swing_ids(swing_root) == []

    def test_list_is_newest_first(self, swing_root):
        for name in ("20260101_000000_aaaa", "20260201_000000_bbbb"):
            directory = swing_root / name
            directory.mkdir()
            (directory / "metadata.json").write_text("{}", encoding="utf-8")
        assert list_swing_ids(swing_root)[0] == "20260201_000000_bbbb"


def _dummy_record(swing_id: str = "20260101_120000_abcd1234") -> SwingRecord:
    return SwingRecord(
        swing_id=swing_id,
        original_filename="My Swing (driver).mp4",
        original_relpath="original.mp4",
        preview_relpath="preview.mp4",
        thumbnail_relpath="thumbnail.jpg",
        video=VideoMetadata(
            path="original.mp4",
            coded_width=1080, coded_height=1920,
            width=1080, height=1920,
            fps=119.88, frame_count=240, duration_seconds=2.002,
        ),
        context=SwingContext(
            club="Driver",
            camera_view=CameraView.DOWN_THE_LINE,
            handedness=Handedness.RIGHT,
            shot_shape=ShotShape.FADE,
            typical_miss="block right",
            carry_yards=245.0,
            notes="windy",
        ),
        app_version="0.1.0",
        analysis_version="1",
    )


class TestSerialization:
    def test_round_trip_preserves_every_field(self, swing_root):
        record = _dummy_record()
        swing_dir(record.swing_id, swing_root).mkdir(parents=True)
        swing_repository.save_record(record, swing_root)

        loaded = swing_repository.load_record(record.swing_id, swing_root)
        assert loaded.swing_id == record.swing_id
        assert loaded.original_filename == record.original_filename
        assert loaded.context.club == "Driver"
        assert loaded.context.camera_view is CameraView.DOWN_THE_LINE
        assert loaded.context.shot_shape is ShotShape.FADE
        assert loaded.context.carry_yards == 245.0
        assert loaded.video.fps == pytest.approx(119.88)
        assert loaded.video.frame_count == 240
        assert loaded.imported_at == record.imported_at

    def test_metadata_json_is_human_readable(self, swing_root):
        record = _dummy_record()
        swing_dir(record.swing_id, swing_root).mkdir(parents=True)
        path = swing_repository.save_record(record, swing_root)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["context"]["camera_view"] == "down_the_line"
        assert payload["status"] == "not_processed"
        assert "\n" in path.read_text(encoding="utf-8")  # indented, not one line

    def test_no_temp_file_left_behind(self, swing_root):
        record = _dummy_record()
        swing_dir(record.swing_id, swing_root).mkdir(parents=True)
        swing_repository.save_record(record, swing_root)
        leftovers = list(swing_dir(record.swing_id, swing_root).glob("*.tmp"))
        assert leftovers == []

    def test_load_missing_swing_raises(self, swing_root):
        with pytest.raises(SwingImportError):
            swing_repository.load_record("nope", swing_root)

    def test_list_records_skips_corrupt_metadata(self, swing_root):
        good = _dummy_record()
        swing_dir(good.swing_id, swing_root).mkdir(parents=True)
        swing_repository.save_record(good, swing_root)

        bad_dir = swing_root / "20250101_000000_bad00000"
        bad_dir.mkdir()
        (bad_dir / "metadata.json").write_text("{not json", encoding="utf-8")

        records = swing_repository.list_records(swing_root)
        assert [r.swing_id for r in records] == [good.swing_id]


class TestValidateUpload:
    def test_rejects_unsupported_extension(self):
        with pytest.raises(SwingImportError, match="unsupported extension"):
            swing_repository.validate_upload("swing.avi", 1000)

    def test_rejects_empty_file(self):
        with pytest.raises(SwingImportError, match="empty"):
            swing_repository.validate_upload("swing.mp4", 0)

    def test_accepts_mp4_and_mov(self):
        swing_repository.validate_upload("swing.mp4", 1000)
        swing_repository.validate_upload("swing.MOV", 1000)


class TestImportPipeline:
    """Integration tests for Pipeline A."""

    def test_imports_and_creates_expected_layout(self, fixture_video, swing_root):
        record = swing_repository.import_swing(
            source_path=fixture_video,
            original_filename="My Swing (7 iron).mp4",
            context=SwingContext(club="7 Iron", camera_view=CameraView.FACE_ON),
            root=swing_root,
        )

        directory = swing_dir(record.swing_id, swing_root)
        assert (directory / "original.mp4").exists()
        assert (directory / "preview.mp4").exists()
        assert (directory / "thumbnail.jpg").exists()
        assert (directory / "metadata.json").exists()
        assert (directory / "exports").is_dir()
        assert record.status is SwingStatus.READY

    def test_original_filename_preserved_in_metadata(self, fixture_video, swing_root):
        record = swing_repository.import_swing(
            fixture_video, "My Swing (7 iron).mp4", SwingContext(), root=swing_root
        )
        loaded = swing_repository.load_record(record.swing_id, swing_root)
        assert loaded.original_filename == "My Swing (7 iron).mp4"

    def test_source_file_is_not_modified_or_moved(self, fixture_video, swing_root):
        before = fixture_video.stat()
        swing_repository.import_swing(
            fixture_video, "swing.mp4", SwingContext(), root=swing_root
        )
        after = fixture_video.stat()
        assert fixture_video.exists()
        assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)

    def test_stored_original_is_byte_identical(self, fixture_video, swing_root):
        record = swing_repository.import_swing(
            fixture_video, "swing.mp4", SwingContext(), root=swing_root
        )
        stored = swing_dir(record.swing_id, swing_root) / record.original_relpath
        assert stored.read_bytes() == fixture_video.read_bytes()

    def test_stored_original_is_read_only(self, fixture_video, swing_root):
        import os
        import stat as stat_module

        record = swing_repository.import_swing(
            fixture_video, "swing.mp4", SwingContext(), root=swing_root
        )
        stored = swing_dir(record.swing_id, swing_root) / record.original_relpath
        mode = stored.stat().st_mode
        assert not mode & stat_module.S_IWRITE
        assert os.access(stored, os.R_OK)

    def test_preview_is_frame_accessible(self, fixture_video, swing_root):
        record = swing_repository.import_swing(
            fixture_video, "swing.mp4", SwingContext(), root=swing_root
        )
        preview = swing_repository.preview_or_original_path(record, swing_root)
        with FrameReader(preview) as reader:
            assert reader.frame_count == pytest.approx(60, abs=1)
            frame = reader.read_frame(30)
        assert frame.shape[2] == 3

    def test_reopening_restores_metadata(self, fixture_video, swing_root):
        context = SwingContext(
            club="Driver",
            camera_view=CameraView.DOWN_THE_LINE,
            handedness=Handedness.LEFT,
            shot_shape=ShotShape.DRAW,
            typical_miss="hook",
            carry_yards=260.0,
            notes="second ball",
        )
        record = swing_repository.import_swing(
            fixture_video, "Session 3 (driver).mp4", context, root=swing_root
        )

        reopened = swing_repository.load_record(record.swing_id, swing_root)
        assert reopened.context.club == "Driver"
        assert reopened.context.handedness is Handedness.LEFT
        assert reopened.context.shot_shape is ShotShape.DRAW
        assert reopened.context.carry_yards == 260.0
        assert reopened.context.notes == "second ball"
        assert reopened.video.frame_count == record.video.frame_count

    def test_awkward_filename_import(self, fixture_video_awkward_name, swing_root):
        record = swing_repository.import_swing(
            fixture_video_awkward_name,
            fixture_video_awkward_name.name,
            SwingContext(),
            root=swing_root,
        )
        assert record.original_filename == "My Swing (driver) 2.mp4"
        assert (swing_dir(record.swing_id, swing_root) / "original.mp4").exists()

    def test_rotated_video_preview_is_upright(self, fixture_video_rotated, swing_root):
        record = swing_repository.import_swing(
            fixture_video_rotated, "portrait.mp4", SwingContext(), root=swing_root
        )
        assert record.video.width == 240 and record.video.height == 320

        preview = swing_repository.preview_or_original_path(record, swing_root)
        with FrameReader(preview) as reader:
            assert (reader.width, reader.height) == (240, 320)

    def test_failed_import_leaves_no_directory(self, tmp_path, swing_root):
        bogus = tmp_path / "broken.mp4"
        bogus.write_text("not a video", encoding="utf-8")

        with pytest.raises(SwingImportError):
            swing_repository.import_swing(
                bogus, "broken.mp4", SwingContext(), root=swing_root
            )
        assert list_swing_ids(swing_root) == []
        assert list(swing_root.iterdir()) == []

    def test_progress_callback_reaches_completion(self, fixture_video, swing_root):
        seen: list[float] = []
        swing_repository.import_swing(
            fixture_video, "swing.mp4", SwingContext(),
            root=swing_root, progress=lambda f, _m: seen.append(f),
        )
        assert seen and seen[-1] == 1.0
        assert seen == sorted(seen)

    def test_multiple_imports_are_isolated(self, fixture_video, swing_root):
        first = swing_repository.import_swing(
            fixture_video, "a.mp4", SwingContext(club="Driver"), root=swing_root
        )
        second = swing_repository.import_swing(
            fixture_video, "b.mp4", SwingContext(club="7 Iron"), root=swing_root
        )
        assert first.swing_id != second.swing_id
        assert len(swing_repository.list_records(swing_root)) == 2
