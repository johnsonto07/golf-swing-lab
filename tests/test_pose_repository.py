"""Storing pose results per swing, and flagging them when they go stale."""

from __future__ import annotations

import json
import os
import time

import pytest

from golf_lab.config import ANALYSIS_VERSION
from golf_lab.pose.smoothing import SmoothingSettings, smooth_sequence
from golf_lab.storage import pose_repository
from golf_lab.storage.file_repository import swing_dir


@pytest.fixture()
def saved_analysis(tmp_path, swing_root, pose_sequence_factory):
    """A stored analysis plus the video it was computed from."""
    swing_id = "20260101_120000_abcd1234"
    swing_dir(swing_id, swing_root).mkdir(parents=True, exist_ok=True)

    video = tmp_path / "preview.mp4"
    video.write_bytes(b"pretend video content")

    raw = pose_sequence_factory(frame_count=40, failed=(10, 11))
    smoothed = smooth_sequence(raw, SmoothingSettings(window_length=7))

    info = pose_repository.build_info(
        swing_id=swing_id,
        raw=raw,
        smoothed=smoothed,
        video_path=video,
        model_key="full",
        model_filename="pose_landmarker_full.task",
        model_sha256="deadbeef",
        backend="fake/test-backend",
        device="cpu",
        mediapipe_version="0.10.14",
    )
    pose_repository.save_pose_analysis(swing_id, raw, smoothed, info, swing_root)
    return swing_id, video, raw, smoothed, info


class TestSaveAndLoad:
    def test_round_trip(self, saved_analysis, swing_root):
        swing_id, _, raw, smoothed, _ = saved_analysis

        assert pose_repository.has_pose_analysis(swing_id, swing_root)

        loaded_raw = pose_repository.load_raw(swing_id, swing_root)
        loaded_smoothed = pose_repository.load_smoothed(swing_id, swing_root)

        assert loaded_raw.frame_count == raw.frame_count
        assert loaded_raw.detected_count == raw.detected_count
        assert loaded_smoothed.smoothing == smoothed.smoothing

    def test_raw_and_smoothed_are_separate_files(self, saved_analysis, swing_root):
        # Raw is the evidence; smoothed is a convenience. Keeping both means a
        # measurement can always be traced back to what was observed.
        swing_id, _, _, _, _ = saved_analysis
        assert pose_repository.raw_path(swing_id, swing_root).exists()
        assert pose_repository.smoothed_path(swing_id, swing_root).exists()
        assert (
            pose_repository.raw_path(swing_id, swing_root)
            != pose_repository.smoothed_path(swing_id, swing_root)
        )

    def test_raw_and_smoothed_agree_on_what_was_detected(self, saved_analysis, swing_root):
        swing_id, _, _, _, _ = saved_analysis
        raw = pose_repository.load_raw(swing_id, swing_root)
        smoothed = pose_repository.load_smoothed(swing_id, swing_root)
        assert list(raw.failed_frames) == list(smoothed.failed_frames)

    def test_info_records_provenance(self, saved_analysis, swing_root):
        swing_id, _, _, _, _ = saved_analysis
        info = pose_repository.load_info(swing_id, swing_root)

        assert info.model_key == "full"
        assert info.mediapipe_version == "0.10.14"
        assert info.device == "cpu"
        assert info.frame_count == 40
        assert info.detected_count == 38
        assert info.detection_rate == pytest.approx(38 / 40)
        assert info.longest_gap_frames == 2
        assert info.analysis_version == ANALYSIS_VERSION
        assert info.created_at

    def test_missing_analysis_loads_as_none(self, swing_root):
        assert pose_repository.load_info("nope", swing_root) is None
        assert pose_repository.load_raw("nope", swing_root) is None
        assert not pose_repository.has_pose_analysis("nope", swing_root)

    def test_delete_removes_everything(self, saved_analysis, swing_root):
        swing_id, _, _, _, _ = saved_analysis
        pose_repository.delete_pose_analysis(swing_id, swing_root)

        assert not pose_repository.has_pose_analysis(swing_id, swing_root)
        assert pose_repository.load_raw(swing_id, swing_root) is None

    def test_delete_does_not_touch_the_video(self, saved_analysis, swing_root):
        swing_id, video, _, _, _ = saved_analysis
        pose_repository.delete_pose_analysis(swing_id, swing_root)
        assert video.exists()

    def test_corrupt_npz_loads_as_none_rather_than_exploding(
        self, saved_analysis, swing_root
    ):
        swing_id, _, _, _, _ = saved_analysis
        pose_repository.raw_path(swing_id, swing_root).write_bytes(b"garbage")
        assert pose_repository.load_raw(swing_id, swing_root) is None

    def test_corrupt_info_loads_as_none(self, saved_analysis, swing_root):
        swing_id, _, _, _, _ = saved_analysis
        pose_repository.info_path(swing_id, swing_root).write_text("{oops", encoding="utf-8")
        assert pose_repository.load_info(swing_id, swing_root) is None


class TestStaleness:
    def test_fresh_analysis_is_not_stale(self, saved_analysis):
        _, video, _, _, info = saved_analysis
        assert pose_repository.staleness_reasons(info, video) == []
        assert not pose_repository.is_stale(info, video)

    def test_missing_analysis_is_stale(self, saved_analysis):
        _, video, _, _, _ = saved_analysis
        reasons = pose_repository.staleness_reasons(None, video)
        assert reasons and "No pose analysis" in reasons[0]

    def test_changed_video_makes_it_stale(self, saved_analysis):
        # The exact failure this guards: showing a skeleton computed from a
        # video the user has since replaced.
        _, video, _, _, info = saved_analysis
        time.sleep(1.1)  # fingerprint uses whole-second mtime
        video.write_bytes(b"completely different video content")
        os.utime(video, None)

        reasons = pose_repository.staleness_reasons(info, video)
        assert any("video file has changed" in reason for reason in reasons)

    def test_older_analysis_version_makes_it_stale(self, saved_analysis):
        _, video, _, _, info = saved_analysis
        info.analysis_version = "0"

        reasons = pose_repository.staleness_reasons(info, video)
        assert any("analysis version" in reason for reason in reasons)

    def test_unreadable_video_is_reported(self, saved_analysis, tmp_path):
        _, _, _, _, info = saved_analysis
        reasons = pose_repository.staleness_reasons(info, tmp_path / "gone.mp4")
        assert reasons and "could not be read" in reasons[0]

    def test_reasons_are_user_facing_sentences(self, saved_analysis):
        _, video, _, _, info = saved_analysis
        info.analysis_version = "0"

        for reason in pose_repository.staleness_reasons(info, video):
            assert reason[0].isupper()
            assert reason.endswith(".")


class TestWriteOrdering:
    def test_info_is_written_last(self, tmp_path, swing_root, pose_sequence_factory):
        # pose_info.json is the marker that a complete analysis exists, so it
        # must not appear before the arrays it describes.
        swing_id = "20260101_130000_ffff0000"
        swing_dir(swing_id, swing_root).mkdir(parents=True, exist_ok=True)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"video")

        raw = pose_sequence_factory(frame_count=10)
        info = pose_repository.build_info(
            swing_id=swing_id,
            raw=raw,
            smoothed=None,
            video_path=video,
            model_key="lite",
            model_filename="pose_landmarker_lite.task",
            model_sha256="abc",
            backend="fake",
            device="cpu",
            mediapipe_version="0.10.14",
        )
        pose_repository.save_pose_analysis(swing_id, raw, None, info, swing_root)

        raw_mtime = pose_repository.raw_path(swing_id, swing_root).stat().st_mtime_ns
        info_mtime = pose_repository.info_path(swing_id, swing_root).stat().st_mtime_ns
        assert info_mtime >= raw_mtime

    def test_no_temp_files_survive(self, saved_analysis, swing_root):
        swing_id, _, _, _, _ = saved_analysis
        directory = swing_dir(swing_id, swing_root)
        assert not list(directory.glob("*.tmp"))

    def test_info_json_is_readable_and_complete(self, saved_analysis, swing_root):
        swing_id, _, _, _, _ = saved_analysis
        data = json.loads(
            pose_repository.info_path(swing_id, swing_root).read_text(encoding="utf-8")
        )
        for key in (
            "model_key",
            "model_sha256",
            "video_fingerprint",
            "detection_rate",
            "smoothing",
            "analysis_version",
            "created_at",
        ):
            assert key in data
