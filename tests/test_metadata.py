"""Metadata extraction, frame-rate parsing, and orientation handling."""

from __future__ import annotations

import pytest

from golf_lab.models.video import VideoMetadata
from golf_lab.video.metadata import (
    VideoMetadataError,
    extract_metadata,
    extract_rotation_degrees,
    parse_frame_rate,
)


class TestParseFrameRate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("30/1", 30.0),
            ("60000/1001", pytest.approx(59.94, abs=0.01)),
            ("120000/1001", pytest.approx(119.88, abs=0.01)),
            ("240", 240.0),
            ("0/0", 0.0),
            ("", 0.0),
            (None, 0.0),
            ("garbage", 0.0),
        ],
    )
    def test_parses_rational_and_plain_rates(self, raw, expected):
        assert parse_frame_rate(raw) == expected


class TestRotationExtraction:
    def test_no_rotation_metadata(self):
        assert extract_rotation_degrees({}) == 0

    def test_rotate_tag(self):
        assert extract_rotation_degrees({"tags": {"rotate": "90"}}) == 90
        assert extract_rotation_degrees({"tags": {"rotate": "180"}}) == 180

    def test_display_matrix_is_counter_clockwise(self):
        # A display matrix rotation of -90 means "rotate 90 clockwise to display".
        stream = {"side_data_list": [{"rotation": -90}]}
        assert extract_rotation_degrees(stream) == 90

    def test_display_matrix_positive_ninety(self):
        stream = {"side_data_list": [{"rotation": 90}]}
        assert extract_rotation_degrees(stream) == 270

    def test_rotate_tag_takes_priority(self):
        stream = {"tags": {"rotate": "180"}, "side_data_list": [{"rotation": -90}]}
        assert extract_rotation_degrees(stream) == 180

    def test_malformed_values_do_not_raise(self):
        assert extract_rotation_degrees({"tags": {"rotate": "sideways"}}) == 0


class TestExtractMetadata:
    def test_reads_core_properties(self, fixture_video):
        meta = extract_metadata(fixture_video)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.fps == pytest.approx(30.0, abs=0.01)
        assert meta.frame_count == 60
        assert meta.duration_seconds == pytest.approx(2.0, abs=0.05)
        assert meta.codec_name == "h264"
        assert meta.has_audio is True
        assert meta.probe_source == "ffprobe"

    def test_rotation_swaps_display_dimensions(self, fixture_video_rotated):
        meta = extract_metadata(fixture_video_rotated)
        assert meta.rotation_degrees in (90, 270)
        assert (meta.coded_width, meta.coded_height) == (320, 240)
        # Display dimensions must be swapped for a quarter-turn rotation.
        assert (meta.width, meta.height) == (240, 320)
        assert meta.is_portrait is True

    def test_handles_filenames_with_spaces_and_parentheses(
        self, fixture_video_awkward_name
    ):
        assert " " in fixture_video_awkward_name.name
        assert "(" in fixture_video_awkward_name.name
        meta = extract_metadata(fixture_video_awkward_name)
        assert meta.frame_count == 60

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(VideoMetadataError):
            extract_metadata(tmp_path / "does_not_exist.mp4")

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(VideoMetadataError):
            extract_metadata(empty)

    def test_non_video_file_raises(self, tmp_path):
        text_file = tmp_path / "not_a_video.mp4"
        text_file.write_text("this is definitely not an mp4", encoding="utf-8")
        with pytest.raises(VideoMetadataError):
            extract_metadata(text_file)


class TestTimestampConversions:
    @pytest.fixture()
    def meta(self) -> VideoMetadata:
        return VideoMetadata(
            path="x.mp4",
            coded_width=1920, coded_height=1080,
            width=1920, height=1080,
            fps=120.0, frame_count=600, duration_seconds=5.0,
        )

    def test_frame_to_timestamp(self, meta):
        assert meta.timestamp_for_frame(0) == 0.0
        assert meta.timestamp_for_frame(120) == pytest.approx(1.0)
        assert meta.timestamp_for_frame(599) == pytest.approx(4.99167, abs=1e-4)

    def test_timestamp_to_frame_round_trips(self, meta):
        for index in (0, 1, 59, 240, 599):
            seconds = meta.timestamp_for_frame(index)
            assert meta.frame_for_timestamp(seconds) == index

    def test_timestamp_to_frame_clamps(self, meta):
        assert meta.frame_for_timestamp(-5.0) == 0
        assert meta.frame_for_timestamp(9999.0) == 599

    def test_zero_fps_is_safe(self):
        meta = VideoMetadata(
            path="x.mp4", coded_width=10, coded_height=10, width=10, height=10,
            fps=0.0, frame_count=0, duration_seconds=0.0,
        )
        assert meta.timestamp_for_frame(5) == 0.0
        assert meta.frame_for_timestamp(5.0) == 0
