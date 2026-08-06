"""Frame-accuracy tests.

These back the acceptance criteria: the displayed frame is the decoded frame,
and stepping moves exactly one frame at a time.
"""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.video.frame_reader import (
    FrameReader,
    FrameReadError,
    format_timestamp,
    frame_for_timestamp,
    timestamp_for_frame,
)


class TestTimestampHelpers:
    def test_timestamp_for_frame(self):
        assert timestamp_for_frame(0, 30.0) == 0.0
        assert timestamp_for_frame(30, 30.0) == pytest.approx(1.0)
        assert timestamp_for_frame(10, 0.0) == 0.0

    def test_frame_for_timestamp_round_trip(self):
        for index in range(0, 60, 7):
            seconds = timestamp_for_frame(index, 120.0)
            assert frame_for_timestamp(seconds, 120.0, 600) == index

    def test_frame_for_timestamp_clamps(self):
        assert frame_for_timestamp(-1.0, 30.0, 60) == 0
        assert frame_for_timestamp(1000.0, 30.0, 60) == 59

    def test_format_timestamp(self):
        assert format_timestamp(0.0) == "00:00.000"
        assert format_timestamp(65.25) == "01:05.250"
        assert format_timestamp(-3.0) == "00:00.000"


class TestFrameReader:
    def test_reports_video_properties(self, fixture_video):
        with FrameReader(fixture_video) as reader:
            assert reader.frame_count == 60
            assert reader.last_index == 59
            assert reader.fps == pytest.approx(30.0, abs=0.01)
            assert (reader.width, reader.height) == (320, 240)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FrameReadError):
            FrameReader(tmp_path / "nope.mp4")

    def test_reads_a_frame_of_expected_shape(self, fixture_video):
        with FrameReader(fixture_video) as reader:
            frame = reader.read_frame(0)
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8

    def test_sequential_stepping_never_repeats_or_skips(self, fixture_video):
        """Consecutive frames of the synthetic clip must differ from each other.

        `testsrc` animates every frame, so identical neighbours would mean the
        reader returned the same frame twice (a skip or a stall).
        """
        with FrameReader(fixture_video) as reader:
            previous = None
            for index in range(0, 20):
                frame = reader.read_frame(index)
                if previous is not None:
                    assert not np.array_equal(frame, previous), (
                        f"Frame {index} is identical to frame {index - 1}; "
                        "the reader skipped or repeated a frame."
                    )
                previous = frame.copy()

    def test_random_access_matches_sequential_access(self, fixture_video):
        """Seeking to frame N must yield the same pixels as walking to frame N."""
        with FrameReader(fixture_video) as sequential:
            walked = {index: sequential.read_frame(index).copy() for index in range(25)}

        with FrameReader(fixture_video) as random_access:
            for index in (24, 0, 17, 3, 11):
                seeked = random_access.read_frame(index)
                assert np.array_equal(seeked, walked[index]), (
                    f"Seeking to frame {index} returned different pixels than "
                    "reading it sequentially."
                )

    def test_backward_stepping_is_exact(self, fixture_video):
        with FrameReader(fixture_video) as reader:
            forward = {index: reader.read_frame(index).copy() for index in range(10)}
            for index in reversed(range(10)):
                assert np.array_equal(reader.read_frame(index), forward[index])

    def test_long_jump_beyond_grab_limit(self, fixture_video):
        with FrameReader(fixture_video) as reader:
            first = reader.read_frame(0).copy()
            far = reader.read_frame(55).copy()
            assert not np.array_equal(first, far)
            assert np.array_equal(reader.read_frame(0), first)

    def test_index_is_clamped_not_wrapped(self, fixture_video):
        with FrameReader(fixture_video) as reader:
            last = reader.read_frame(reader.last_index).copy()
            assert np.array_equal(reader.read_frame(9999), last)
            first = reader.read_frame(0).copy()
            assert np.array_equal(reader.read_frame(-50), first)

    def test_cache_returns_identical_array(self, fixture_video):
        with FrameReader(fixture_video) as reader:
            a = reader.read_frame(5)
            b = reader.read_frame(5)
            assert np.array_equal(a, b)

    def test_save_frame_png(self, fixture_video, tmp_path):
        destination = tmp_path / "frame_000010.png"
        with FrameReader(fixture_video) as reader:
            saved = reader.save_frame(10, destination)
        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_save_frame_jpg(self, fixture_video, tmp_path):
        destination = tmp_path / "frame.jpg"
        with FrameReader(fixture_video) as reader:
            reader.save_frame(3, destination)
        assert destination.exists()

    def test_save_frame_to_path_with_spaces_and_parentheses(
        self, fixture_video, tmp_path
    ):
        destination = tmp_path / "My Exports (2026)" / "frame 7 (impact).png"
        with FrameReader(fixture_video) as reader:
            reader.save_frame(7, destination)
        assert destination.exists()

    def test_reading_does_not_modify_source(self, fixture_video):
        before = fixture_video.stat()
        with FrameReader(fixture_video) as reader:
            reader.read_frame(20)
            reader.read_frame(0)
        after = fixture_video.stat()
        assert before.st_size == after.st_size
        assert before.st_mtime == after.st_mtime
