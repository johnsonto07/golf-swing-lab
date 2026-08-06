"""Preview and thumbnail generation, including orientation correctness."""

from __future__ import annotations

import pytest

from golf_lab.video.frame_reader import FrameReader
from golf_lab.video.metadata import extract_metadata
from golf_lab.video.preview import (
    build_preview_command,
    generate_preview,
    generate_thumbnail,
)


class TestPreviewCommand:
    def test_includes_browser_safe_encoding_flags(self, fixture_video):
        meta = extract_metadata(fixture_video)
        command = build_preview_command(
            "ffmpeg", fixture_video, fixture_video.parent / "out.mp4", meta
        )
        assert "libx264" in command
        assert "yuv420p" in command  # required for browser <video> playback
        assert "+faststart" in command
        assert "-an" in command      # preview is silent by design

    def test_no_upscaling_for_small_video(self, fixture_video):
        meta = extract_metadata(fixture_video)
        command = build_preview_command(
            "ffmpeg", fixture_video, fixture_video.parent / "out.mp4", meta, max_width=960
        )
        filters = command[command.index("-vf") + 1]
        assert "scale=" not in filters

    def test_downscales_when_wider_than_cap(self, fixture_video):
        meta = extract_metadata(fixture_video)
        command = build_preview_command(
            "ffmpeg", fixture_video, fixture_video.parent / "out.mp4", meta, max_width=160
        )
        filters = command[command.index("-vf") + 1]
        assert "scale=160:-2" in filters


class TestGeneratePreview:
    def test_creates_playable_preview(self, fixture_video, tmp_path):
        meta = extract_metadata(fixture_video)
        destination = tmp_path / "preview.mp4"
        generate_preview(fixture_video, destination, meta)

        assert destination.exists() and destination.stat().st_size > 0
        preview_meta = extract_metadata(destination)
        assert preview_meta.codec_name == "h264"
        assert preview_meta.has_audio is False

    def test_preserves_frame_count(self, fixture_video, tmp_path):
        """Frame indices in the preview must map 1:1 to the original."""
        meta = extract_metadata(fixture_video)
        destination = tmp_path / "preview.mp4"
        generate_preview(fixture_video, destination, meta)

        preview_meta = extract_metadata(destination)
        assert abs(preview_meta.frame_count - meta.frame_count) <= 1

    def test_preserves_duration_within_tolerance(self, fixture_video, tmp_path):
        meta = extract_metadata(fixture_video)
        destination = tmp_path / "preview.mp4"
        generate_preview(fixture_video, destination, meta)

        preview_meta = extract_metadata(destination)
        # Documented tolerance: within one frame period plus a small margin.
        tolerance = (1.0 / meta.fps) + 0.05
        assert preview_meta.duration_seconds == pytest.approx(
            meta.duration_seconds, abs=tolerance
        )

    def test_bakes_rotation_into_pixels(self, fixture_video_rotated, tmp_path):
        """A portrait-tagged clip must come out upright with no rotation left."""
        meta = extract_metadata(fixture_video_rotated)
        assert meta.rotation_degrees in (90, 270)
        assert (meta.width, meta.height) == (240, 320)

        destination = tmp_path / "preview.mp4"
        generate_preview(fixture_video_rotated, destination, meta)
        preview_meta = extract_metadata(destination)

        # The preview should already be in display orientation...
        assert (preview_meta.width, preview_meta.height) == (240, 320)
        # ...with no residual rotation metadata for anything downstream to apply.
        assert preview_meta.rotation_degrees == 0

    def test_opencv_sees_upright_preview(self, fixture_video_rotated, tmp_path):
        """OpenCV ignores container rotation, so the preview must be pre-rotated."""
        meta = extract_metadata(fixture_video_rotated)
        destination = tmp_path / "preview.mp4"
        generate_preview(fixture_video_rotated, destination, meta)

        with FrameReader(destination) as reader:
            frame = reader.read_frame(0)
        height, width = frame.shape[:2]
        assert (width, height) == (240, 320), (
            "OpenCV read the preview in the wrong orientation."
        )

    def test_does_not_modify_source(self, fixture_video, tmp_path):
        before = fixture_video.stat()
        meta = extract_metadata(fixture_video)
        generate_preview(fixture_video, tmp_path / "preview.mp4", meta)
        after = fixture_video.stat()
        assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)


class TestThumbnail:
    def test_creates_jpeg(self, fixture_video, tmp_path):
        destination = tmp_path / "thumbnail.jpg"
        generate_thumbnail(fixture_video, destination, 0.5)
        assert destination.exists() and destination.stat().st_size > 0

    def test_seek_past_end_falls_back_to_first_frame(self, fixture_video, tmp_path):
        destination = tmp_path / "thumbnail.jpg"
        generate_thumbnail(fixture_video, destination, 9999.0)
        assert destination.exists() and destination.stat().st_size > 0
