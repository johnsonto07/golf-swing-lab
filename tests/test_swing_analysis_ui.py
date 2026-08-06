"""UI acceptance tests for the Swing Analysis page.

Drives the real page script through Streamlit's AppTest harness with a swing
and a stored pose analysis in place, using a fake backend so no model file and
no MediaPipe install are required.

The behaviours pinned here are the ones that would quietly mislead a user if
they regressed: an undetected frame must announce itself rather than showing a
bare image, a stale analysis must be flagged before its numbers are read, and
the page must never start a model download on its own.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit testing harness unavailable"
).AppTest

from golf_lab.models.video import CameraView, SwingContext  # noqa: E402
from golf_lab.pose.inference import estimate_pose_sequence  # noqa: E402
from golf_lab.pose.smoothing import SmoothingSettings, smooth_sequence  # noqa: E402
from golf_lab.storage import pose_repository, swing_repository  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SWING_ANALYSIS = str(REPO_ROOT / "pages" / "2_Swing_Analysis.py")


def _redirect_storage(monkeypatch, swing_root):
    """Point every module-level SWINGS_DIR default at the temp root."""
    for target in (
        "golf_lab.storage.file_repository.SWINGS_DIR",
        "golf_lab.storage.swing_repository.SWINGS_DIR",
    ):
        monkeypatch.setattr(target, swing_root)


@pytest.fixture()
def analysed_swing(fixture_video, swing_root, fake_backend, monkeypatch):
    """A swing with a complete, current pose analysis stored alongside it."""
    record = swing_repository.import_swing(
        source_path=fixture_video,
        original_filename="Range Session (7 iron) 2.mp4",
        context=SwingContext(club="7 Iron", camera_view=CameraView.FACE_ON),
        root=swing_root,
    )
    _redirect_storage(monkeypatch, swing_root)

    video_path = swing_repository.preview_or_original_path(record, swing_root)
    raw = estimate_pose_sequence(video_path, fake_backend(fail_frames=(4, 5, 6)))
    smoothed = smooth_sequence(raw, SmoothingSettings(window_length=7))

    info = pose_repository.build_info(
        swing_id=record.swing_id,
        raw=raw,
        smoothed=smoothed,
        video_path=video_path,
        model_key="full",
        model_filename="pose_landmarker_full.task",
        model_sha256="abc123",
        backend="fake/test-backend",
        device="cpu",
        mediapipe_version="0.10.14",
    )
    pose_repository.save_pose_analysis(record.swing_id, raw, smoothed, info, swing_root)
    return record, video_path, info


@pytest.fixture()
def swing_without_pose(fixture_video, swing_root, monkeypatch):
    record = swing_repository.import_swing(
        source_path=fixture_video,
        original_filename="No Pose Yet.mp4",
        context=SwingContext(club="Driver", camera_view=CameraView.DOWN_THE_LINE),
        root=swing_root,
    )
    _redirect_storage(monkeypatch, swing_root)
    return record


def _find_button(app, label_fragment: str):
    for button in app.button:
        if label_fragment.lower() in button.label.lower():
            return button
    raise AssertionError(
        f"No button matching '{label_fragment}'. "
        f"Found: {[b.label for b in app.button]}"
    )


def _all_text(app) -> str:
    parts = []
    for collection in (app.markdown, app.warning, app.info, app.success, app.error, app.caption):
        parts.extend(str(element.value) for element in collection)
    return " ".join(parts)


class TestPageLoads:
    def test_runs_with_an_analysed_swing(self, analysed_swing):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        assert not app.exception

    def test_runs_with_a_swing_that_has_no_analysis(self, swing_without_pose):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        assert not app.exception
        assert "No pose analysis" in _all_text(app)

    def test_shows_the_stored_metrics(self, analysed_swing):
        _, _, info = analysed_swing
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()

        metrics = {m.label: m.value for m in app.metric}
        assert metrics["Frames detected"] == f"{info.detected_count}/{info.frame_count}"
        assert info.detected_count == 57  # 60 frames, 3 deliberately failed


class TestNoSurpriseDownload:
    def test_page_load_does_not_download_a_model(self, analysed_swing, tmp_path, monkeypatch):
        # Opening a page must never start a network transfer. If it tried,
        # this patched downloader would fail the test.
        def _explode(*args, **kwargs):
            raise AssertionError("the page tried to download a model on its own")

        monkeypatch.setattr("golf_lab.pose.model_manager.download_model", _explode)
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        assert not app.exception

    def test_analyse_button_is_disabled_without_a_model(self, analysed_swing, monkeypatch):
        monkeypatch.setattr(
            "golf_lab.pose.model_manager.is_downloaded", lambda *a, **k: False
        )
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        assert _find_button(app, "Analyse this swing").disabled


class TestFrameStepping:
    def test_next_advances_exactly_one_frame(self, analysed_swing):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        assert app.session_state["frame_index"] == 0

        _find_button(app, "Next frame").click().run()
        assert app.session_state["frame_index"] == 1

        _find_button(app, "Next frame").click().run()
        assert app.session_state["frame_index"] == 2

    def test_previous_goes_back_exactly_one_frame(self, analysed_swing):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        for _ in range(3):
            _find_button(app, "Next frame").click().run()
        _find_button(app, "Previous frame").click().run()
        assert app.session_state["frame_index"] == 2

    def test_frame_confidence_is_reported_per_frame(self, analysed_swing):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        metrics = {m.label: m.value for m in app.metric}
        assert float(metrics["Frame confidence"]) > 0.0


class TestUndetectedFramesAreAnnounced:
    def test_frame_with_no_pose_says_so(self, analysed_swing):
        # Frames 4-6 have no pose. The page must say that rather than showing
        # a bare image that looks like a successful detection.
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        for _ in range(4):
            _find_button(app, "Next frame").click().run()
        assert app.session_state["frame_index"] == 4

        text = _all_text(app)
        assert "No pose was detected" in text
        assert "does not fill in the gap" in text

    def test_frame_with_a_pose_does_not_warn(self, analysed_swing):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        _find_button(app, "Next frame").click().run()
        assert app.session_state["frame_index"] == 1
        assert "No pose was detected" not in _all_text(app)

    def test_confidence_is_zero_on_an_undetected_frame(self, analysed_swing):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        for _ in range(5):
            _find_button(app, "Next frame").click().run()

        metrics = {m.label: m.value for m in app.metric}
        assert float(metrics["Frame confidence"]) == 0.0


class TestStalenessIsSurfaced:
    def test_current_analysis_is_reported_as_current(self, analysed_swing):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        assert "A current pose analysis exists" in _all_text(app)

    def test_changed_video_is_flagged_before_the_numbers(self, analysed_swing):
        # The failure being prevented: showing a skeleton computed from a
        # video the user has since replaced, with no indication.
        _, video_path, _ = analysed_swing
        time.sleep(1.1)  # the fingerprint uses whole-second mtime
        video_path.write_bytes(b"a completely different video")
        os.utime(video_path, None)

        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        text = _all_text(app)
        assert "out of date" in text
        assert "video file has changed" in text


class TestSavingOverlayFrames:
    def test_saves_an_overlay_png_into_exports(self, analysed_swing, swing_root):
        record, _, _ = analysed_swing
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        for _ in range(2):
            _find_button(app, "Next frame").click().run()

        _find_button(app, "Save this frame with overlay").click().run()
        assert not app.exception

        exports = swing_root / record.swing_id / "exports"
        saved = list(exports.glob("pose_frame_*.png"))
        assert saved, f"nothing written to {exports}"
        assert saved[0].stat().st_size > 0

    def test_original_video_is_untouched(self, analysed_swing, swing_root):
        record, _, _ = analysed_swing
        original = swing_root / record.swing_id / record.original_relpath
        before = original.stat()

        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()
        for _ in range(3):
            _find_button(app, "Next frame").click().run()
        _find_button(app, "Save this frame with overlay").click().run()

        after = original.stat()
        assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)


class TestDeletingAnalysis:
    def test_delete_removes_pose_but_keeps_the_video(self, analysed_swing, swing_root):
        record, video_path, _ = analysed_swing
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=90).run()

        _find_button(app, "Delete this swing's pose analysis").click().run()
        assert not app.exception

        assert not pose_repository.has_pose_analysis(record.swing_id, swing_root)
        assert video_path.exists()
