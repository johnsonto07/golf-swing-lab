"""End-to-end UI acceptance tests for the Video Lab with a real swing loaded.

These drive the actual page script through Streamlit's AppTest harness, so they
exercise the same code path the browser does — including the session-state and
callback behaviour that makes single-frame stepping exact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit testing harness unavailable"
).AppTest

from golf_lab.models.video import CameraView, SwingContext  # noqa: E402
from golf_lab.storage import swing_repository  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_LAB = str(REPO_ROOT / "pages" / "1_Video_Lab.py")
HISTORY = str(REPO_ROOT / "pages" / "5_History.py")


@pytest.fixture()
def imported_swing(fixture_video, swing_root, monkeypatch):
    """Import a swing into an isolated root and point the app's storage at it."""
    record = swing_repository.import_swing(
        source_path=fixture_video,
        original_filename="Range Session (7 iron) 2.mp4",
        context=SwingContext(club="7 Iron", camera_view=CameraView.FACE_ON),
        root=swing_root,
    )

    # The pages call the repository with root=None, which resolves to the
    # module-level default. Redirect that default at the temp root.
    monkeypatch.setattr("golf_lab.storage.file_repository.SWINGS_DIR", swing_root)
    monkeypatch.setattr("golf_lab.storage.swing_repository.SWINGS_DIR", swing_root)
    return record


def _find_button(app, label_fragment: str):
    for button in app.button:
        if label_fragment.lower() in button.label.lower():
            return button
    raise AssertionError(
        f"No button matching '{label_fragment}'. "
        f"Found: {[b.label for b in app.button]}"
    )


class TestVideoLabWithSwing:
    def test_page_loads_the_swing(self, imported_swing):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        assert not app.exception
        rendered = " ".join(str(m.value) for m in app.markdown)
        assert "Range Session (7 iron) 2.mp4" in rendered + " ".join(
            str(s.value) for s in app.subheader
        )

    def test_frame_slider_is_present_and_bounded(self, imported_swing):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        assert not app.exception
        sliders = [s for s in app.slider if s.label == "Frame"]
        assert sliders, "Frame slider is missing"
        assert sliders[0].value == 0

    def test_next_button_advances_exactly_one_frame(self, imported_swing):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        assert app.session_state["frame_index"] == 0

        _find_button(app, "Next frame").click().run()
        assert app.session_state["frame_index"] == 1

        _find_button(app, "Next frame").click().run()
        assert app.session_state["frame_index"] == 2

    def test_previous_button_goes_back_exactly_one_frame(self, imported_swing):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        for _ in range(3):
            _find_button(app, "Next frame").click().run()
        assert app.session_state["frame_index"] == 3

        _find_button(app, "Previous frame").click().run()
        assert app.session_state["frame_index"] == 2

    def test_frame_index_cannot_go_negative(self, imported_swing):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        _find_button(app, "First").click().run()
        assert app.session_state["frame_index"] == 0
        # Previous is disabled at frame 0, so the floor holds.
        assert _find_button(app, "Previous frame").disabled

    def test_last_button_jumps_to_final_frame(self, imported_swing):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        _find_button(app, "Last").click().run()
        assert not app.exception

        last = app.session_state["frame_index"]
        assert last >= 55  # fixture is 60 frames
        assert _find_button(app, "Next frame").disabled

    def test_displayed_caption_matches_frame_and_timestamp(self, imported_swing):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        for _ in range(5):
            _find_button(app, "Next frame").click().run()

        index = app.session_state["frame_index"]
        assert index == 5

        expected_seconds = index / imported_swing.video.fps
        metrics = {m.label: m.value for m in app.metric}
        assert metrics["Frame"] == str(index)
        # 5 frames at 30 fps = 0.167 s
        assert f"{expected_seconds:06.3f}" in metrics["Timestamp"]

    def test_save_frame_writes_into_exports(self, imported_swing, swing_root):
        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        for _ in range(4):
            _find_button(app, "Next frame").click().run()

        _find_button(app, "Save frame").click().run()
        assert not app.exception

        exports = swing_root / imported_swing.swing_id / "exports"
        saved = list(exports.glob("frame_*"))
        assert saved, f"No frame written to {exports}"
        assert saved[0].stat().st_size > 0

    def test_original_untouched_after_ui_interaction(self, imported_swing, swing_root):
        original = swing_root / imported_swing.swing_id / imported_swing.original_relpath
        before = original.stat()

        app = AppTest.from_file(VIDEO_LAB, default_timeout=90).run()
        for _ in range(6):
            _find_button(app, "Next frame").click().run()
        _find_button(app, "Save frame").click().run()

        after = original.stat()
        assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)


class TestHistoryWithSwing:
    def test_history_lists_the_swing(self, imported_swing):
        app = AppTest.from_file(HISTORY, default_timeout=90).run()
        assert not app.exception
        rendered = " ".join(str(m.value) for m in app.markdown)
        assert "Range Session (7 iron) 2.mp4" in rendered
