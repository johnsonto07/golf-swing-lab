"""UI acceptance tests for the Phases tab.

Drives the real page script with a swing, stored pose data, and a stored phase
analysis in place. The behaviours pinned here are the ones that would quietly
mislead someone if they regressed: an unsupported metric must say it is
unsupported rather than vanish, phases must be labelled as preview frames, and
the tempo block must stay absent while GSL-1 is open.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit testing harness unavailable"
).AppTest

from golf_lab.models.video import CameraView, SwingContext  # noqa: E402
from golf_lab.pose.inference import estimate_pose_sequence  # noqa: E402
from golf_lab.pose.smoothing import SmoothingSettings, smooth_sequence  # noqa: E402
from golf_lab.storage import (  # noqa: E402
    analysis_repository,
    pose_repository,
    swing_repository,
)
from golf_lab.swing.geometry_detector import default_detector  # noqa: E402
from golf_lab.swing.metric_registry import evaluate_all  # noqa: E402
from golf_lab.swing.phases import PHASE_ORDER  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SWING_ANALYSIS = str(REPO_ROOT / "pages" / "2_Swing_Analysis.py")


def _redirect_storage(monkeypatch, swing_root):
    for target in (
        "golf_lab.storage.file_repository.SWINGS_DIR",
        "golf_lab.storage.swing_repository.SWINGS_DIR",
    ):
        monkeypatch.setattr(target, swing_root)


@pytest.fixture()
def swing_with_phases(
    fixture_video, swing_root, fake_backend, swing_pose_factory, monkeypatch, request
):
    """A swing with pose data and phases stored, at a chosen camera view."""
    camera_view = getattr(request, "param", CameraView.FACE_ON)

    record = swing_repository.import_swing(
        source_path=fixture_video,
        original_filename="Phase Test Swing.mp4",
        context=SwingContext(club="7 Iron", camera_view=camera_view),
        root=swing_root,
    )
    _redirect_storage(monkeypatch, swing_root)

    video_path = swing_repository.preview_or_original_path(record, swing_root)
    raw = estimate_pose_sequence(video_path, fake_backend())
    smoothed = smooth_sequence(raw, SmoothingSettings(window_length=7))
    info = pose_repository.build_info(
        swing_id=record.swing_id,
        raw=raw,
        smoothed=smoothed,
        video_path=video_path,
        model_key="full",
        model_filename="pose_landmarker_full.task",
        model_sha256="abc",
        backend="fake/test-backend",
        device="cpu",
        mediapipe_version="0.10.14",
    )
    pose_repository.save_pose_analysis(record.swing_id, raw, smoothed, info, swing_root)

    # Phases from a realistic synthetic swing rather than the flat fake-backend
    # output, so the tab has something meaningful to render.
    sequence = swing_pose_factory()
    detector = default_detector()
    phases = detector.detect(sequence, camera_view)
    phase_frames = {
        phase.value: frame
        for phase in PHASE_ORDER
        if (frame := phases.frame_for(phase)) is not None
    }
    metrics = evaluate_all(sequence, camera_view, phase_frames)
    analysis_repository.save_analysis(
        analysis_repository.SwingAnalysis(
            swing_id=record.swing_id,
            phases=phases,
            metrics=metrics,
            pose_created_at=info.created_at,
            pose_video_fingerprint=info.video_fingerprint,
        ),
        swing_root,
    )
    return record, phases, metrics


def _all_text(app) -> str:
    parts = []
    for collection in (
        app.markdown,
        app.warning,
        app.info,
        app.success,
        app.error,
        app.caption,
    ):
        parts.extend(str(element.value) for element in collection)
    return " ".join(parts)


class TestPhasesTabRenders:
    def test_page_runs_with_stored_phases(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        assert not app.exception

    def test_detected_phases_are_shown_with_preview_frames(self, swing_with_phases):
        _, phases, _ = swing_with_phases
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)

        assert "Address" in text
        assert "Top Of Backswing" in text
        # Frame numbers must be labelled as preview frames, everywhere.
        assert "preview frame" in text.lower()

    def test_unattempted_phases_are_listed_as_such(self, swing_with_phases):
        # "Not attempted by this detector" and "attempted and failed" are
        # different facts; collapsing them hides the one worth acting on.
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)
        assert "not attempted by this detector" in text.lower()

    def test_detector_provenance_is_shown(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)
        assert "hand-path-geometry" in text
        assert "schema v" in text.lower()


class TestTimingHonesty:
    def test_tempo_is_not_offered(self, swing_with_phases):
        # GSL-1 is open. Nothing on this page may present tempo.
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app).lower()

        assert "not offered yet" in text
        assert "gsl-1" in text

        # No ratio like "2.8:1" is displayed. Matched precisely rather than by
        # searching for ":1", which also matches clock times such as the
        # swing's own "imported 2026-08-06 12:16" caption — that made this test
        # fail only during minutes beginning with 1.
        assert re.search(r"\d\.\d+\s*:\s*1(?!\d)", text) is None

    def test_timestamps_are_labelled_preview(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app).lower()
        if "t = " in text:
            assert "preview t" in text


class TestCameraViewGatingInUi:
    @pytest.mark.parametrize(
        "swing_with_phases", [CameraView.FACE_ON], indirect=True
    )
    def test_face_on_shows_face_on_metrics(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)

        assert "Head sway" in text
        # A down-the-line-only metric must not be presented as measurable here.
        assert "Spine angle at address" not in text

    @pytest.mark.parametrize(
        "swing_with_phases", [CameraView.DOWN_THE_LINE], indirect=True
    )
    def test_down_the_line_shows_dtl_metrics(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)

        assert "Spine angle" in text
        assert "Hip sway" not in text

    @pytest.mark.parametrize(
        "swing_with_phases", [CameraView.UNKNOWN], indirect=True
    )
    def test_unknown_view_offers_no_metrics_and_says_why(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)
        assert "No metrics are defined" in text

    @pytest.mark.parametrize(
        "swing_with_phases", [CameraView.FACE_ON], indirect=True
    )
    def test_metric_states_are_explained_not_blank(self, swing_with_phases):
        _, _, metrics = swing_with_phases
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)

        for metric in metrics:
            if not metric.status.is_usable:
                # Every unavailable metric shows its reason rather than a gap
                # the user has to interpret.
                assert metric.reason in text

    @pytest.mark.parametrize(
        "swing_with_phases", [CameraView.FACE_ON], indirect=True
    )
    def test_no_metric_is_rendered_as_zero(self, swing_with_phases):
        _, _, metrics = swing_with_phases
        for metric in metrics:
            if not metric.status.is_usable:
                assert metric.value is None
                assert metric.display_value() == "—"
