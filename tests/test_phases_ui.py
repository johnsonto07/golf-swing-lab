"""UI acceptance tests for the Phases tab.

Drives the real page script with a swing, stored pose data, and a stored phase
analysis in place. The behaviours pinned here are the ones that would quietly
mislead someone if they regressed: an unsupported metric must say it is
unsupported rather than vanish, phases must be labelled as preview frames, and
and tempo appears only when the measured timeline can support it.
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
    timeline_repository,
)
from golf_lab.swing.geometry_detector import default_detector  # noqa: E402
from golf_lab.swing.metric_registry import evaluate_all  # noqa: E402
from golf_lab.swing.phases import PHASE_ORDER, SwingPhase, SwingPhases  # noqa: E402

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

    def test_all_seven_phases_are_rendered(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)
        for phase in PHASE_ORDER:
            assert phase.display_name in text, f"{phase.value} missing from the page"

    def test_unattempted_phases_are_listed_as_such(
        self, swing_with_phases, swing_root
    ):
        """A narrower detector must render as "not attempted", not as a gap.

        The current detector attempts every phase, so this stores an analysis
        from a hypothetical narrower one to exercise the branch. "Not attempted
        by this detector" and "attempted and failed" are different facts, and
        collapsing them hides the one worth acting on.
        """
        record, phases, metrics = swing_with_phases

        narrowed = SwingPhases(
            detector_name=phases.detector_name,
            detector_version=phases.detector_version,
            camera_view=phases.camera_view,
            frame_count=phases.frame_count,
            preview_fps=phases.preview_fps,
        )
        for phase in (SwingPhase.ADDRESS, SwingPhase.TOP_OF_BACKSWING):
            narrowed.set(phases.get(phase))

        stored = analysis_repository.load_analysis(record.swing_id, swing_root)
        stored.phases = narrowed
        analysis_repository.save_analysis(stored, swing_root)

        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        assert "not attempted by this detector" in _all_text(app).lower()

    def test_detector_provenance_is_shown(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)
        assert "hand-path-geometry" in text
        assert "schema v" in text.lower()


class TestRangePhaseRendering:
    def test_range_phases_render_as_ranges(self, swing_with_phases):
        # Collapsing the impact region to a single frame would present a span
        # the camera cannot resolve as an instant.
        _, phases, _ = swing_with_phases
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app)

        impact = phases.get(SwingPhase.IMPACT_REGION)
        if impact and impact.status.is_usable and impact.is_range:
            assert f"{impact.start_frame}–{impact.end_frame}" in text
            assert "frames**" in text or "frames)" in text

    def test_impact_region_carries_its_caveat(self, swing_with_phases):
        _, phases, _ = swing_with_phases
        impact = phases.get(SwingPhase.IMPACT_REGION)
        if not (impact and impact.status.is_usable):
            pytest.skip("no impact region detected for this fixture")

        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app).lower()
        assert "region, not a frame" in text
        assert "clubhead is not tracked" in text

    def test_every_located_phase_offers_a_jump_button(self, swing_with_phases):
        _, phases, _ = swing_with_phases
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        labels = [b.label for b in app.button]

        for outcome in phases.available:
            assert f"Jump to {outcome.phase.display_name}" in labels

    def test_unavailable_phases_offer_no_jump_button(self, swing_with_phases):
        _, phases, _ = swing_with_phases
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        labels = [b.label for b in app.button]

        for outcome in phases.attempted:
            if not outcome.status.is_usable:
                assert f"Jump to {outcome.phase.display_name}" not in labels


class TestTimingHonesty:
    def test_tempo_is_refused_without_a_measured_timeline(self, swing_with_phases):
        """Tempo is now offered — but only where timing was measured.

        This fixture stores no timeline, so the page must show the tempo row
        as blocked with a reason rather than computing a ratio from a nominal
        frame rate. Replaces the older assertion that tempo was never offered
        at all, which described the state before timing could be measured.
        """
        record, _, _ = swing_with_phases
        # Import measures timing, so remove it to reach the refusal path.
        timeline_repository.delete_timeline(record.swing_id)

        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        lowered = _all_text(app).lower()

        assert "tempo" in lowered, "the tempo row should be present, not absent"

        # The substantive guarantee: with no measured timing, no ratio number
        # appears anywhere. Matched precisely rather than by searching for
        # ":1", which also matches clock times such as the swing's own
        # "imported 2026-08-06 12:16" caption — that made an earlier version of
        # this test fail only during minutes beginning with 1.
        assert re.search(r"\d\.\d+\s*:\s*1(?!\d)", lowered) is None

    def test_tempo_is_offered_when_timing_is_measured(self, swing_with_phases):
        """Import measures timing, so the fixture supports a real ratio.

        The counterpart to the refusal test: tempo is no longer blocked
        wholesale — it is blocked exactly when the timing cannot support it.
        """
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        lowered = _all_text(app).lower()

        assert "tempo" in lowered
        assert "measured" in lowered, "the timing basis must be stated"
        assert re.search(r"\d\.\d+\s*:\s*1(?!\d)", lowered) is not None, (
            "a measured timeline should produce an actual tempo ratio"
        )

    def test_source_timing_states_its_basis(self, swing_with_phases):
        app = AppTest.from_file(SWING_ANALYSIS, default_timeout=120).run()
        text = _all_text(app).lower()
        assert "source-time timing" in text
        # Durations must never be presented without saying what they rest on.
        assert "measured" in text

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
