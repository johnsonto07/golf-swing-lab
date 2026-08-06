"""Derived-analysis storage: round-trips, versioning, and raw-data safety."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from golf_lab.models.video import CameraView
from golf_lab.pose.sequence import PoseSequence
from golf_lab.storage import analysis_repository as repo
from golf_lab.storage.file_repository import swing_dir
from golf_lab.swing.geometry_detector import default_detector
from golf_lab.swing.metric_registry import evaluate_all
from golf_lab.swing.phases import PHASE_SCHEMA_VERSION, SwingPhase
from golf_lab.swing.results import ResultStatus

SWING_ID = "20260101_120000_abcd1234"


@dataclass
class FakePoseInfo:
    created_at: str = "2026-08-06T10:00:00+00:00"
    video_fingerprint: str = "abc123"


@pytest.fixture()
def analysis(swing_pose_factory, swing_root):
    swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
    sequence = swing_pose_factory()
    detector = default_detector()
    phases = detector.detect(sequence, CameraView.FACE_ON)

    phase_frames = {
        phase.value: frame
        for phase in (SwingPhase.ADDRESS, SwingPhase.TOP_OF_BACKSWING)
        if (frame := phases.frame_for(phase)) is not None
    }
    metrics = evaluate_all(sequence, CameraView.FACE_ON, phase_frames)

    return repo.SwingAnalysis(
        swing_id=SWING_ID,
        phases=phases,
        metrics=metrics,
        pose_created_at=FakePoseInfo().created_at,
        pose_video_fingerprint=FakePoseInfo().video_fingerprint,
    )


class TestRoundTrip:
    def test_save_and_load(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        assert repo.has_analysis(SWING_ID, swing_root)

        loaded = repo.load_analysis(SWING_ID, swing_root)
        assert loaded is not None
        assert loaded.swing_id == SWING_ID
        assert loaded.phases.detector_name == analysis.phases.detector_name
        assert loaded.phases.detector_version == analysis.phases.detector_version
        assert loaded.phases.camera_view is CameraView.FACE_ON

    def test_phase_frames_survive(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        loaded = repo.load_analysis(SWING_ID, swing_root)

        for phase in (SwingPhase.ADDRESS, SwingPhase.TOP_OF_BACKSWING):
            assert loaded.phases.frame_for(phase) == analysis.phases.frame_for(phase)

    def test_metric_statuses_and_reasons_survive(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        loaded = repo.load_analysis(SWING_ID, swing_root)

        assert len(loaded.metrics) == len(analysis.metrics)
        for original, restored in zip(analysis.metrics, loaded.metrics):
            assert restored.key == original.key
            assert restored.status is original.status
            assert restored.reason == original.reason
            if original.value is None:
                assert restored.value is None
            else:
                assert restored.value == pytest.approx(original.value)

    def test_unavailable_metrics_never_gain_a_value_through_storage(
        self, analysis, swing_root
    ):
        repo.save_analysis(analysis, swing_root)
        loaded = repo.load_analysis(SWING_ID, swing_root)

        for metric in loaded.metrics:
            if not metric.status.is_usable:
                assert metric.value is None

    def test_missing_analysis_loads_as_none(self, swing_root):
        assert repo.load_analysis("nope", swing_root) is None
        assert not repo.has_analysis("nope", swing_root)

    def test_corrupt_file_loads_as_none(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        repo.analysis_path(SWING_ID, swing_root).write_text("{oops", encoding="utf-8")
        assert repo.load_analysis(SWING_ID, swing_root) is None

    def test_no_temp_files_survive(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        assert not list(swing_dir(SWING_ID, swing_root).glob("*.tmp"))

    def test_json_is_human_readable(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        data = json.loads(
            repo.analysis_path(SWING_ID, swing_root).read_text(encoding="utf-8")
        )
        assert data["schema_version"] == PHASE_SCHEMA_VERSION
        assert data["phases"]["detector_name"]
        assert isinstance(data["metrics"], list)


class TestRawDataIsPreserved:
    def test_derived_results_live_in_their_own_file(self, analysis, swing_root):
        # Phase detection is a cheap pure function of the landmarks. Storing it
        # beside them, rather than inside them, is what lets a detector change
        # invalidate derived results without re-running MediaPipe.
        repo.save_analysis(analysis, swing_root)
        directory = swing_dir(SWING_ID, swing_root)

        assert (directory / "swing_analysis.json").exists()
        assert not (directory / "pose_raw.npz").exists()

    def test_saving_analysis_does_not_touch_pose_files(
        self, analysis, swing_root, swing_pose_factory
    ):
        directory = swing_dir(SWING_ID, swing_root)
        raw_path = directory / "pose_raw.npz"
        sequence = swing_pose_factory()
        sequence.save(raw_path)

        before_bytes = raw_path.read_bytes()
        before_mtime = raw_path.stat().st_mtime_ns

        repo.save_analysis(analysis, swing_root)

        assert raw_path.read_bytes() == before_bytes
        assert raw_path.stat().st_mtime_ns == before_mtime

    def test_deleting_analysis_leaves_pose_data(
        self, analysis, swing_root, swing_pose_factory
    ):
        directory = swing_dir(SWING_ID, swing_root)
        raw_path = directory / "pose_raw.npz"
        swing_pose_factory().save(raw_path)
        repo.save_analysis(analysis, swing_root)

        repo.delete_analysis(SWING_ID, swing_root)

        assert not repo.has_analysis(SWING_ID, swing_root)
        assert raw_path.exists()
        assert PoseSequence.load(raw_path).frame_count > 0

    def test_reloaded_pose_landmarks_are_bit_identical(
        self, analysis, swing_root, swing_pose_factory
    ):
        directory = swing_dir(SWING_ID, swing_root)
        raw_path = directory / "pose_raw.npz"
        original = swing_pose_factory()
        original.save(raw_path)

        repo.save_analysis(analysis, swing_root)
        repo.delete_analysis(SWING_ID, swing_root)

        reloaded = PoseSequence.load(raw_path)
        np.testing.assert_array_equal(reloaded.detected, original.detected)
        np.testing.assert_allclose(
            reloaded.landmarks, original.landmarks, equal_nan=True, rtol=0
        )


class TestStaleness:
    def test_fresh_analysis_is_current(self, analysis):
        detector = default_detector()
        assert (
            repo.staleness_reasons(
                analysis, FakePoseInfo(), detector.name, detector.version
            )
            == []
        )

    def test_missing_analysis_is_stale(self):
        reasons = repo.staleness_reasons(None)
        assert reasons and "have not been detected" in reasons[0]

    def test_detector_version_change_invalidates(self, analysis):
        detector = default_detector()
        reasons = repo.staleness_reasons(analysis, FakePoseInfo(), detector.name, "99")
        assert any("detector version" in r for r in reasons)

    def test_detector_swap_invalidates(self, analysis):
        reasons = repo.staleness_reasons(
            analysis, FakePoseInfo(), "some-other-detector", "1"
        )
        assert any("detector" in r for r in reasons)

    def test_schema_change_invalidates(self, analysis):
        analysis.schema_version = 99
        assert any("schema" in r for r in repo.staleness_reasons(analysis))

    def test_recomputed_pose_invalidates_derived_results(self, analysis):
        # The landmarks it was derived from changed underneath it.
        newer = FakePoseInfo(created_at="2026-08-07T10:00:00+00:00")
        reasons = repo.staleness_reasons(analysis, newer)
        assert any("recomputed" in r for r in reasons)

    def test_analysis_version_change_invalidates(self, analysis):
        analysis.analysis_version = "0"
        assert any("analysis version" in r for r in repo.staleness_reasons(analysis))

    def test_reasons_are_user_facing_sentences(self, analysis):
        analysis.schema_version = 99
        analysis.analysis_version = "0"
        for reason in repo.staleness_reasons(analysis):
            assert reason[0].isupper()
            assert reason.endswith(".")

    def test_video_change_is_not_double_reported(self, analysis):
        # The pose layer already reports a changed video. Reporting it again
        # here would show the user two warnings for one cause.
        newer = FakePoseInfo(video_fingerprint="totally-different")
        reasons = repo.staleness_reasons(analysis, newer)
        assert not any("video" in r.lower() for r in reasons)


class TestRangePhaseStorage:
    def test_range_phases_survive_the_round_trip(self, analysis, swing_root):
        """Ranges must keep both endpoints, not collapse to a start frame."""
        repo.save_analysis(analysis, swing_root)
        loaded = repo.load_analysis(SWING_ID, swing_root)

        ranges = [r for r in analysis.phases.available if r.is_range]
        assert ranges, "expected the synthetic swing to produce range phases"

        for original in ranges:
            restored = loaded.phases.get(original.phase)
            assert restored.start_frame == original.start_frame
            assert restored.end_frame == original.end_frame
            assert restored.is_range

    def test_impact_region_detail_survives(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        loaded = repo.load_analysis(SWING_ID, swing_root)

        impact = loaded.phases.get(SwingPhase.IMPACT_REGION)
        if impact is None or not impact.status.is_usable:
            pytest.skip("no impact region for this fixture")
        assert "centre_frame" in impact.detail
        assert "half_width_frames" in impact.detail

    def test_all_seven_phases_persist(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        loaded = repo.load_analysis(SWING_ID, swing_root)

        from golf_lab.swing.phases import PHASE_ORDER

        for phase in PHASE_ORDER:
            assert loaded.phases.get(phase) is not None, f"{phase.value} lost"

    def test_cascaded_failures_keep_their_status_and_reason(
        self, swing_pose_factory, swing_root, tmp_path
    ):
        """An unavailable phase must not come back from disk looking located."""
        import numpy as np

        from golf_lab.pose import landmarks as lmk

        swing_dir("cascade_swing", swing_root).mkdir(parents=True, exist_ok=True)
        sequence = swing_pose_factory(address_frames=12, backswing_frames=30)
        count = sequence.frame_count
        drift = np.concatenate(
            [
                np.full(12, 0.45, dtype=np.float32),
                np.linspace(0.45, 0.62, count - 12, dtype=np.float32),
            ]
        )
        for wrist in (lmk.LEFT_WRIST, lmk.RIGHT_WRIST):
            sequence.landmarks[:, wrist, 1] = np.full(count, 0.75, dtype=np.float32)
            sequence.landmarks[:, wrist, 0] = drift

        phases = default_detector().detect(sequence, CameraView.FACE_ON)
        stored = repo.SwingAnalysis(swing_id="cascade_swing", phases=phases)
        repo.save_analysis(stored, swing_root)

        loaded = repo.load_analysis("cascade_swing", swing_root)
        blocked = loaded.phases.get(SwingPhase.IMPACT_REGION)
        assert not blocked.status.is_usable
        assert blocked.start_frame is None
        assert blocked.end_frame is None
        assert blocked.reason


class TestDetectorVersioning:
    def test_detector_v1_analysis_is_stale_against_v2(self, analysis):
        """The version bump must invalidate results from the older detector."""
        analysis.phases.detector_version = "1"
        detector = default_detector()
        assert detector.version == "2"

        reasons = repo.staleness_reasons(
            analysis, FakePoseInfo(), detector.name, detector.version
        )
        assert any("detector version" in r for r in reasons)
        assert repo.is_stale(analysis, FakePoseInfo(), detector.name, detector.version)

    def test_current_detector_output_is_not_stale(self, analysis):
        detector = default_detector()
        assert (
            repo.staleness_reasons(
                analysis, FakePoseInfo(), detector.name, detector.version
            )
            == []
        )

    def test_stored_analysis_records_the_detector_version(self, analysis, swing_root):
        repo.save_analysis(analysis, swing_root)
        loaded = repo.load_analysis(SWING_ID, swing_root)
        assert loaded.phases.detector_version == default_detector().version


class TestVfrHonesty:
    def test_approximate_timeline_flag_survives_storage(
        self, swing_pose_factory, swing_root
    ):
        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        phases = default_detector().detect(
            swing_pose_factory(), CameraView.FACE_ON, timeline_is_approximate=True
        )
        analysis = repo.SwingAnalysis(swing_id=SWING_ID, phases=phases)
        repo.save_analysis(analysis, swing_root)

        loaded = repo.load_analysis(SWING_ID, swing_root)
        assert loaded.phases.timeline_is_approximate
        assert any("GSL-1" in note for note in loaded.phases.notes)

    def test_stored_analysis_offers_no_durations(self, analysis, swing_root):
        # Nothing in the persisted shape may tempt a consumer into treating
        # preview frames as source timing.
        repo.save_analysis(analysis, swing_root)
        data = json.loads(
            repo.analysis_path(SWING_ID, swing_root).read_text(encoding="utf-8")
        )
        serialized = json.dumps(data).lower()
        for forbidden in ("tempo", "duration_seconds", "backswing_seconds"):
            assert forbidden not in serialized
