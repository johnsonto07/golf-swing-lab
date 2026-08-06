"""Camera-view gating and metric quality states."""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.models.video import CameraView
from golf_lab.pose import landmarks as lmk
from golf_lab.swing import metric_registry as reg
from golf_lab.swing.results import ResultStatus


@pytest.fixture()
def phase_frames():
    return {"address": 11, "top_of_backswing": 41}


@pytest.fixture()
def swing(swing_pose_factory):
    return swing_pose_factory()


class TestRegistry:
    def test_every_metric_declares_views_and_landmarks(self):
        for spec in reg.all_specs():
            assert spec.camera_views, f"{spec.key} declares no camera view"
            assert spec.required_landmarks, f"{spec.key} declares no landmarks"
            assert spec.description

    def test_landmark_indices_are_valid(self):
        for spec in reg.all_specs():
            for index in spec.required_landmarks:
                assert 0 <= index < lmk.NUM_LANDMARKS

    def test_views_partition_sensibly(self):
        face_on = {s.key for s in reg.specs_for_view(CameraView.FACE_ON)}
        dtl = {s.key for s in reg.specs_for_view(CameraView.DOWN_THE_LINE)}

        assert "head_sway" in face_on and "hip_sway" in face_on
        assert "spine_angle" in dtl and "head_movement" in dtl
        # The whole point of gating: lateral sway is a face-on measurement and
        # spine angle is a down-the-line one. Neither may leak.
        assert "hip_sway" not in dtl
        assert "spine_angle" not in face_on

    def test_unknown_view_supports_nothing(self):
        # "unknown" must not silently behave like a permissive wildcard.
        assert reg.specs_for_view(CameraView.UNKNOWN) == []
        assert reg.specs_for_view(CameraView.OTHER) == []


class TestCameraViewGating:
    def test_face_on_metric_refused_down_the_line(self, swing, phase_frames):
        result = reg.evaluate("hip_sway", swing, CameraView.DOWN_THE_LINE, phase_frames)

        assert result.status is ResultStatus.UNSUPPORTED_CAMERA_VIEW
        assert result.value is None
        assert "face on" in result.reason
        assert "down the line" in result.reason

    def test_down_the_line_metric_refused_face_on(self, swing, phase_frames):
        result = reg.evaluate("spine_angle", swing, CameraView.FACE_ON, phase_frames)
        assert result.status is ResultStatus.UNSUPPORTED_CAMERA_VIEW
        assert result.value is None

    def test_unknown_view_refuses_everything(self, swing, phase_frames):
        for key in ("head_sway", "spine_angle"):
            result = reg.evaluate(key, swing, CameraView.UNKNOWN, phase_frames)
            assert result.status is ResultStatus.UNSUPPORTED_CAMERA_VIEW

    def test_gating_happens_before_any_computation(self, swing, phase_frames, monkeypatch):
        # If the wrong-view check ran after computing, a bad number would exist
        # in memory and could leak out through a later refactor.
        def _explode(*args, **kwargs):
            raise AssertionError("computed a metric for an unsupported view")

        monkeypatch.setattr(reg, "_hip_sway", _explode)
        result = reg.evaluate("hip_sway", swing, CameraView.DOWN_THE_LINE, phase_frames)
        assert result.status is ResultStatus.UNSUPPORTED_CAMERA_VIEW

    def test_evaluate_all_returns_only_valid_metrics_by_default(self, swing, phase_frames):
        results = reg.evaluate_all(swing, CameraView.FACE_ON, phase_frames)
        assert results
        for result in results:
            assert result.status is not ResultStatus.UNSUPPORTED_CAMERA_VIEW

    def test_evaluate_all_can_include_unsupported_for_display(self, swing, phase_frames):
        results = reg.evaluate_all(
            swing, CameraView.FACE_ON, phase_frames, include_unsupported=True
        )
        statuses = {r.status for r in results}
        assert ResultStatus.UNSUPPORTED_CAMERA_VIEW in statuses


class TestComputedMetrics:
    def test_head_sway_is_computed_face_on(self, swing, phase_frames):
        result = reg.evaluate("head_sway", swing, CameraView.FACE_ON, phase_frames)
        assert result.status.is_usable
        assert result.value is not None
        assert np.isfinite(result.value)
        assert result.unit == "shoulder widths"

    def test_shoulder_tilt_is_computed(self, swing, phase_frames):
        result = reg.evaluate("shoulder_tilt", swing, CameraView.FACE_ON, phase_frames)
        assert result.status.is_usable
        # The synthetic figure has level shoulders.
        assert abs(result.value) < 5.0

    def test_level_shoulders_read_as_zero_when_mirrored(
        self, swing_pose_factory, phase_frames
    ):
        """Regression: real MediaPipe output is mirrored, and read as 179°.

        MediaPipe labels landmarks anatomically, so for a golfer facing the
        camera the *left* shoulder appears on the *right* of the image. Raw
        arctan2 then returns ~180° for perfectly level shoulders. Caught by
        running the app on a real clip, not by this suite — the synthetic
        fixture happened to place them the other way round.
        """
        sequence = swing_pose_factory()
        address = phase_frames["address"]
        # Swap the shoulders into the arrangement MediaPipe actually produces.
        left = sequence.landmarks[address, lmk.LEFT_SHOULDER, :2].copy()
        right = sequence.landmarks[address, lmk.RIGHT_SHOULDER, :2].copy()
        sequence.landmarks[address, lmk.LEFT_SHOULDER, :2] = right
        sequence.landmarks[address, lmk.RIGHT_SHOULDER, :2] = left

        result = reg.evaluate("shoulder_tilt", sequence, CameraView.FACE_ON, phase_frames)
        assert result.status.is_usable
        assert abs(result.value) < 5.0, (
            f"level shoulders reported as {result.value:.1f}°"
        )

    def test_tilted_shoulders_report_their_tilt_either_way_round(
        self, swing_pose_factory, phase_frames
    ):
        sequence = swing_pose_factory()
        address = phase_frames["address"]

        def tilt(left_xy, right_xy):
            sequence.landmarks[address, lmk.LEFT_SHOULDER, :2] = left_xy
            sequence.landmarks[address, lmk.RIGHT_SHOULDER, :2] = right_xy
            return reg.evaluate(
                "shoulder_tilt", sequence, CameraView.FACE_ON, phase_frames
            ).value

        # Same physical tilt, landmarks in either image arrangement.
        normal = tilt([0.41, 0.34], [0.59, 0.38])
        mirrored = tilt([0.59, 0.38], [0.41, 0.34])

        assert abs(normal) == pytest.approx(abs(mirrored), abs=0.5)
        assert 5.0 < abs(normal) < 45.0

    def test_spine_angle_is_computed_down_the_line(self, swing, phase_frames):
        result = reg.evaluate("spine_angle", swing, CameraView.DOWN_THE_LINE, phase_frames)
        assert result.status.is_usable
        assert 0.0 <= result.value <= 90.0

    def test_results_record_the_frames_they_used(self, swing, phase_frames):
        result = reg.evaluate("head_sway", swing, CameraView.FACE_ON, phase_frames)
        assert result.detail["frames"]["address"] == 11


class TestQualityStates:
    def test_missing_phase_reports_insufficient_frames(self, swing):
        # head_sway needs both address and top.
        result = reg.evaluate("head_sway", swing, CameraView.FACE_ON, {"address": 11})

        assert result.status is ResultStatus.INSUFFICIENT_FRAMES
        assert result.value is None
        assert "top of backswing" in result.reason

    def test_undetected_frame_reports_missing_landmarks(
        self, swing_pose_factory, phase_frames
    ):
        sequence = swing_pose_factory()
        sequence.mark_failed(phase_frames["address"])

        result = reg.evaluate("head_sway", sequence, CameraView.FACE_ON, phase_frames)
        assert result.status is ResultStatus.MISSING_LANDMARKS
        assert result.value is None
        assert str(phase_frames["address"]) in result.reason

    def test_invisible_landmark_reports_missing_landmarks(
        self, swing_pose_factory, phase_frames
    ):
        sequence = swing_pose_factory()
        sequence.visibility[phase_frames["address"], lmk.NOSE] = 0.05

        result = reg.evaluate("head_sway", sequence, CameraView.FACE_ON, phase_frames)
        assert result.status is ResultStatus.MISSING_LANDMARKS
        assert result.value is None

    def test_weak_landmark_reports_low_confidence_with_a_value(
        self, swing_pose_factory, phase_frames
    ):
        # Between "missing" and "fine": a real measurement worth showing, with
        # a caveat attached rather than silently presented as clean.
        sequence = swing_pose_factory()
        sequence.visibility[phase_frames["address"], lmk.NOSE] = 0.35

        result = reg.evaluate("head_sway", sequence, CameraView.FACE_ON, phase_frames)
        assert result.status is ResultStatus.LOW_CONFIDENCE
        assert result.value is not None
        assert "0.35" in result.reason

    def test_unimplemented_metric_never_invents_a_value(self, swing, phase_frames):
        assert not reg.is_implemented("lead_arm_angle")
        result = reg.evaluate("lead_arm_angle", swing, CameraView.FACE_ON, phase_frames)

        assert result.value is None
        assert "not implemented" in result.reason

    def test_degenerate_geometry_fails_rather_than_returning_zero(
        self, swing_pose_factory, phase_frames
    ):
        # Collapse the shoulders so body-scale normalization is undefined.
        # A zero here would read as "no head sway at all".
        sequence = swing_pose_factory()
        for frame in phase_frames.values():
            sequence.landmarks[frame, lmk.LEFT_SHOULDER, :2] = [0.5, 0.36]
            sequence.landmarks[frame, lmk.RIGHT_SHOULDER, :2] = [0.5, 0.36]

        result = reg.evaluate("head_sway", sequence, CameraView.FACE_ON, phase_frames)
        assert not result.status.is_usable
        assert result.value is None

    def test_no_result_is_ever_a_silent_zero(self, swing_pose_factory, phase_frames):
        # Sweep every unavailable path and assert none produced a number.
        sequence = swing_pose_factory()
        sequence.mark_failed(phase_frames["top_of_backswing"])

        results = reg.evaluate_all(
            sequence, CameraView.FACE_ON, phase_frames, include_unsupported=True
        )
        assert results
        for result in results:
            if not result.status.is_usable:
                assert result.value is None, (
                    f"{result.key} reported {result.status.value} with value "
                    f"{result.value!r}"
                )
                assert result.reason
