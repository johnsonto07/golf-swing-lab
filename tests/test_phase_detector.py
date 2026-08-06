"""Phase detection on deterministic synthetic swings.

The fixture builds a swing whose address and top frames are known exactly, so
these assert the detector found *the right frame*, not merely a plausible one.
"""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.models.video import CameraView
from golf_lab.pose.sequence import PoseSequence
from golf_lab.swing.geometry_detector import HandPathPhaseDetector, default_detector
from golf_lab.swing.phases import (
    PHASE_ORDER,
    PhaseResult,
    SwingPhase,
    SwingPhaseDetector,
    SwingPhases,
)
from golf_lab.swing.results import ResultStatus


@pytest.fixture()
def detector():
    return default_detector()


class TestProtocolConformance:
    def test_detector_satisfies_the_protocol(self, detector):
        assert isinstance(detector, SwingPhaseDetector)

    def test_detector_declares_identity_and_scope(self, detector):
        assert detector.name
        assert detector.version
        # It declares what it attempts rather than emitting placeholders for
        # phases it cannot find.
        assert set(detector.supported_phases) == {
            SwingPhase.ADDRESS,
            SwingPhase.TOP_OF_BACKSWING,
        }

    def test_unsupported_phases_are_simply_absent(self, detector, swing_pose_factory):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        for phase in PHASE_ORDER:
            if phase not in detector.supported_phases:
                assert result.get(phase) is None, (
                    f"{phase.value} was reported by a detector that does not "
                    "claim to detect it"
                )

    def test_result_records_provenance(self, detector, swing_pose_factory):
        sequence = swing_pose_factory()
        result = detector.detect(sequence, CameraView.DOWN_THE_LINE)
        assert result.detector_name == detector.name
        assert result.detector_version == detector.version
        assert result.camera_view is CameraView.DOWN_THE_LINE
        assert result.frame_count == sequence.frame_count
        assert result.created_at


class TestAddressDetection:
    def test_finds_the_end_of_the_quiet_period(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(address_frames=12)
        result = detector.detect(sequence, CameraView.FACE_ON)

        address = result.get(SwingPhase.ADDRESS)
        assert address.status.is_usable
        expected = int(sequence.metadata["expected_address"])
        assert abs(address.start_frame - expected) <= 2, (
            f"address at {address.start_frame}, expected near {expected}"
        )

    def test_takes_the_last_quiet_frame_not_the_first(self, detector, swing_pose_factory):
        # A long settle before the swing must not drag address to frame 0.
        sequence = swing_pose_factory(address_frames=25)
        address = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.ADDRESS)
        assert address.start_frame > 15

    def test_fails_cleanly_when_the_swing_is_already_moving(
        self, detector, swing_pose_factory
    ):
        sequence = swing_pose_factory(address_frames=0, backswing_frames=30)
        address = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.ADDRESS)

        assert not address.status.is_usable
        assert address.start_frame is None
        assert "hands are moving" in address.reason

    def test_top_is_not_reported_when_address_failed(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(address_frames=0)
        top = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.TOP_OF_BACKSWING)
        assert not top.status.is_usable
        assert top.start_frame is None


class TestTopDetection:
    def test_finds_the_top_of_the_backswing(self, detector, swing_pose_factory):
        sequence = swing_pose_factory()
        top = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.TOP_OF_BACKSWING)

        assert top.status.is_usable
        expected = int(sequence.metadata["expected_top"])
        assert abs(top.start_frame - expected) <= 3, (
            f"top at {top.start_frame}, expected near {expected}"
        )

    def test_does_not_mistake_the_finish_for_the_top(self, detector, swing_pose_factory):
        # The finish has the hands just as high. Bounding the search by peak
        # hand speed is what keeps them apart, and this is the test that
        # would fail if that bound were removed.
        sequence = swing_pose_factory(follow_frames=40)
        result = detector.detect(sequence, CameraView.FACE_ON)

        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        address = result.get(SwingPhase.ADDRESS)
        assert top.start_frame < sequence.frame_count * 0.65
        assert top.start_frame > address.start_frame

    def test_top_comes_after_address(self, detector, swing_pose_factory):
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        assert (
            result.get(SwingPhase.TOP_OF_BACKSWING).start_frame
            > result.get(SwingPhase.ADDRESS).start_frame
        )

    def test_records_how_it_decided(self, detector, swing_pose_factory):
        top = detector.detect(swing_pose_factory(), CameraView.FACE_ON).get(
            SwingPhase.TOP_OF_BACKSWING
        )
        assert "peak_speed_frame" in top.detail
        assert top.detail["rise_shoulder_widths"] > 0

    def test_no_backswing_means_no_top(self, detector, swing_pose_factory):
        # Hands never rise: a putting stroke, or a clip that stops early.
        sequence = swing_pose_factory(backswing_frames=30)
        from golf_lab.pose import landmarks as lmk

        sequence.landmarks[:, lmk.LEFT_WRIST, 1] = 0.75
        sequence.landmarks[:, lmk.RIGHT_WRIST, 1] = 0.75
        sequence.landmarks[:, lmk.LEFT_WRIST, 0] = np.linspace(
            0.45, 0.55, sequence.frame_count
        )
        sequence.landmarks[:, lmk.RIGHT_WRIST, 0] = np.linspace(
            0.46, 0.56, sequence.frame_count
        )

        top = detector.detect(sequence, CameraView.FACE_ON).get(SwingPhase.TOP_OF_BACKSWING)
        assert not top.status.is_usable
        assert "never rise" in top.reason or "backswing" in top.reason


class TestInsufficientData:
    def test_too_few_frames(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(
            address_frames=2, backswing_frames=2, downswing_frames=1, follow_frames=1
        )
        result = detector.detect(sequence, CameraView.FACE_ON)

        for phase in detector.supported_phases:
            outcome = result.get(phase)
            assert outcome.status is ResultStatus.INSUFFICIENT_FRAMES
            assert outcome.start_frame is None

    def test_mostly_undetected_clip_is_refused(self, detector, swing_pose_factory):
        sequence = swing_pose_factory()
        # Knock out 70% of frames: too sparse to describe a swing.
        for index in range(0, sequence.frame_count):
            if index % 10 >= 3:
                sequence.mark_failed(index)

        result = detector.detect(sequence, CameraView.FACE_ON)
        assert result.get(SwingPhase.ADDRESS).status is ResultStatus.INSUFFICIENT_FRAMES

    def test_empty_sequence(self, detector):
        result = detector.detect(PoseSequence.empty(0, fps=30.0), CameraView.FACE_ON)
        assert result.get(SwingPhase.ADDRESS).status is ResultStatus.INSUFFICIENT_FRAMES


class TestMissingLandmarks:
    def test_invisible_hands_are_reported_as_missing_landmarks(
        self, detector, swing_pose_factory
    ):
        # Both phases are derived from the hand path; without hands there is
        # nothing to measure, and that is a different problem from a short clip.
        sequence = swing_pose_factory(hand_visibility=0.01)
        result = detector.detect(sequence, CameraView.FACE_ON)

        address = result.get(SwingPhase.ADDRESS)
        assert address.status is ResultStatus.MISSING_LANDMARKS
        assert address.start_frame is None
        assert "hands" in address.reason.lower()

    def test_partially_visible_hands_still_work(self, detector, swing_pose_factory):
        # Down-the-line footage routinely occludes one wrist; using the better
        # of the two must keep the detector working.
        sequence = swing_pose_factory(hand_visibility=0.45)
        result = detector.detect(sequence, CameraView.FACE_ON)
        assert result.get(SwingPhase.ADDRESS).status.is_usable


class TestGapsAndNoise:
    def test_survives_scattered_detection_gaps(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(failed_frames=(15, 16, 31, 47))
        result = detector.detect(sequence, CameraView.FACE_ON)

        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        assert top.status.is_usable
        expected = int(sequence.metadata["expected_top"])
        assert abs(top.start_frame - expected) <= 5

    def test_never_anchors_a_phase_to_an_undetected_frame(
        self, detector, swing_pose_factory
    ):
        sequence = swing_pose_factory(failed_frames=(11, 12, 13, 41, 42))
        result = detector.detect(sequence, CameraView.FACE_ON)

        for outcome in result.available:
            assert sequence.detected[outcome.start_frame], (
                f"{outcome.phase.value} anchored to undetected frame "
                f"{outcome.start_frame}"
            )

    def test_tolerates_landmark_jitter(self, detector, swing_pose_factory):
        sequence = swing_pose_factory(jitter=0.004, seed=3)
        result = detector.detect(sequence, CameraView.FACE_ON)

        top = result.get(SwingPhase.TOP_OF_BACKSWING)
        assert top.status.is_usable
        expected = int(sequence.metadata["expected_top"])
        assert abs(top.start_frame - expected) <= 6


class TestCameraViewIndependence:
    @pytest.mark.parametrize(
        "view",
        [CameraView.FACE_ON, CameraView.DOWN_THE_LINE, CameraView.OTHER, CameraView.UNKNOWN],
    )
    def test_phase_detection_works_from_any_view(
        self, detector, swing_pose_factory, view
    ):
        # Phases come from the hand path, which reads similarly from any angle.
        # Metric *gating* is what depends on the view — that is tested
        # separately, and conflating the two would be a design error.
        result = detector.detect(swing_pose_factory(), view)
        assert result.get(SwingPhase.ADDRESS).status.is_usable
        assert result.get(SwingPhase.TOP_OF_BACKSWING).status.is_usable


class TestTimelineHonesty:
    def test_approximate_timeline_is_recorded_and_explained(
        self, detector, swing_pose_factory
    ):
        result = detector.detect(
            swing_pose_factory(), CameraView.FACE_ON, timeline_is_approximate=True
        )
        assert result.timeline_is_approximate
        assert any("GSL-1" in note for note in result.notes)
        assert any("preview frame numbers" in note for note in result.notes)

    def test_exact_timeline_carries_no_warning(self, detector, swing_pose_factory):
        result = detector.detect(
            swing_pose_factory(), CameraView.FACE_ON, timeline_is_approximate=False
        )
        assert not result.timeline_is_approximate
        assert result.notes == []

    def test_no_duration_between_phases_is_offered(self, detector, swing_pose_factory):
        # Durations in seconds are blocked by GSL-1. The container must not
        # expose one, or a caller will use it.
        result = detector.detect(swing_pose_factory(), CameraView.FACE_ON)
        assert not hasattr(result, "duration_seconds")
        assert not hasattr(result, "tempo_ratio")


class TestSwingPhasesContainer:
    def test_available_excludes_failed_phases(self, swing_pose_factory):
        phases = SwingPhases()
        phases.set(PhaseResult.found(SwingPhase.ADDRESS, frame=5, confidence=0.9))
        phases.set(
            PhaseResult.unavailable(
                SwingPhase.TOP_OF_BACKSWING, ResultStatus.DETECTION_FAILED, "nope"
            )
        )
        assert [r.phase for r in phases.available] == [SwingPhase.ADDRESS]
        assert len(phases.attempted) == 2

    def test_frame_for_returns_none_for_unusable_phases(self):
        phases = SwingPhases()
        phases.set(
            PhaseResult.unavailable(
                SwingPhase.ADDRESS, ResultStatus.MISSING_LANDMARKS, "no hands"
            )
        )
        assert phases.frame_for(SwingPhase.ADDRESS) is None

    def test_available_is_returned_in_swing_order(self):
        phases = SwingPhases()
        phases.set(PhaseResult.found(SwingPhase.TOP_OF_BACKSWING, frame=40, confidence=0.9))
        phases.set(PhaseResult.found(SwingPhase.ADDRESS, frame=10, confidence=0.9))
        assert [r.phase for r in phases.available] == [
            SwingPhase.ADDRESS,
            SwingPhase.TOP_OF_BACKSWING,
        ]

    def test_round_trip(self, detector, swing_pose_factory):
        original = detector.detect(
            swing_pose_factory(), CameraView.FACE_ON, timeline_is_approximate=True
        )
        restored = SwingPhases.from_dict(original.to_dict())

        assert restored.detector_name == original.detector_name
        assert restored.detector_version == original.detector_version
        assert restored.camera_view is original.camera_view
        assert restored.timeline_is_approximate
        assert restored.notes == original.notes
        for phase in original.results:
            assert restored.get(phase) == original.get(phase)
