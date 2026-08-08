"""The tracer data model: provenance, the impact rule, and presets."""

from __future__ import annotations

import pytest

from golf_lab.models.video import Handedness, ShotShape
from golf_lab.tracer.model import (
    BallPoint,
    CurveControls,
    ImpactSource,
    PointSource,
    TracerError,
    TracerHeight,
    TracerSpec,
    seed_controls,
)

SWING_ID = "20260101_120000_abcd1234"


def a_spec(impact: int | None = 10) -> TracerSpec:
    spec = TracerSpec(swing_id=SWING_ID)
    if impact is not None:
        spec.confirm_impact(impact, ImpactSource.USER)
    return spec


def a_point(frame: int, source: PointSource = PointSource.CONFIRMED) -> BallPoint:
    confidence = 0.8 if source is PointSource.TRACKED else None
    return BallPoint(frame=frame, x=0.5, y=0.5, source=source, confidence=confidence)


class TestBallPoint:
    def test_confirmed_point_outside_the_frame_is_refused(self):
        with pytest.raises(TracerError, match="inside the frame"):
            BallPoint(frame=1, x=1.4, y=0.5, source=PointSource.CONFIRMED)

    def test_estimated_point_may_leave_the_frame(self):
        # A struck ball flies out of shot; the curve has to follow it.
        point = BallPoint(frame=1, x=1.4, y=-0.2, source=PointSource.ESTIMATED)
        assert not point.is_inside_frame

    def test_tracked_point_may_leave_the_frame(self):
        point = BallPoint(
            frame=1, x=-0.1, y=0.5, source=PointSource.TRACKED, confidence=0.5
        )
        assert not point.is_inside_frame

    def test_negative_frame_is_refused(self):
        with pytest.raises(TracerError, match="frame index"):
            BallPoint(frame=-1, x=0.5, y=0.5, source=PointSource.CONFIRMED)

    def test_only_tracked_points_carry_confidence(self):
        with pytest.raises(TracerError, match="Only tracked points"):
            BallPoint(
                frame=1,
                x=0.5,
                y=0.5,
                source=PointSource.CONFIRMED,
                confidence=0.9,
            )

    def test_confidence_outside_zero_to_one_is_refused(self):
        with pytest.raises(TracerError, match="between 0 and 1"):
            BallPoint(
                frame=1, x=0.5, y=0.5, source=PointSource.TRACKED, confidence=1.5
            )

    def test_to_pixels_scales_by_frame_size(self):
        point = BallPoint(frame=1, x=0.5, y=0.25, source=PointSource.CONFIRMED)
        assert point.to_pixels(960, 540) == (480.0, 135.0)

    def test_to_pixels_refuses_unknown_dimensions(self):
        # Multiplying by zero would yield a confident (0, 0).
        point = BallPoint(frame=1, x=0.5, y=0.5, source=PointSource.CONFIRMED)
        with pytest.raises(TracerError, match="dimensions are unknown"):
            point.to_pixels(0, 540)

    def test_correcting_a_tracked_point_makes_it_confirmed(self):
        tracked = a_point(5, PointSource.TRACKED)
        corrected = tracked.moved_to(0.4, 0.6)

        assert corrected.source is PointSource.CONFIRMED
        assert (corrected.x, corrected.y) == (0.4, 0.6)
        # The tracker's confidence must not survive a human correction.
        assert corrected.confidence is None

    def test_estimated_points_are_not_observed(self):
        assert PointSource.CONFIRMED.is_observed
        assert PointSource.TRACKED.is_observed
        assert not PointSource.ESTIMATED.is_observed

    def test_round_trip_preserves_provenance(self):
        for source in PointSource:
            point = a_point(3, source)
            assert BallPoint.from_dict(point.to_dict()) == point

    def test_confidence_is_omitted_when_absent(self):
        assert "confidence" not in a_point(3).to_dict()


class TestImpactRule:
    def test_points_cannot_be_placed_before_impact_is_confirmed(self):
        spec = a_spec(impact=None)
        with pytest.raises(TracerError, match="Confirm the impact frame"):
            spec.add_point(a_point(5))

    def test_point_before_impact_is_refused(self):
        spec = a_spec(impact=10)
        with pytest.raises(TracerError, match="before the confirmed impact"):
            spec.add_point(a_point(9))

    def test_point_on_the_impact_frame_is_allowed(self):
        spec = a_spec(impact=10)
        spec.add_point(a_point(10))
        assert len(spec.points) == 1

    def test_tracer_is_invisible_before_impact(self):
        spec = a_spec(impact=10)
        assert not spec.is_visible_at(9)
        assert spec.is_visible_at(10)
        assert spec.is_visible_at(11)

    def test_tracer_is_invisible_everywhere_without_impact(self):
        spec = a_spec(impact=None)
        assert not spec.is_visible_at(0)
        assert not spec.is_visible_at(999)

    def test_moving_impact_later_drops_stranded_points(self):
        spec = a_spec(impact=10)
        spec.add_point(a_point(10))
        spec.add_point(a_point(12))
        spec.add_point(a_point(20))

        spec.confirm_impact(15, ImpactSource.USER)

        assert [p.frame for p in spec.points] == [20]

    def test_impact_frame_and_source_must_be_set_together(self):
        with pytest.raises(TracerError, match="set together"):
            TracerSpec(swing_id=SWING_ID, impact_frame=10)
        with pytest.raises(TracerError, match="set together"):
            TracerSpec(swing_id=SWING_ID, impact_source=ImpactSource.USER)

    def test_detector_and_user_impact_are_distinguishable(self):
        spec = a_spec(impact=None)
        spec.confirm_impact(10, ImpactSource.DETECTOR)
        assert spec.impact_source is ImpactSource.DETECTOR

        spec.confirm_impact(12, ImpactSource.USER)
        assert spec.impact_source is ImpactSource.USER

    def test_negative_impact_frame_is_refused(self):
        with pytest.raises(TracerError, match="frame index"):
            a_spec(impact=None).confirm_impact(-1, ImpactSource.USER)


class TestPointManagement:
    def test_one_frame_holds_one_ball(self):
        spec = a_spec(impact=0)
        spec.add_point(BallPoint(frame=5, x=0.1, y=0.1, source=PointSource.CONFIRMED))
        spec.add_point(BallPoint(frame=5, x=0.9, y=0.9, source=PointSource.CONFIRMED))

        assert len(spec.points) == 1
        assert spec.point_at(5).x == 0.9

    def test_points_stay_sorted_by_frame(self):
        spec = a_spec(impact=0)
        for frame in (30, 10, 20):
            spec.add_point(a_point(frame))

        assert [p.frame for p in spec.points] == [10, 20, 30]

    def test_remove_point_reports_whether_it_existed(self):
        spec = a_spec(impact=0)
        spec.add_point(a_point(10))

        assert spec.remove_point(10) is True
        assert spec.remove_point(10) is False

    def test_observed_and_confirmed_points_are_separable(self):
        spec = a_spec(impact=0)
        spec.add_point(a_point(10, PointSource.CONFIRMED))
        spec.add_point(a_point(20, PointSource.TRACKED))
        spec.add_point(a_point(30, PointSource.ESTIMATED))

        assert [p.frame for p in spec.confirmed_points] == [10]
        assert [p.frame for p in spec.observed_points] == [10, 20]


class TestPresets:
    def test_straight_seeds_no_curvature(self):
        controls = seed_controls(ShotShape.STRAIGHT, TracerHeight.MEDIUM)
        assert controls.curvature == 0.0
        assert controls.launch_direction_degrees == 0.0

    def test_slice_bends_further_than_fade(self):
        fade = seed_controls(ShotShape.FADE, TracerHeight.MEDIUM)
        slice_ = seed_controls(ShotShape.SLICE, TracerHeight.MEDIUM)
        assert slice_.curvature > fade.curvature > 0

    def test_draw_and_hook_bend_the_other_way(self):
        draw = seed_controls(ShotShape.DRAW, TracerHeight.MEDIUM)
        hook = seed_controls(ShotShape.HOOK, TracerHeight.MEDIUM)
        assert hook.curvature < draw.curvature < 0

    def test_push_and_pull_start_offline_but_fly_straight(self):
        push = seed_controls(ShotShape.PUSH, TracerHeight.MEDIUM)
        pull = seed_controls(ShotShape.PULL, TracerHeight.MEDIUM)

        assert push.curvature == 0.0 and pull.curvature == 0.0
        assert push.launch_direction_degrees > 0
        assert pull.launch_direction_degrees < 0

    def test_left_handed_shapes_mirror(self):
        right = seed_controls(ShotShape.FADE, TracerHeight.MEDIUM, Handedness.RIGHT)
        left = seed_controls(ShotShape.FADE, TracerHeight.MEDIUM, Handedness.LEFT)
        assert left.curvature == -right.curvature

    def test_height_is_unaffected_by_handedness(self):
        right = seed_controls(ShotShape.FADE, TracerHeight.HIGH, Handedness.RIGHT)
        left = seed_controls(ShotShape.FADE, TracerHeight.HIGH, Handedness.LEFT)
        assert left.apex_height == right.apex_height

    def test_height_presets_are_ordered(self):
        heights = [
            seed_controls(ShotShape.STRAIGHT, h).apex_height
            for h in (TracerHeight.LOW, TracerHeight.MEDIUM, TracerHeight.HIGH)
        ]
        assert heights == sorted(heights)

    def test_unknown_shape_asserts_nothing(self):
        controls = seed_controls(ShotShape.UNKNOWN, TracerHeight.MEDIUM)
        assert controls.curvature == 0.0
        assert controls.launch_direction_degrees == 0.0

    def test_every_shot_shape_has_a_preset(self):
        # A missing entry would raise KeyError at the worst possible moment.
        for shape in ShotShape:
            seed_controls(shape, TracerHeight.MEDIUM)

    def test_applying_a_preset_replaces_manual_edits(self):
        spec = a_spec()
        spec.controls.curvature = 0.9
        spec.apply_preset(ShotShape.STRAIGHT, TracerHeight.LOW)

        assert spec.controls.curvature == 0.0
        assert spec.shape is ShotShape.STRAIGHT
        assert spec.height is TracerHeight.LOW

    def test_preset_respects_the_specs_handedness(self):
        spec = TracerSpec(swing_id=SWING_ID, handedness=Handedness.LEFT)
        spec.apply_preset(ShotShape.FADE, TracerHeight.MEDIUM)
        assert spec.controls.curvature < 0

    def test_default_controls_assert_nothing(self):
        controls = CurveControls()
        assert controls.curvature == 0.0
        assert controls.launch_direction_degrees == 0.0
        # Not placed is different from placed at the origin.
        assert controls.endpoint is None


class TestSerialization:
    def test_round_trip_preserves_everything(self):
        spec = a_spec(impact=10)
        spec.apply_preset(ShotShape.DRAW, TracerHeight.HIGH)
        spec.controls.endpoint = (0.8, 0.2)
        spec.preview_fingerprint = "deadbeef"
        spec.add_point(a_point(10, PointSource.CONFIRMED))
        spec.add_point(a_point(14, PointSource.TRACKED))
        spec.add_point(a_point(18, PointSource.ESTIMATED))

        restored = TracerSpec.from_dict(spec.to_dict())

        assert restored.to_dict() == spec.to_dict()
        assert restored.impact_frame == 10
        assert restored.impact_source is ImpactSource.USER
        assert [p.source for p in restored.points] == [
            PointSource.CONFIRMED,
            PointSource.TRACKED,
            PointSource.ESTIMATED,
        ]
        assert restored.controls.endpoint == (0.8, 0.2)

    def test_round_trip_without_impact(self):
        spec = TracerSpec(swing_id=SWING_ID)
        restored = TracerSpec.from_dict(spec.to_dict())

        assert restored.impact_frame is None
        assert restored.impact_source is None
        assert not restored.has_impact

    def test_missing_optional_fields_load(self):
        restored = TracerSpec.from_dict({"swing_id": SWING_ID})

        assert restored.swing_id == SWING_ID
        assert restored.points == []
        assert restored.shape is ShotShape.UNKNOWN
        assert restored.height is TracerHeight.MEDIUM
        assert not restored.has_impact

    def test_endpoint_survives_being_absent(self):
        restored = TracerSpec.from_dict(TracerSpec(swing_id=SWING_ID).to_dict())
        assert restored.controls.endpoint is None
