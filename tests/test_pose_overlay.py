"""Skeleton overlay drawing."""

from __future__ import annotations

import numpy as np
import pytest

from golf_lab.pose.landmarks import NUM_LANDMARKS
from golf_lab.pose.overlay import (
    OverlayStyle,
    draw_pose_on_frame,
    draw_sequence_frame,
)


@pytest.fixture()
def blank_frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture()
def centred_points():
    """A plausible skeleton spread across the middle of the frame."""
    points = np.zeros((NUM_LANDMARKS, 2), dtype=np.float64)
    points[:, 0] = np.linspace(80, 240, NUM_LANDMARKS)
    points[:, 1] = np.linspace(40, 200, NUM_LANDMARKS)
    return points


class TestDrawing:
    def test_draws_something(self, blank_frame, centred_points):
        result = draw_pose_on_frame(blank_frame, centred_points)
        assert result.shape == blank_frame.shape
        assert result.any(), "nothing was drawn onto the frame"

    def test_does_not_modify_the_input_frame(self, blank_frame, centred_points):
        # The caller's frame may be sitting in the LRU frame cache; drawing on
        # it in place would poison every later read of that frame.
        before = blank_frame.copy()
        draw_pose_on_frame(blank_frame, centred_points)
        np.testing.assert_array_equal(blank_frame, before)

    def test_rejects_wrong_point_shape(self, blank_frame):
        with pytest.raises(ValueError):
            draw_pose_on_frame(blank_frame, np.zeros((10, 2)))

    def test_hidden_below_threshold_draws_nothing(self, blank_frame, centred_points):
        visibility = np.zeros(NUM_LANDMARKS, dtype=np.float32)
        result = draw_pose_on_frame(blank_frame, centred_points, visibility)
        assert not result.any()

    def test_low_confidence_is_dimmer_than_high(self, blank_frame, centred_points):
        # Faded rather than hidden: you need to see *that* the model was
        # unsure, not be shown a confident-looking skeleton with gaps.
        bright = draw_pose_on_frame(
            blank_frame, centred_points, np.full(NUM_LANDMARKS, 1.0, dtype=np.float32)
        )
        dim = draw_pose_on_frame(
            blank_frame, centred_points, np.full(NUM_LANDMARKS, 0.2, dtype=np.float32)
        )
        assert dim.sum() < bright.sum()
        assert dim.any(), "low-confidence joints should still be visible"

    def test_nan_points_are_skipped_not_drawn_at_origin(self, blank_frame, centred_points):
        points = centred_points.copy()
        points[5] = np.nan
        result = draw_pose_on_frame(blank_frame, points)
        # A NaN turned into int would land at a garbage coordinate; the top-left
        # corner staying black shows it was skipped instead.
        assert result[0:5, 0:5].sum() == 0

    def test_points_outside_the_frame_do_not_crash(self, blank_frame, centred_points):
        points = centred_points.copy()
        points[0] = [-500, -500]
        points[1] = [9999, 9999]
        result = draw_pose_on_frame(blank_frame, points)
        assert result.shape == blank_frame.shape

    def test_face_and_legs_can_be_turned_off(self, blank_frame, centred_points):
        full = draw_pose_on_frame(blank_frame, centred_points, style=OverlayStyle())
        reduced = draw_pose_on_frame(
            blank_frame,
            centred_points,
            style=OverlayStyle(draw_face=False, draw_legs=False),
        )
        assert reduced.sum() < full.sum()


class TestDrawSequenceFrame:
    def test_reports_when_a_pose_was_drawn(self, blank_frame, pose_sequence_factory):
        sequence = pose_sequence_factory(frame_count=5, width=320, height=240)
        image, drawn = draw_sequence_frame(blank_frame, sequence, 0)
        assert drawn
        assert image.any()

    def test_reports_when_a_frame_has_no_pose(self, blank_frame, pose_sequence_factory):
        # The caller needs to be able to say "no pose on this frame" rather
        # than showing a bare image that looks like a successful result.
        sequence = pose_sequence_factory(frame_count=5, failed=(2,))
        image, drawn = draw_sequence_frame(blank_frame, sequence, 2)
        assert not drawn
        np.testing.assert_array_equal(image, blank_frame)

    def test_uses_actual_frame_dimensions(self, pose_sequence_factory):
        # The overlay must scale to the image it is given, not to whatever
        # dimensions were recorded at inference time.
        sequence = pose_sequence_factory(frame_count=3, width=320, height=240)
        big_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        image, drawn = draw_sequence_frame(big_frame, sequence, 0)
        assert drawn
        assert image.shape == (480, 640, 3)
