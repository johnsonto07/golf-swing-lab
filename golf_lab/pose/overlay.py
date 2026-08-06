"""Draw a pose skeleton onto a frame.

Low-confidence joints are drawn dimmer and smaller rather than hidden. Hiding
them would make a bad estimate look like a clean one with fewer joints; fading
them shows you *that* the model was unsure and *where*, which is the
information you need to decide whether to trust a measurement.

Colours are BGR because that is OpenCV's order and the frames come straight
from the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from golf_lab.pose.landmarks import (
    ARM_CONNECTIONS,
    FACE_CONNECTIONS,
    FACE_LANDMARKS,
    LEG_CONNECTIONS,
    NUM_LANDMARKS,
    TORSO_CONNECTIONS,
)
from golf_lab.pose.sequence import PoseSequence

BGR = Tuple[int, int, int]


@dataclass
class OverlayStyle:
    """Appearance of the skeleton overlay."""

    torso_color: BGR = (0, 215, 255)   # amber
    arm_color: BGR = (80, 220, 100)    # green
    leg_color: BGR = (255, 170, 60)    # blue
    face_color: BGR = (200, 200, 200)  # grey
    joint_color: BGR = (255, 255, 255)

    line_thickness: int = 2
    joint_radius: int = 4
    draw_face: bool = True
    draw_legs: bool = True

    # Joints below this visibility are drawn faded. Below `hide_below` they
    # are not drawn at all, because at that point the position is meaningless.
    fade_below: float = 0.5
    hide_below: float = 0.1

    connection_colors: Dict[str, BGR] = field(default_factory=dict)


def _blend(color: BGR, background: BGR, alpha: float) -> BGR:
    """Approximate transparency by mixing toward the background colour."""
    alpha = max(0.0, min(1.0, alpha))
    return tuple(  # type: ignore[return-value]
        int(round(c * alpha + b * (1.0 - alpha)))
        for c, b in zip(color, background)
    )


def _confidence_alpha(confidence: float, style: OverlayStyle) -> float:
    """Map a visibility score to a drawing opacity."""
    if confidence >= style.fade_below:
        return 1.0
    span = max(style.fade_below - style.hide_below, 1e-6)
    return 0.25 + 0.75 * max(0.0, (confidence - style.hide_below) / span)


def draw_pose_on_frame(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    style: Optional[OverlayStyle] = None,
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with the skeleton drawn on it.

    ``points`` is (33, 2) in pixel coordinates. The input frame is never
    modified in place — the caller's decoded frame may be sitting in a cache.
    """
    import cv2

    style = style or OverlayStyle()
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            f"points must have shape ({NUM_LANDMARKS}, 2), got {points.shape}"
        )
    if visibility is None:
        visibility = np.ones(NUM_LANDMARKS, dtype=np.float32)

    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]

    def visible(index: int) -> bool:
        if visibility[index] < style.hide_below:
            return False
        x, y = points[index]
        # NaN comparisons are always False, so this rejects missing points too.
        return bool(np.isfinite(x) and np.isfinite(y))

    groups = [
        (TORSO_CONNECTIONS, style.torso_color),
        (ARM_CONNECTIONS, style.arm_color),
    ]
    if style.draw_legs:
        groups.append((LEG_CONNECTIONS, style.leg_color))
    if style.draw_face:
        groups.append((FACE_CONNECTIONS, style.face_color))

    # Edges first so joints sit on top of them.
    for connections, color in groups:
        for start_index, end_index in connections:
            if not (visible(start_index) and visible(end_index)):
                continue
            confidence = float(min(visibility[start_index], visibility[end_index]))
            drawn = _blend(color, (0, 0, 0), _confidence_alpha(confidence, style))
            cv2.line(
                canvas,
                (int(round(points[start_index][0])), int(round(points[start_index][1]))),
                (int(round(points[end_index][0])), int(round(points[end_index][1]))),
                drawn,
                style.line_thickness,
                lineType=cv2.LINE_AA,
            )

    for index in range(NUM_LANDMARKS):
        if not visible(index):
            continue
        if index in FACE_LANDMARKS and not style.draw_face:
            continue
        confidence = float(visibility[index])
        alpha = _confidence_alpha(confidence, style)
        radius = max(1, int(round(style.joint_radius * (0.6 + 0.4 * alpha))))
        cv2.circle(
            canvas,
            (int(round(points[index][0])), int(round(points[index][1]))),
            radius,
            _blend(style.joint_color, (0, 0, 0), alpha),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # Guard against a landmark predicted outside the image dragging the draw
    # calls out of bounds; OpenCV clips, but the assertion documents intent.
    assert canvas.shape[:2] == (height, width)
    return canvas


def draw_sequence_frame(
    frame_bgr: np.ndarray,
    sequence: PoseSequence,
    frame_index: int,
    style: Optional[OverlayStyle] = None,
) -> Tuple[np.ndarray, bool]:
    """Overlay ``sequence``'s pose for one frame.

    Returns ``(image, was_drawn)``. ``was_drawn`` is False when that frame has
    no pose, so the caller can say "no pose detected on this frame" rather
    than showing a bare image that looks like a successful result.
    """
    height, width = frame_bgr.shape[:2]
    points = sequence.pixel_coordinates(frame_index, width=width, height=height)
    if points is None:
        return frame_bgr, False

    visibility = sequence.visibility[frame_index]
    return draw_pose_on_frame(frame_bgr, points, visibility, style), True
