"""Render a synthetic golf swing clip for documentation and demos.

Deliberately synthetic: the README needs screenshots, and the repository must
never contain anyone's real golf footage. The figure is drawn with human
proportions because MediaPipe needs to recognise it as a person, and it moves
through a full swing so the pose overlay has something meaningful to track.

Constant frame rate, so it also serves as the CFR counterexample to the VFR
clip that produced GSL-1.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 720, 1280
FPS = 30
FRAMES = 150

SKIN = (120, 140, 190)
SHIRT = (95, 75, 62)
TROUSERS = (72, 60, 108)
SHOE = (40, 40, 40)
CLUB = (60, 60, 60)
GRASS = (90, 160, 110)
SKY = (200, 190, 175)


def _swing_phase(t: float) -> float:
    """Swing angle in degrees: 0 at address, negative up, positive through.

    Piecewise so the motion has the right *rhythm* — a slow backswing and a
    much faster downswing — which is what makes the overlay look like a swing
    rather than a metronome.
    """
    if t < 0.13:                      # address, still
        return 0.0
    if t < 0.50:                      # backswing
        u = (t - 0.13) / 0.37
        return -175.0 * (1 - math.cos(u * math.pi / 2))
    if t < 0.62:                      # downswing, fast
        u = (t - 0.50) / 0.12
        return -175.0 + 175.0 * u
    if t < 0.78:                      # through impact
        u = (t - 0.62) / 0.16
        return 175.0 * u
    return 175.0                      # finish, held


def draw_frame(index: int) -> np.ndarray:
    t = index / (FRAMES - 1)
    angle = _swing_phase(t)

    img = np.full((HEIGHT, WIDTH, 3), SKY, dtype=np.uint8)
    cv2.rectangle(img, (0, 880), (WIDTH, HEIGHT), GRASS, -1)

    cx, hip_y = WIDTH // 2, 700
    shoulder_y = 470
    # The head must sit ON the shoulders with a neck. A floating head above a
    # gap is not recognised as a person at all — it was the single reason
    # address frames failed detection entirely.
    head_y = shoulder_y - 92

    # Weight shifts back then through, and the body rises into the finish.
    lean = int(18 * math.sin(math.radians(angle) / 2))
    rise = int(28 * max(0.0, (angle / 175.0)))

    head = (cx + lean // 2, head_y - rise)
    l_sh = (cx - 62 + lean, shoulder_y - rise)
    r_sh = (cx + 62 + lean, shoulder_y - rise)
    l_hip = (cx - 44, hip_y - rise // 2)
    r_hip = (cx + 44, hip_y - rise // 2)

    # Hands travel on an arc around the sternum.
    # angle 0 puts the hands straight down from the sternum (address);
    # negative swings them up and behind, positive up and through.
    pivot = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2 + 40)
    radius = 200
    rad = math.radians(angle)
    unit = (math.sin(rad), math.cos(rad))
    hands = (int(pivot[0] + radius * unit[0]), int(pivot[1] + radius * unit[1]))

    def elbow(shoulder, bend):
        mx = (shoulder[0] + hands[0]) // 2
        my = (shoulder[1] + hands[1]) // 2
        return (mx + bend, my)

    l_el, r_el = elbow(l_sh, -22), elbow(r_sh, 22)

    # legs
    cv2.line(img, l_hip, (cx - 70, 940), TROUSERS, 46)
    cv2.line(img, r_hip, (cx + 70, 940), TROUSERS, 46)
    cv2.ellipse(img, (cx - 78, 952), (44, 18), 0, 0, 360, SHOE, -1)
    cv2.ellipse(img, (cx + 78, 952), (44, 18), 0, 0, 360, SHOE, -1)

    # torso + neck
    torso = np.array([l_sh, r_sh, r_hip, l_hip], dtype=np.int32)
    cv2.fillPoly(img, [torso], SHIRT)
    cv2.line(img, ((l_sh[0] + r_sh[0]) // 2, l_sh[1]), head, SKIN, 34)

    # club, drawn before the arms so the hands sit on top of the grip
    club_len = 300
    club_end = (
        int(hands[0] + club_len * unit[0]),
        int(hands[1] + club_len * unit[1]),
    )
    cv2.line(img, hands, club_end, CLUB, 7)
    cv2.circle(img, club_end, 13, (35, 35, 35), -1)

    # arms
    for sh, el in ((l_sh, l_el), (r_sh, r_el)):
        cv2.line(img, sh, el, SKIN, 30)
        cv2.line(img, el, hands, SKIN, 26)
    cv2.circle(img, hands, 20, (245, 245, 245), -1)

    # head
    cv2.circle(img, head, 52, SKIN, -1)

    # ball on the tee until it is struck
    if angle < 5:
        cv2.circle(img, (cx + 150, 946), 11, (250, 250, 250), -1)

    return img


def render(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for index in range(FRAMES):
        writer.write(draw_frame(index))
    writer.release()
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        nargs="?",
        default="synthetic_swing.mp4",
        help="where to write the clip (default: ./synthetic_swing.mp4)",
    )
    args = parser.parse_args()

    out = render(Path(args.destination))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB, {FRAMES} frames @ {FPS} fps)")
    print("Import it through the Video Lab to reproduce the README screenshots.")
