"""The MediaPipe Pose landmark topology, as plain data.

Deliberately free of any ``mediapipe`` import. Overlay drawing, smoothing, and
every test that reasons about joints can then work without the heavy
dependency (or the model file) being present — which is also what lets the
test suite run on a machine that has never downloaded a model.

Indices are MediaPipe's 33-point BlazePose topology and are part of the
on-disk contract: ``pose_raw.npz`` stores landmarks in this order, so the
numbering must not be rearranged without bumping ``ANALYSIS_VERSION``.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

NUM_LANDMARKS = 33

# --- Index constants ----------------------------------------------------
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LANDMARK_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# --- Skeleton edges -----------------------------------------------------
# Grouped by body region so the overlay can draw them in distinct colours;
# for a golf swing the arms and torso carry nearly all the signal, and the
# face landmarks are noise you mostly want to be able to turn down.
FACE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (NOSE, LEFT_EYE_INNER),
    (LEFT_EYE_INNER, LEFT_EYE),
    (LEFT_EYE, LEFT_EYE_OUTER),
    (LEFT_EYE_OUTER, LEFT_EAR),
    (NOSE, RIGHT_EYE_INNER),
    (RIGHT_EYE_INNER, RIGHT_EYE),
    (RIGHT_EYE, RIGHT_EYE_OUTER),
    (RIGHT_EYE_OUTER, RIGHT_EAR),
    (MOUTH_LEFT, MOUTH_RIGHT),
)

TORSO_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
)

ARM_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_PINKY),
    (LEFT_WRIST, LEFT_INDEX),
    (LEFT_WRIST, LEFT_THUMB),
    (LEFT_PINKY, LEFT_INDEX),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_WRIST, RIGHT_PINKY),
    (RIGHT_WRIST, RIGHT_INDEX),
    (RIGHT_WRIST, RIGHT_THUMB),
    (RIGHT_PINKY, RIGHT_INDEX),
)

LEG_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
)

POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    FACE_CONNECTIONS + TORSO_CONNECTIONS + ARM_CONNECTIONS + LEG_CONNECTIONS
)

# Joints that actually matter for swing measurement. Used for the headline
# confidence number so a clip is not called "good" because it nailed 10 face
# landmarks while losing both wrists.
KEY_SWING_LANDMARKS: Tuple[int, ...] = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)

FACE_LANDMARKS: FrozenSet[int] = frozenset(range(NOSE, LEFT_SHOULDER))


def connection_list() -> List[Tuple[int, int]]:
    """All skeleton edges as a mutable list."""
    return list(POSE_CONNECTIONS)


def landmark_name(index: int) -> str:
    """Human-readable name for a landmark index."""
    if 0 <= index < NUM_LANDMARKS:
        return LANDMARK_NAMES[index]
    raise IndexError(f"Landmark index {index} is outside 0..{NUM_LANDMARKS - 1}")
