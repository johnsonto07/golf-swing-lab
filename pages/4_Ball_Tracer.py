"""Ball Tracer — placeholder for Milestones 5 and 6."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.ui import coming_soon, page_setup  # noqa: E402

page_setup("Ball Tracer", icon="🎯")
st.title("🎯 Ball Tracer")

coming_soon(
    "Milestone 5 (manual tracer) and Milestone 6 (assisted tracking)",
    [
        "Confirm the impact frame, then click the ball center at impact",
        "Add, move, or delete ball points on any frame where the ball is visible",
        "Shot-shape and height presets that seed an editable spline",
        "Manual control of launch direction, apex, curvature, and endpoint",
        "Three visually distinct layers: confirmed, tracked, and estimated",
        "Growing tracer preview that never appears before the confirmed impact frame",
        "Final render at original quality with the original audio remuxed back in",
    ],
)

st.info(
    "The tracer draws a smooth screen-space curve that you shape by hand. It is a "
    "visualization of the flight you saw — not a measured or physically simulated "
    "ball flight, and not comparable to launch-monitor data.",
    icon="ℹ️",
)
