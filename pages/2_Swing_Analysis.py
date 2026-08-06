"""Swing Analysis — placeholder for Milestones 2 and 3."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.ui import coming_soon, page_setup  # noqa: E402

page_setup("Swing Analysis", icon="🦴")
st.title("🦴 Swing Analysis")

coming_soon(
    "Milestone 2 (pose overlay) and Milestone 3 (phases and qualitative metrics)",
    [
        "Local MediaPipe Pose Landmarker inference, cached per swing",
        "Raw and smoothed landmarks stored separately, with failed frames marked",
        "Skeleton overlay on the preview, with a per-frame confidence readout",
        "Suggested P1 address, P4 top, P7 impact, and P9 finish — all editable",
        "Manual phase marking that always overrides the automatic suggestion",
        "Body-scale-normalized measurements (head movement, hip sway, hand height)",
        "Qualitative classifications with low/medium/high confidence labels",
        "Metrics gated by camera view, so nothing is reported that 2D cannot support",
    ],
)
