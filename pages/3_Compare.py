"""Compare — placeholder for Milestone 4."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.ui import coming_soon, page_setup  # noqa: E402

page_setup("Compare", icon="👥")
st.title("👥 Compare")

coming_soon(
    "Milestone 4 (reference comparison)",
    [
        "Reference library with golfer, camera angle, club, handedness, and licensing notes",
        "Synchronization by matched swing phase, not by timestamp or percentage",
        "Normalization by hip center and body scale before any overlay",
        "Side-by-side playback and ghost overlay",
        "Optional mirroring when handedness differs",
        "Comparison against a pro, your own best swing, or your average swing",
        "Difference summaries phrased as observations, never automatically as faults",
    ],
)

st.warning(
    "Reference footage will only ever be added from clips you own or that are "
    "explicitly licensed for this use. The app will not download professional video.",
    icon="⚖️",
)
