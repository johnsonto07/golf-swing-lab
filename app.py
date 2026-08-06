"""Golf Swing Lab — local Streamlit entry point.

Run with:   streamlit run app.py

Everything in this application runs on this machine. No video, image, or
measurement leaves the computer in Milestone 0/1; there is no cloud code path
at all yet, and no API key is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.config import APP_VERSION  # noqa: E402
from golf_lab.logging_config import get_logger  # noqa: E402
from golf_lab.storage import swing_repository  # noqa: E402
from golf_lab.ui import page_setup, status_badge, swing_label  # noqa: E402
from golf_lab.video.ffmpeg import find_ffmpeg  # noqa: E402

logger = get_logger(__name__)

page_setup("Home")

st.title("🏌️ Golf Swing Lab")
st.caption(f"Local swing-analysis workspace · v{APP_VERSION}")

ffmpeg_tools = find_ffmpeg(required=False)
if ffmpeg_tools is None:
    st.error(
        "**FFmpeg was not found on your PATH.** Video import, preview generation, "
        "and export all depend on it.\n\n"
        "Install it on Windows with `winget install Gyan.FFmpeg`, then close and "
        "reopen your terminal and restart this app. "
        "See the Settings page for full diagnostics."
    )

left, right = st.columns([3, 2])

with left:
    st.subheader("What works today")
    st.markdown(
        """
**Milestone 1 — Video Lab** is implemented:

- Import MP4 / MOV swing videos with club, camera view, and handedness
- The original file is copied once and then left untouched, permanently
- Correct handling of phone rotation metadata
- A browser-playable, upright preview for fast scrubbing
- Frame-accurate slider with single-frame previous / next stepping
- Frame number and timestamp that agree with the decoded frame
- Save any frame as a PNG or JPEG
- Reopen a saved swing and get its metadata back

Everything else in the navigation is a planned milestone and says so.
"""
    )
    st.subheader("Product stance")
    st.markdown(
        """
This is a **semi-automatic** tool, not a launch monitor. It works from ordinary
2D phone video, so it will suggest phases, positions, and observations, and you
correct them. Later milestones deliberately report qualitative findings with a
confidence label rather than false-precision numbers.
"""
    )

with right:
    st.subheader("Saved swings")
    records = swing_repository.list_records()
    if not records:
        st.info("No swings imported yet. Open **Video Lab** to import your first video.")
    else:
        st.metric("Swings on disk", len(records))
        for record in records[:8]:
            with st.container(border=True):
                st.markdown(f"**{swing_label(record)}**")
                st.caption(status_badge(record))
                if record.status_detail:
                    st.caption(record.status_detail)

st.divider()
st.caption(
    "Privacy: all processing is local. Optional cloud coaching is a later milestone, "
    "is off by default, and will always require explicit opt-in before anything is sent."
)
