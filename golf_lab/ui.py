"""Small shared helpers for the Streamlit pages.

Keeps the pages thin: page files should describe layout and user intent, not
contain pipeline logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Streamlit runs page scripts directly, so guarantee the repo root is
# importable regardless of how the app was launched.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.config import APP_VERSION, ensure_data_dirs  # noqa: E402
from golf_lab.models.video import SwingRecord, SwingStatus  # noqa: E402
from golf_lab.storage import swing_repository  # noqa: E402

SELECTED_SWING_KEY = "selected_swing_id"
FRAME_INDEX_KEY = "frame_index"

STATUS_BADGES = {
    SwingStatus.NOT_PROCESSED: ("Not processed", "⚪"),
    SwingStatus.PROCESSING: ("Processing", "🔵"),
    SwingStatus.READY: ("Ready", "🟢"),
    SwingStatus.NEEDS_REVIEW: ("Needs review", "🟠"),
    SwingStatus.LOW_CONFIDENCE: ("Low confidence", "🟡"),
    SwingStatus.FAILED: ("Failed", "🔴"),
}


def page_setup(title: str, icon: str = "🏌️") -> None:
    """Standard page configuration + sidebar chrome."""
    st.set_page_config(page_title=f"{title} · Golf Swing Lab", page_icon=icon, layout="wide")
    ensure_data_dirs()
    with st.sidebar:
        st.caption(f"Golf Swing Lab v{APP_VERSION} · fully local")


def status_badge(record: SwingRecord) -> str:
    label, icon = STATUS_BADGES.get(record.status, ("Unknown", "⚪"))
    return f"{icon} {label}"


def swing_label(record: SwingRecord) -> str:
    view = record.context.camera_view.value.replace("_", " ")
    return (
        f"{record.imported_at.strftime('%Y-%m-%d %H:%M')} · "
        f"{record.context.club} · {view} · {record.original_filename}"
    )


def swing_selector(label: str = "Swing") -> Optional[SwingRecord]:
    """Sidebar selector that persists the chosen swing across pages."""
    records = swing_repository.list_records()
    if not records:
        return None

    ids = [r.swing_id for r in records]
    current = st.session_state.get(SELECTED_SWING_KEY)
    index = ids.index(current) if current in ids else 0

    chosen_id = st.sidebar.selectbox(
        label,
        options=ids,
        index=index,
        format_func=lambda sid: swing_label(next(r for r in records if r.swing_id == sid)),
    )
    if chosen_id != st.session_state.get(SELECTED_SWING_KEY):
        # Selecting a different swing must reset the frame cursor, otherwise
        # the slider can point past the end of the new clip.
        st.session_state[FRAME_INDEX_KEY] = 0
    st.session_state[SELECTED_SWING_KEY] = chosen_id
    return next(r for r in records if r.swing_id == chosen_id)


def coming_soon(milestone: str, bullets: list[str]) -> None:
    """Consistent placeholder for pages whose milestone isn't built yet."""
    st.info(
        f"This page is part of **{milestone}** and is not implemented yet. "
        "It is listed here so the navigation reflects the full product shape."
    )
    st.markdown("**Planned for this page:**")
    for bullet in bullets:
        st.markdown(f"- {bullet}")
    st.caption("See docs/ROADMAP.md for the full milestone plan.")
