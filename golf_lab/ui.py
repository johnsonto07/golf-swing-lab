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


@st.cache_resource(show_spinner=False)
def load_timeline(swing_id: str, fingerprint: str):
    """Measured timeline for a swing, cached per media fingerprint."""
    from golf_lab.storage import timeline_repository

    return timeline_repository.load_timeline(swing_id)


def effective_status_detail(record: SwingRecord, timeline) -> str:
    """The record's status text, suppressed when measurement contradicts it.

    Swings imported before timing was measured carry a stored `status_detail`
    claiming variable frame rate, written by the old metadata-comparison
    classifier. That text is wrong for clips whose measured spacing is
    constant, and it is persisted data — re-running the app cannot rewrite
    every user's `metadata.json`, so the display defers to the measurement
    instead of the stored sentence.
    """
    detail = record.status_detail or ""
    if not detail or timeline is None:
        return detail

    from golf_lab.video.timeline import RateClassification

    claims_vfr = "variable-frame-rate" in detail.lower() or "variable frame rate" in detail.lower()
    if claims_vfr and timeline.rate_classification is RateClassification.CONSTANT:
        return ""
    return detail


def timing_notices(record: SwingRecord, timeline) -> list[tuple[str, str]]:
    """Warnings to show for a swing's timing, as ``(level, message)`` pairs.

    Two things that used to be conflated are now reported separately, because
    conflating them produced a false alarm: **inconsistent container metadata**
    is a property of the file's headers, while **variable frame rate** is a
    property of the decoded frames. The clip that prompted this has the first
    and not the second — 438 frames decode at a constant 25.000 fps while its
    container advertises 484 at 22.873.
    """
    from golf_lab.video.timeline import RateClassification, TimelineConfidence

    notices: list[tuple[str, str]] = []

    if timeline is None:
        notices.append((
            "warning",
            "**Frame timing has not been measured for this swing.** It was "
            "imported before timings were measured, or the measurement failed. "
            "Frame stepping and the pose overlay work normally, but durations "
            "and tempo are refused rather than estimated. Re-import to measure.",
        ))
        return notices

    if timeline.rate_classification is RateClassification.VARIABLE:
        notices.append((
            "warning",
            f"**This clip is genuinely variable frame rate.** Measured from its "
            f"own frame timestamps: {timeline.frame_count} frames spanning "
            f"{timeline.duration_seconds:.2f}s, with spacing that varies beyond "
            "tolerance. Timestamps shown are measured, so durations remain "
            "trustworthy — but frames are not evenly spaced in time.",
        ))
    elif timeline.confidence is TimelineConfidence.NOMINAL:
        notices.append((
            "warning",
            "**No frame timestamps could be read from this video.** Timing falls "
            "back to a nominal frame rate, so durations and tempo are refused "
            "rather than estimated from an assumption.",
        ))
    elif timeline.confidence is TimelineConfidence.DEGRADED:
        notices.append((
            "info",
            "Some frames in this video had no readable timestamp and were "
            "interpolated between their neighbours. Durations are available but "
            "reported as low confidence.",
        ))

    # Reported as what it is: bad headers, not a bad video.
    if timeline.container_metadata_is_inconsistent:
        notices.append((
            "info",
            f"This file's container metadata is inconsistent — it claims "
            f"{timeline.container_frame_count} frames, but "
            f"{timeline.frame_count} actually decode. The measured values are "
            "used throughout. This does **not** mean the video is variable "
            "frame rate; its measured spacing is "
            f"**{timeline.rate_classification.label}**.",
        ))
    return notices


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
