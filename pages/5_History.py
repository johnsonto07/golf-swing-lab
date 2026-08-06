"""History — a working list of imported swings (full trends land in Milestone 8)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.storage import swing_repository  # noqa: E402
from golf_lab.storage.file_repository import swing_dir  # noqa: E402
from golf_lab.ui import page_setup, status_badge  # noqa: E402

page_setup("History", icon="📚")
st.title("📚 History")

records = swing_repository.list_records()

if not records:
    st.info("No swings imported yet. Start in the **Video Lab**.")
else:
    st.caption(f"{len(records)} swing(s) stored locally under `data/swings/`.")

    clubs = sorted({r.context.club for r in records})
    chosen_clubs = st.multiselect("Filter by club", clubs, default=clubs)
    visible = [r for r in records if r.context.club in chosen_clubs]

    for record in visible:
        with st.container(border=True):
            cols = st.columns([1, 3])
            thumbnail = swing_repository.resolve_path(record, record.thumbnail_relpath)
            with cols[0]:
                if thumbnail:
                    st.image(str(thumbnail), use_container_width=True)
                else:
                    st.caption("No thumbnail")
            with cols[1]:
                st.markdown(f"**{record.original_filename}**")
                st.caption(
                    f"{status_badge(record)} · "
                    f"{record.imported_at.strftime('%Y-%m-%d %H:%M')} · "
                    f"{record.context.club} · "
                    f"{record.context.camera_view.value.replace('_', ' ')} · "
                    f"{record.video.fps:.0f} fps · "
                    f"{record.video.duration_seconds:.1f} s"
                )
                if record.context.notes:
                    st.write(record.context.notes)
                st.code(str(swing_dir(record.swing_id)), language=None)

st.divider()
st.markdown(
    "**Milestone 8** will add tags, trend summaries over time, comparison against "
    "your personal best, export management, and analysis-version tracking so old "
    "results are flagged when the analysis code changes."
)
st.caption(
    "Deleting swings is intentionally not available in the UI yet — removing your "
    "own footage should be a deliberate action you take in the file explorer."
)
