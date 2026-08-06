"""Video Lab — import swings and inspect them frame by frame (Milestone 1).

Design notes worth knowing before editing this file:

* Expensive work (import) is behind an explicit button, never triggered by an
  ordinary widget change. Streamlit reruns the whole script on every
  interaction, so anything slow must be gated or cached.
* The FrameReader is cached as a resource keyed on the file identity, so
  moving the slider does not reopen and re-decode the video from scratch.
* Previous/Next use on_click callbacks. Mutating session state inside a
  callback happens before the widgets are re-instantiated, which is what makes
  single-frame stepping exact instead of off-by-one.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.config import SUPPORTED_VIDEO_EXTENSIONS  # noqa: E402
from golf_lab.logging_config import get_logger  # noqa: E402
from golf_lab.models.video import (  # noqa: E402
    CameraView,
    Handedness,
    ShotShape,
    SwingContext,
)
from golf_lab.storage import swing_repository  # noqa: E402
from golf_lab.storage.file_repository import exports_dir  # noqa: E402
from golf_lab.storage.swing_repository import SwingImportError  # noqa: E402
from golf_lab.ui import (  # noqa: E402
    FRAME_INDEX_KEY,
    page_setup,
    status_badge,
    swing_selector,
)
from golf_lab.video.frame_reader import (  # noqa: E402
    FrameReader,
    FrameReadError,
    format_timestamp,
)

logger = get_logger(__name__)

page_setup("Video Lab", icon="🎬")
st.title("🎬 Video Lab")

CLUB_OPTIONS = [
    "Driver", "3 Wood", "5 Wood", "Hybrid",
    "4 Iron", "5 Iron", "6 Iron", "7 Iron", "8 Iron", "9 Iron",
    "Pitching Wedge", "Gap Wedge", "Sand Wedge", "Lob Wedge",
    "Putter", "Unknown",
]


@st.cache_resource(show_spinner=False)
def _open_reader(video_path: str, size: int, mtime: int) -> FrameReader:
    """Cached FrameReader.

    ``size`` and ``mtime`` are part of the cache key so that replacing the
    file on disk invalidates the cached reader.
    """
    return FrameReader(Path(video_path))


def _step_frame(delta: int, last_index: int) -> None:
    current = int(st.session_state.get(FRAME_INDEX_KEY, 0))
    st.session_state[FRAME_INDEX_KEY] = max(0, min(current + delta, last_index))


import_tab, inspect_tab = st.tabs(["Import a swing", "Inspect frames"])

# ---------------------------------------------------------------- Import
with import_tab:
    st.subheader("Import a swing video")
    st.caption(
        "Your original file is copied into the swing folder and then marked "
        "read-only. It is never edited, re-encoded, or overwritten."
    )

    upload = st.file_uploader(
        "Swing video",
        type=[ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS],
        accept_multiple_files=False,
        help="MP4 or MOV straight from your phone is fine.",
    )

    with st.form("swing_metadata_form", clear_on_submit=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            club = st.selectbox("Club", CLUB_OPTIONS, index=CLUB_OPTIONS.index("7 Iron"))
            handedness = st.radio(
                "Handedness",
                options=[Handedness.RIGHT, Handedness.LEFT],
                format_func=lambda h: h.value.capitalize(),
                horizontal=True,
            )
        with col_b:
            camera_view = st.selectbox(
                "Camera view",
                options=[
                    CameraView.FACE_ON,
                    CameraView.DOWN_THE_LINE,
                    CameraView.OTHER,
                    CameraView.UNKNOWN,
                ],
                format_func=lambda v: v.value.replace("_", " ").title(),
                help=(
                    "Later milestones only apply a metric when the camera view "
                    "can actually support it, so this matters."
                ),
            )
            shot_shape = st.selectbox(
                "Shot shape you observed",
                options=list(ShotShape),
                index=list(ShotShape).index(ShotShape.UNKNOWN),
                format_func=lambda s: s.value.replace("_", " ").title(),
            )
        with col_c:
            typical_miss = st.text_input("Typical miss", placeholder="e.g. pull hook")
            carry = st.number_input(
                "Carry (yards, optional)", min_value=0.0, max_value=450.0,
                value=0.0, step=5.0,
            )
        notes = st.text_area("Notes", placeholder="Range session, into the wind, ...")

        submitted = st.form_submit_button("Import swing", type="primary")

    if submitted:
        if upload is None:
            st.error("Choose a video file first.")
        else:
            context = SwingContext(
                club=club,
                camera_view=camera_view,
                handedness=handedness,
                shot_shape=shot_shape,
                typical_miss=typical_miss.strip(),
                carry_yards=carry if carry > 0 else None,
                notes=notes.strip(),
            )

            progress_bar = st.progress(0.0, text="Starting import…")

            def report(fraction: float, message: str) -> None:
                progress_bar.progress(min(1.0, fraction), text=message)

            temp_path = None
            try:
                suffix = Path(upload.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(upload.getbuffer())
                    temp_path = Path(handle.name)

                record = swing_repository.import_swing(
                    source_path=temp_path,
                    original_filename=upload.name,
                    context=context,
                    progress=report,
                )
            except SwingImportError as exc:
                progress_bar.empty()
                st.error(f"**Import failed.**\n\n{exc}")
                logger.exception("Import failed for %s", upload.name)
            except Exception as exc:  # noqa: BLE001 - surface, never swallow
                progress_bar.empty()
                st.error(f"**Unexpected error during import.**\n\n`{type(exc).__name__}: {exc}`")
                logger.exception("Unexpected import failure")
            else:
                progress_bar.empty()
                st.success(f"Imported as swing `{record.swing_id}`.")
                if record.status_detail:
                    st.warning(record.status_detail)
                st.session_state["selected_swing_id"] = record.swing_id
                st.session_state[FRAME_INDEX_KEY] = 0
                st.info("Open the **Inspect frames** tab to step through it.")
            finally:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        logger.warning("Could not remove temp upload %s", temp_path)

# --------------------------------------------------------------- Inspect
with inspect_tab:
    record = swing_selector("Swing to inspect")

    if record is None:
        st.info("No swings yet. Import one in the **Import a swing** tab.")
    else:
        meta = record.video
        st.subheader(record.original_filename)
        st.caption(
            f"{status_badge(record)} · swing id `{record.swing_id}` · "
            f"imported {record.imported_at.strftime('%Y-%m-%d %H:%M')}"
        )
        if record.status_detail:
            st.warning(record.status_detail)

        if record.timeline_is_approximate:
            st.warning(
                "**Variable frame rate — the preview timeline is approximate.**\n\n"
                "The frames in your original are not evenly spaced in time, so "
                "generating a browser-playable preview resampled them to a "
                "constant rate. Preview frame numbers and timestamps therefore "
                "do **not** map exactly back to the original file.\n\n"
                "- Fine: the pose overlay, joint positions, and what the swing "
                "looks like at a given preview frame.\n"
                "- Not yet reliable: tempo ratios, phase durations, and lining a "
                "preview frame up with an exact frame of the source.\n\n"
                "Tracked as GSL-1 in docs/KNOWN_ISSUES.md."
            )

        try:
            video_path = swing_repository.preview_or_original_path(record)
            stat = video_path.stat()
            reader = _open_reader(str(video_path), stat.st_size, int(stat.st_mtime))
        except (SwingImportError, FrameReadError) as exc:
            st.error(f"Could not open this swing's video.\n\n{exc}")
            st.stop()

        last_index = reader.last_index
        st.session_state.setdefault(FRAME_INDEX_KEY, 0)
        if st.session_state[FRAME_INDEX_KEY] > last_index:
            st.session_state[FRAME_INDEX_KEY] = last_index

        viewer, sidebar_col = st.columns([3, 2])

        with viewer:
            if last_index > 0:
                st.slider(
                    "Frame",
                    min_value=0,
                    max_value=last_index,
                    key=FRAME_INDEX_KEY,
                )
            else:
                st.caption("This clip has a single frame.")

            btn_prev, btn_next, btn_start, btn_end = st.columns(4)
            btn_prev.button(
                "◀ Previous frame", use_container_width=True,
                on_click=_step_frame, args=(-1, last_index),
                disabled=st.session_state[FRAME_INDEX_KEY] <= 0,
            )
            btn_next.button(
                "Next frame ▶", use_container_width=True,
                on_click=_step_frame, args=(1, last_index),
                disabled=st.session_state[FRAME_INDEX_KEY] >= last_index,
            )
            btn_start.button(
                "⏮ First", use_container_width=True,
                on_click=_step_frame, args=(-10**9, last_index),
            )
            btn_end.button(
                "Last ⏭", use_container_width=True,
                on_click=_step_frame, args=(10**9, last_index),
            )

            frame_index = int(st.session_state[FRAME_INDEX_KEY])
            try:
                frame_rgb = reader.read_frame_rgb(frame_index)
            except FrameReadError as exc:
                st.error(str(exc))
                st.stop()

            timestamp = reader.timestamp_for_frame(frame_index)
            approximate = record.timeline_is_approximate
            st.image(
                frame_rgb,
                caption=(
                    f"Preview frame {frame_index} of {last_index}  ·  "
                    f"t = {format_timestamp(timestamp)}  ·  {timestamp:.4f} s"
                    + ("  ·  preview timeline (approximate vs source)" if approximate else "")
                ),
                use_container_width=True,
            )

            info_a, info_b, info_c = st.columns(3)
            info_a.metric("Preview frame", f"{frame_index}")
            info_b.metric(
                "Preview timestamp",
                format_timestamp(timestamp),
                help=(
                    "Measured on the preview's constant-frame-rate timeline. "
                    "For this clip that is not the same as time in the original "
                    "file — see the warning above."
                )
                if approximate
                else None,
            )
            info_c.metric("Preview frame rate", f"{reader.fps:.2f} fps")

        with sidebar_col:
            st.markdown("#### Save this frame")
            save_format = st.radio(
                "Format", options=[".png", ".jpg"], horizontal=True,
                help="PNG is lossless and better for later annotation work.",
            )
            if st.button("Save frame to exports", use_container_width=True):
                destination = (
                    exports_dir(record.swing_id)
                    / f"frame_{frame_index:06d}{save_format}"
                )
                try:
                    saved = reader.save_frame(frame_index, destination)
                except FrameReadError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Saved `{saved.name}`")
                    st.caption(str(saved))
                    st.download_button(
                        "Download it",
                        data=saved.read_bytes(),
                        file_name=saved.name,
                        use_container_width=True,
                    )

            st.markdown("#### Playback")
            preview = swing_repository.resolve_path(record, record.preview_relpath)
            if preview:
                st.video(str(preview))
                st.caption(
                    "Preview is downscaled, upright, and silent. The original "
                    "keeps its full resolution and audio for export."
                )
            else:
                st.caption("No preview available for this swing.")

            st.markdown("#### Shot context")
            st.write(
                {
                    "Club": record.context.club,
                    "Camera view": record.context.camera_view.value.replace("_", " "),
                    "Handedness": record.context.handedness.value,
                    "Shot shape": record.context.shot_shape.value,
                    "Typical miss": record.context.typical_miss or "—",
                    "Carry (yd)": record.context.carry_yards or "—",
                    "Notes": record.context.notes or "—",
                }
            )

            st.markdown("#### Original file")
            st.caption("Untouched. Used only for final export.")
            st.write(
                {
                    "Display size": f"{meta.width} × {meta.height}",
                    "Stored size": f"{meta.coded_width} × {meta.coded_height}",
                    "Rotation applied": f"{meta.rotation_degrees}°",
                    "Frame rate (average)": f"{meta.fps:.3f} fps",
                    "Frame rate (nominal)": (
                        f"{meta.r_frame_rate:.3f} fps" if meta.r_frame_rate else "—"
                    ),
                    "Frames": (
                        f"{meta.frame_count}"
                        + (" (estimated)" if meta.frame_count_is_estimated else "")
                    ),
                    "Duration": f"{meta.duration_seconds:.3f} s",
                    "Variable frame rate": "yes" if meta.is_variable_frame_rate else "no",
                    "Video codec": meta.codec_name or "unknown",
                    "Audio": meta.audio_codec_name or "none",
                    "Probed with": meta.probe_source,
                }
            )

            st.markdown("#### Preview (what you are stepping through)")
            preview_meta = record.preview_video
            st.write(
                {
                    "Display size": (
                        f"{preview_meta.width} × {preview_meta.height}"
                        if preview_meta
                        else f"{reader.width} × {reader.height}"
                    ),
                    "Frame rate": f"{reader.fps:.3f} fps",
                    "Frames": f"{last_index + 1}",
                    "Duration": (
                        f"{preview_meta.duration_seconds:.3f} s" if preview_meta else "—"
                    ),
                    "Codec": "h264 (browser-safe, silent)",
                }
            )
            st.caption(
                "Every frame number, slider position, and timestamp in this app "
                "refers to the preview."
            )

            if meta.fps < 60:
                st.info(
                    f"This clip is {meta.fps:.0f} fps. Body analysis will still work, "
                    "but impact and ball tracking are limited at this frame rate. "
                    "See docs/RECORDING_GUIDE.md."
                )
