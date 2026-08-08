"""Swing Analysis — pose estimation and skeleton overlay (Milestone 2).

Follows the same interaction rules as the Video Lab: expensive work only ever
happens behind an explicit button, the FrameReader and the pose data are
cached resources so scrubbing does not recompute anything, and Previous/Next
use on_click callbacks so single-frame stepping is exact.

Two things this page deliberately refuses to do:

* Download a model without being asked. The button is the only path.
* Present a cached analysis as current when the video or the analysis version
  has changed underneath it. Staleness is shown, loudly, before the numbers.

Phase detection and qualitative metrics are Milestone 3 and are not here yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.config import MODELS_DIR  # noqa: E402
from golf_lab.logging_config import get_logger  # noqa: E402
from golf_lab.pose.inference import (  # noqa: E402
    PoseInferenceCancelled,
    estimate_pose_sequence,
)
from golf_lab.pose.landmarks import KEY_SWING_LANDMARKS, landmark_name  # noqa: E402
from golf_lab.pose.mediapipe_backend import (  # noqa: E402
    MediaPipePoseBackend,
    mediapipe_available,
    mediapipe_version,
)
from golf_lab.pose.model_manager import (  # noqa: E402
    DEFAULT_MODEL_KEY,
    PoseModelError,
    available_specs,
    download_model,
    get_spec,
    is_downloaded,
    manifest_entry,
    model_path,
    verify_model,
)
from golf_lab.pose.overlay import OverlayStyle, draw_sequence_frame  # noqa: E402
from golf_lab.pose.backend import PoseBackendError  # noqa: E402
from golf_lab.pose.smoothing import SmoothingSettings, smooth_sequence  # noqa: E402
from golf_lab.storage import (  # noqa: E402
    analysis_repository,
    pose_repository,
    swing_repository,
)
from golf_lab.storage.file_repository import exports_dir  # noqa: E402
from golf_lab.storage.swing_repository import SwingImportError  # noqa: E402
from golf_lab.swing import metric_registry  # noqa: E402
from golf_lab.swing.geometry_detector import default_detector  # noqa: E402
from golf_lab.swing.phases import PHASE_ORDER, SwingPhase  # noqa: E402
from golf_lab.swing.results import ResultStatus  # noqa: E402
from golf_lab.swing.source_timing import all_source_timings  # noqa: E402
from golf_lab.ui import (  # noqa: E402
    FRAME_INDEX_KEY,
    effective_status_detail,
    load_timeline,
    page_setup,
    swing_selector,
    timing_notices,
)
from golf_lab.video.frame_reader import (  # noqa: E402
    FrameReader,
    FrameReadError,
    format_timestamp,
)

logger = get_logger(__name__)

page_setup("Swing Analysis", icon="🦴")
st.title("🦴 Swing Analysis")

CANCEL_KEY = "pose_cancel_requested"
MODEL_CHOICE_KEY = "pose_model_choice"


@st.cache_resource(show_spinner=False)
def _open_reader(
    video_path: str, size: int, mtime: int, swing_id: str = ""
) -> FrameReader:
    """Cached reader carrying the measured timeline, keyed on file identity."""
    from golf_lab.storage import timeline_repository

    timeline = timeline_repository.load_timeline(swing_id) if swing_id else None
    return FrameReader(Path(video_path), timeline=timeline)


@st.cache_resource(show_spinner=False)
def _load_pose(swing_id: str, kind: str, fingerprint: str):
    """Cached pose data. ``fingerprint`` in the key invalidates on recompute."""
    if kind == "raw":
        return pose_repository.load_raw(swing_id)
    return pose_repository.load_smoothed(swing_id)


def _step_frame(delta: int, last_index: int) -> None:
    current = int(st.session_state.get(FRAME_INDEX_KEY, 0))
    st.session_state[FRAME_INDEX_KEY] = max(0, min(current + delta, last_index))


def _request_cancel() -> None:
    st.session_state[CANCEL_KEY] = True


def _set_frame(frame_index: int) -> None:
    """Jump the viewer to a specific preview frame (used by the phase list)."""
    st.session_state[FRAME_INDEX_KEY] = int(frame_index)


record = swing_selector("Swing to analyse")

if record is None:
    st.info("No swings yet. Import one in the **Video Lab** first.")
    st.stop()

st.subheader(record.original_filename)
st.caption(f"swing id `{record.swing_id}`")

_timeline = load_timeline(record.swing_id, record.preview_relpath or "")
for _level, _message in timing_notices(record, _timeline):
    (st.warning if _level == "warning" else st.info)(_message)

try:
    video_path = swing_repository.preview_or_original_path(record)
except SwingImportError as exc:
    st.error(f"Could not open this swing's video.\n\n{exc}")
    st.stop()

analysis_tab, viewer_tab, phases_tab, model_tab = st.tabs(
    ["Run analysis", "Skeleton overlay", "Phases", "Model"]
)

# ------------------------------------------------------------------ Model
with model_tab:
    st.markdown("#### Pose model")
    st.caption(
        "Downloading a model is the only step in this app that uses the "
        "internet. Everything after it — inference, overlay, storage — runs "
        "entirely on your machine."
    )

    if not mediapipe_available():
        st.error(
            "**MediaPipe is not installed**, so pose estimation is unavailable.\n\n"
            "Install it with:\n\n```\npip install -e \".[pose]\"\n```\n\n"
            "Everything else in the app works without it."
        )

    specs = available_specs()
    keys = [spec.key for spec in specs]
    chosen_key = st.radio(
        "Model",
        options=keys,
        index=keys.index(st.session_state.get(MODEL_CHOICE_KEY, DEFAULT_MODEL_KEY)),
        format_func=lambda key: get_spec(key).display_name,
        key=MODEL_CHOICE_KEY,
    )
    spec = get_spec(chosen_key)

    st.caption(spec.description)
    st.write(
        {
            "Source": spec.source,
            "Licence": spec.license_name,
            "Download size": f"~{spec.approx_megabytes:.0f} MB",
            "Stored at": str(model_path(spec, MODELS_DIR)),
        }
    )
    st.caption(f"Licence text: {spec.license_url}")

    if is_downloaded(spec, MODELS_DIR):
        problem = verify_model(spec, MODELS_DIR)
        if problem:
            st.warning(problem)
        else:
            entry = manifest_entry(spec, MODELS_DIR) or {}
            st.success(f"{spec.display_name} is downloaded and intact.")
            st.caption(
                f"sha256 `{entry.get('sha256', '?')[:32]}…` · "
                "recorded on first download, then verified on every load."
            )
        if st.button("Re-download this model", use_container_width=True):
            st.session_state["_force_download"] = True
    else:
        st.info(f"{spec.display_name} has not been downloaded yet.")

    wants_download = st.session_state.pop("_force_download", False) or (
        not is_downloaded(spec, MODELS_DIR)
        and st.button(
            f"Download {spec.display_name} (~{spec.approx_megabytes:.0f} MB)",
            type="primary",
            use_container_width=True,
        )
    )

    if wants_download:
        progress_bar = st.progress(0.0, text="Starting download…")
        try:
            download_model(
                spec,
                models_dir=MODELS_DIR,
                progress=lambda fraction, message: progress_bar.progress(
                    min(1.0, max(0.0, fraction)), text=message
                ),
                force=True,
            )
        except PoseModelError as exc:
            progress_bar.empty()
            st.error(f"**Download failed.**\n\n{exc}")
        else:
            progress_bar.empty()
            st.success(f"{spec.display_name} is ready.")
            st.rerun()

# ----------------------------------------------------------- Run analysis
with analysis_tab:
    spec = get_spec(st.session_state.get(MODEL_CHOICE_KEY, DEFAULT_MODEL_KEY))
    existing_info = pose_repository.load_info(record.swing_id)
    reasons = pose_repository.staleness_reasons(existing_info, video_path, record)

    if existing_info is not None and reasons:
        st.warning(
            "**This swing's saved pose analysis is out of date.**\n\n"
            + "\n".join(f"- {reason}" for reason in reasons)
            + "\n\nRe-run it before trusting the overlay."
        )
    elif existing_info is not None:
        st.success("A current pose analysis exists for this swing.")

    if existing_info is not None:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Frames detected", f"{existing_info.detected_count}/{existing_info.frame_count}")
        col_b.metric("Detection rate", f"{existing_info.detection_rate * 100:.1f}%")
        col_c.metric("Mean confidence", f"{existing_info.mean_confidence:.2f}")
        col_d.metric("Longest gap", f"{existing_info.longest_gap_frames} frames")
        st.caption(
            f"Model `{existing_info.model_key}` · {existing_info.backend} on "
            f"{existing_info.device} · MediaPipe {existing_info.mediapipe_version} · "
            f"smoothing {existing_info.smoothing} · "
            f"took {existing_info.elapsed_seconds:.1f}s · "
            f"computed {existing_info.created_at[:19].replace('T', ' ')} UTC"
        )

    st.divider()
    st.markdown("#### Run pose estimation")

    with st.expander("Settings", expanded=False):
        smoothing_method = st.selectbox(
            "Smoothing",
            options=["savgol", "moving_average", "none"],
            help=(
                "Smoothing is stored separately; the raw landmarks are always "
                "kept so any measurement can be traced back to what was "
                "actually observed."
            ),
        )
        window_length = st.slider("Smoothing window (frames)", 3, 21, 7, step=2)
        use_gpu = st.checkbox(
            "Try GPU delegate",
            value=False,
            help="CPU is the default and is what has been tested. Falls back "
                 "with a clear error if your setup cannot provide a GPU delegate.",
        )
        detection_confidence = st.slider(
            "Minimum detection confidence", 0.1, 0.9, 0.5, step=0.05
        )

    model_ready = is_downloaded(spec, MODELS_DIR)
    if not model_ready:
        st.warning(
            f"The **{spec.display_name}** model is not downloaded yet. "
            "Open the **Model** tab to get it — it is a one-time download."
        )
    if not mediapipe_available():
        st.error(
            'MediaPipe is not installed. Run `pip install -e ".[pose]"` and '
            "restart the app."
        )

    run_col, cancel_col = st.columns([3, 1])
    start = run_col.button(
        "Analyse this swing",
        type="primary",
        use_container_width=True,
        disabled=not (model_ready and mediapipe_available()),
    )
    cancel_col.button(
        "Cancel", use_container_width=True, on_click=_request_cancel
    )

    if start:
        st.session_state[CANCEL_KEY] = False
        progress_bar = st.progress(0.0, text="Preparing…")
        status = st.empty()
        backend = None

        try:
            backend = MediaPipePoseBackend(
                model_path=model_path(spec, MODELS_DIR),
                use_gpu=use_gpu,
                min_pose_detection_confidence=detection_confidence,
            )
            raw = estimate_pose_sequence(
                video_path,
                backend,
                progress=lambda fraction, message: progress_bar.progress(
                    min(1.0, max(0.0, fraction)), text=message
                ),
                should_cancel=lambda: bool(st.session_state.get(CANCEL_KEY)),
            )

            status.info("Smoothing…")
            settings = SmoothingSettings(
                method=smoothing_method, window_length=window_length
            )
            smoothed = smooth_sequence(raw, settings)

            info = pose_repository.build_info(
                swing_id=record.swing_id,
                raw=raw,
                smoothed=smoothed,
                video_path=video_path,
                model_key=spec.key,
                model_filename=spec.filename,
                model_sha256=(manifest_entry(spec, MODELS_DIR) or {}).get("sha256", ""),
                backend=backend.name,
                device=backend.device,
                mediapipe_version=mediapipe_version() or "unknown",
            )
            pose_repository.save_pose_analysis(record.swing_id, raw, smoothed, info)

        except PoseInferenceCancelled as exc:
            progress_bar.empty()
            status.empty()
            st.warning(
                f"{exc} Nothing was saved — the partial result is discarded "
                "rather than stored as if it were complete."
            )
        except (PoseBackendError, PoseModelError) as exc:
            progress_bar.empty()
            status.empty()
            st.error(f"**Pose estimation failed.**\n\n{exc}")
            logger.exception("Pose estimation failed for %s", record.swing_id)
        except Exception as exc:  # noqa: BLE001 - surface, never swallow
            progress_bar.empty()
            status.empty()
            st.error(f"**Unexpected error.**\n\n`{type(exc).__name__}: {exc}`")
            logger.exception("Unexpected pose failure for %s", record.swing_id)
        else:
            progress_bar.empty()
            status.empty()
            _load_pose.clear()
            st.success(
                f"Analysed {info.detected_count} of {info.frame_count} frames "
                f"in {info.elapsed_seconds:.1f}s."
            )
            if info.detection_rate < 0.8:
                st.warning(
                    f"Only {info.detection_rate * 100:.0f}% of frames produced a "
                    "pose. The frames that failed are recorded as failed, not "
                    "filled in. Check framing and lighting — "
                    "docs/RECORDING_GUIDE.md covers what helps."
                )
            st.info("Open the **Skeleton overlay** tab to step through it.")
        finally:
            if backend is not None:
                backend.close()
            st.session_state[CANCEL_KEY] = False

    if pose_repository.has_pose_analysis(record.swing_id):
        st.divider()
        if st.button("Delete this swing's pose analysis"):
            pose_repository.delete_pose_analysis(record.swing_id)
            _load_pose.clear()
            st.success("Pose analysis deleted. Your video is untouched.")
            st.rerun()

# --------------------------------------------------------------- Overlay
with viewer_tab:
    info = pose_repository.load_info(record.swing_id)
    if info is None:
        st.info(
            "No pose analysis for this swing yet. Run it in the "
            "**Run analysis** tab."
        )
        st.stop()

    stale_reasons = pose_repository.staleness_reasons(info, video_path, record)
    if stale_reasons:
        st.warning(
            "**Showing an out-of-date analysis.**\n\n"
            + "\n".join(f"- {reason}" for reason in stale_reasons)
        )

    try:
        stat = video_path.stat()
        reader = _open_reader(
            str(video_path), stat.st_size, int(stat.st_mtime), record.swing_id
        )
    except FrameReadError as exc:
        st.error(str(exc))
        st.stop()

    controls, options = st.columns([3, 2])

    with options:
        st.markdown("#### Overlay")
        source_choice = st.radio(
            "Landmark source",
            options=["smoothed", "raw"],
            horizontal=True,
            help=(
                "Raw is exactly what the model produced. Smoothed is filtered "
                "for display and velocity work. Both are stored; raw is never "
                "overwritten."
            ),
        )
        show_overlay = st.checkbox("Draw skeleton", value=True)
        draw_face = st.checkbox("Include face landmarks", value=False)
        draw_legs = st.checkbox("Include legs", value=True)
        thickness = st.slider("Line thickness", 1, 6, 2)

    sequence = _load_pose(record.swing_id, source_choice, info.created_at)
    if sequence is None:
        st.error(
            "The stored pose data for this swing could not be read. "
            "Re-run the analysis to rebuild it."
        )
        st.stop()

    last_index = min(reader.last_index, sequence.frame_count - 1)
    st.session_state.setdefault(FRAME_INDEX_KEY, 0)
    if st.session_state[FRAME_INDEX_KEY] > last_index:
        st.session_state[FRAME_INDEX_KEY] = last_index

    with controls:
        if last_index > 0:
            st.slider("Frame", 0, last_index, key=FRAME_INDEX_KEY)

        prev_col, next_col, first_col, last_col = st.columns(4)
        prev_col.button(
            "◀ Previous frame", use_container_width=True,
            on_click=_step_frame, args=(-1, last_index),
            disabled=st.session_state[FRAME_INDEX_KEY] <= 0,
        )
        next_col.button(
            "Next frame ▶", use_container_width=True,
            on_click=_step_frame, args=(1, last_index),
            disabled=st.session_state[FRAME_INDEX_KEY] >= last_index,
        )
        first_col.button(
            "⏮ First", use_container_width=True,
            on_click=_step_frame, args=(-10**9, last_index),
        )
        last_col.button(
            "Last ⏭", use_container_width=True,
            on_click=_step_frame, args=(10**9, last_index),
        )

        frame_index = int(st.session_state[FRAME_INDEX_KEY])
        try:
            frame_bgr = reader.read_frame(frame_index)
        except FrameReadError as exc:
            st.error(str(exc))
            st.stop()

        drawn = False
        image_bgr = frame_bgr
        if show_overlay:
            style = OverlayStyle(
                line_thickness=thickness,
                draw_face=draw_face,
                draw_legs=draw_legs,
            )
            image_bgr, drawn = draw_sequence_frame(
                frame_bgr, sequence, frame_index, style
            )

        import cv2

        timestamp = reader.timestamp_for_frame(frame_index)
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption=(
                f"Frame {frame_index} of {last_index}  ·  "
                f"t = {format_timestamp(timestamp)}  ·  "
                + ("pose detected" if sequence.detected[frame_index] else "no pose detected")
            ),
            use_container_width=True,
        )

        if show_overlay and not drawn:
            st.warning(
                f"No pose was detected on frame {frame_index}. Nothing is drawn "
                "here — the landmarks were not estimated, and this app does not "
                "fill in the gap."
            )

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Frame", str(frame_index))
        metric_b.metric(
            "Frame confidence", f"{sequence.confidence_for_frame(frame_index):.2f}"
        )
        metric_c.metric("Timestamp", format_timestamp(timestamp))

    with options:
        st.markdown("#### This frame")
        if sequence.detected[frame_index]:
            rows = {
                landmark_name(index): f"{sequence.visibility[frame_index, index]:.2f}"
                for index in KEY_SWING_LANDMARKS
            }
            st.caption("Per-joint visibility (swing-relevant joints only)")
            st.write(rows)
        else:
            st.caption("No landmarks on this frame.")

        st.markdown("#### Whole clip")
        st.write(
            {
                "Frames": sequence.frame_count,
                "Detected": f"{sequence.detected_count} "
                            f"({sequence.detection_rate * 100:.1f}%)",
                "Mean confidence": f"{sequence.mean_confidence():.2f}",
                "Longest gap": f"{sequence.longest_gap()[1]} frames",
                "Smoothing": sequence.smoothing,
            }
        )

        if st.button("Save this frame with overlay", use_container_width=True):
            import cv2 as _cv2

            destination = (
                exports_dir(record.swing_id)
                / f"pose_frame_{frame_index:06d}.png"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            ok, buffer = _cv2.imencode(".png", image_bgr)
            if not ok:
                st.error("Could not encode this frame as PNG.")
            else:
                destination.write_bytes(buffer.tobytes())
                st.success(f"Saved `{destination.name}`")
                st.download_button(
                    "Download it",
                    data=destination.read_bytes(),
                    file_name=destination.name,
                    use_container_width=True,
                )

# ---------------------------------------------------------------- Phases
with phases_tab:
    pose_info = pose_repository.load_info(record.swing_id)
    if pose_info is None:
        st.info(
            "Phase detection reads the stored pose data. Run pose estimation "
            "in the **Run analysis** tab first."
        )
        st.stop()

    detector = default_detector()
    stored = analysis_repository.load_analysis(record.swing_id)
    stale = analysis_repository.staleness_reasons(
        stored, pose_info, detector.name, detector.version
    )

    st.markdown("#### Swing phases")
    st.caption(
        f"Detector `{detector.name}` v{detector.version}. It locates address "
        "and the top of the backswing from the hand path; the remaining phases "
        "are not attempted yet and are listed as such rather than guessed."
    )

    if stored is not None and stale:
        st.warning(
            "**The stored phase detection is out of date.**\n\n"
            + "\n".join(f"- {reason}" for reason in stale)
            + "\n\nRe-run it below."
        )

    if st.button("Detect swing phases", type="primary", use_container_width=True):
        sequence = pose_repository.load_smoothed(
            record.swing_id
        ) or pose_repository.load_raw(record.swing_id)
        if sequence is None:
            st.error("The stored pose data could not be read. Re-run pose estimation.")
        else:
            phases = detector.detect(
                sequence,
                record.context.camera_view,
                timeline_is_approximate=record.container_metadata_is_inconsistent,
            )
            phase_frames = {
                phase.value: frame
                for phase in PHASE_ORDER
                if (frame := phases.frame_for(phase)) is not None
            }
            metrics = metric_registry.evaluate_all(
                sequence, record.context.camera_view, phase_frames
            )
            analysis_repository.save_analysis(
                analysis_repository.SwingAnalysis(
                    swing_id=record.swing_id,
                    phases=phases,
                    metrics=metrics,
                    pose_created_at=pose_info.created_at,
                    pose_video_fingerprint=pose_info.video_fingerprint,
                )
            )
            st.success("Phase detection complete.")
            st.rerun()

    stored = analysis_repository.load_analysis(record.swing_id)
    if stored is None:
        st.info("Phases have not been detected for this swing yet.")
        st.stop()

    for note in stored.phases.notes:
        st.warning(note)

    st.markdown("##### Detected phases")
    for phase in PHASE_ORDER:
        result = stored.phases.get(phase)
        if result is None:
            st.markdown(
                f"⚪ **{phase.display_name}** — not attempted by this detector"
            )
            continue

        if result.status.is_usable:
            seconds = result.preview_seconds(record.preview_fps)
            timing = (
                f" · preview t = {seconds:.3f}s" if seconds is not None else ""
            )
            confidence = (
                f" · confidence {result.confidence:.2f}"
                if result.confidence is not None
                else ""
            )
            # Ranges are shown as ranges. Collapsing the impact region to its
            # centre frame would present a span the camera cannot resolve as a
            # single instant.
            if result.is_range:
                span = (
                    f"preview frames **{result.start_frame}–{result.end_frame}** "
                    f"({result.end_frame - result.start_frame + 1} frames)"
                )
            else:
                span = f"preview frame **{result.start_frame}**"

            st.markdown(
                f"{result.icon} **{phase.display_name}** — {span}{timing}{confidence}"
            )
            if result.reason:
                st.caption(result.reason)
            if phase is SwingPhase.IMPACT_REGION:
                st.caption(
                    "Impact is reported as a region, not a frame: the clubhead "
                    "is not tracked, and at this frame rate it crosses the ball "
                    "in less than one frame."
                )
            st.button(
                f"Jump to {phase.display_name}",
                key=f"jump_{phase.value}",
                on_click=_set_frame,
                args=(result.start_frame,),
            )
        else:
            st.markdown(
                f"{result.icon} **{phase.display_name}** — {result.label}"
            )
            st.caption(result.reason)

    st.markdown("##### Metrics")
    view_name = record.context.camera_view.value.replace("_", " ")
    st.caption(
        f"This swing is recorded as **{view_name}**. Only metrics that a "
        f"{view_name} camera can actually support are evaluated — the rest are "
        "not applicable to this angle, rather than merely unavailable."
    )
    if not stored.metrics:
        st.info(
            f"No metrics are defined for a {view_name} view. Set the camera "
            "view on the swing to face-on or down-the-line to see metrics."
        )
    for metric in stored.metrics:
        left, right = st.columns([3, 2])
        left.markdown(f"{metric.icon} **{metric.display_name}**")
        spec_decimals = 2
        try:
            spec_decimals = metric_registry.get_spec(metric.key).decimals
        except KeyError:
            pass
        right.markdown(
            f"**{metric.display_value(spec_decimals)}**"
            if metric.status.is_usable
            else f"_{metric.label}_"
        )
        if metric.reason:
            st.caption(metric.reason)

    st.caption(
        f"Detected {stored.phases.created_at[:19].replace('T', ' ')} UTC by "
        f"`{stored.phases.detector_name}` v{stored.phases.detector_version} "
        f"(schema v{stored.schema_version})."
    )

    st.markdown("##### Source-time timing")
    if _timeline is None:
        st.caption(
            "Timing has not been measured for this swing, so durations and "
            "tempo are refused rather than estimated. Re-import to measure."
        )
    else:
        st.caption(
            f"Timing basis: **{_timeline.confidence.label}** · frame spacing "
            f"**{_timeline.rate_classification.label}** · measured "
            f"{_timeline.measured_fps:.3f} fps over {_timeline.frame_count} "
            "decoded frames."
        )

    for timing in all_source_timings(stored.phases, _timeline):
        left, right = st.columns([3, 2])
        left.markdown(f"{timing.icon} **{timing.display_name}**")
        right.markdown(
            f"**{timing.display_value(2)}**"
            if timing.status.is_usable
            else f"_{timing.label}_"
        )
        if timing.reason:
            st.caption(timing.reason)

    st.caption(
        "Durations and tempo are computed from measured presentation "
        "timestamps, not from a nominal frame rate. Phase positions above "
        "remain preview frame indices; both are shown so neither is mistaken "
        "for the other."
    )

st.divider()
st.caption(
    "Reference comparison, ball tracer, and coaching are later milestones. "
    "See docs/ROADMAP.md."
)
