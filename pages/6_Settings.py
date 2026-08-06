"""Settings and diagnostics.

The diagnostics report never displays secret values — the OpenAI row shows
only whether a key is present in the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from golf_lab.diagnostics import collect_diagnostics  # noqa: E402
from golf_lab.ui import page_setup  # noqa: E402
from golf_lab.video.ffmpeg import refresh_ffmpeg  # noqa: E402

page_setup("Settings", icon="⚙️")
st.title("⚙️ Settings & Diagnostics")

if st.button("Re-check environment"):
    # FFmpeg discovery is cached per process; this picks up an install that
    # happened after the app started, without needing a restart.
    refresh_ffmpeg()

report = collect_diagnostics()

st.subheader("Environment")
col_a, col_b, col_c = st.columns(3)
col_a.metric("App version", report.app_version)
col_b.metric("Python", report.python_version)
col_c.metric("Analysis version", report.analysis_version)
st.caption(report.platform_summary)

st.subheader("Media tooling")
if report.ffmpeg_version:
    st.success(f"FFmpeg found: {report.ffmpeg_version}")
    st.caption(report.ffmpeg_path or "")
else:
    st.error(
        "FFmpeg not found on PATH. Import, preview, and export will not work.\n\n"
        "Windows: `winget install Gyan.FFmpeg`, then restart your terminal and this app."
    )

st.subheader("Hardware")
st.write(
    {
        "CPU": report.cpu.get("summary", "unknown"),
        "Memory": report.memory.get("summary", "unknown"),
        "GPU": report.gpu.get("summary", "unknown"),
        "Selected inference device": report.inference_device,
    }
)
st.caption(
    "CUDA is not required. All current and near-term processing runs on CPU; "
    "a GPU delegate is an optional optimization in Milestone 2."
)

st.subheader("Packages")
st.table(
    [{"package": name, "version": version} for name, version in report.packages.items()]
)

st.subheader("Storage")
st.table(report.directories)

st.subheader("Cloud features")
if report.openai_enabled:
    st.info(
        "An `OPENAI_API_KEY` is present in your environment. **No code path in this "
        "version uses it** — optional cloud coaching arrives in Milestone 7 and will "
        "require an explicit opt-in per swing before anything is sent.",
        icon="🔒",
    )
else:
    st.success(
        "No `OPENAI_API_KEY` set. The application is running fully offline, which is "
        "the intended default.",
        icon="🔒",
    )
st.caption("The key value is never logged, printed, or displayed anywhere in this app.")

st.divider()
st.subheader("Copy diagnostics")
st.caption("Safe to paste when reporting a problem — it contains no secrets.")
st.code(report.to_text(), language=None)
st.caption("Equivalent command line: `python -m golf_lab.diagnostics`")
