# Golf Swing Lab

A local, private golf swing analysis workspace. Runs entirely on your own
machine and opens in your browser. No account, no cloud, no API key.

**Current status: Milestone 1 (Video Lab) complete.** You can import swing
videos and step through them frame by frame. Pose estimation, swing phases,
comparison, ball tracer, and coaching are later milestones — see
[docs/ROADMAP.md](docs/ROADMAP.md).

---

## Setup (Windows)

### 1. Install Python 3.10, 3.11, or 3.12

Not 3.13 — MediaPipe (needed from Milestone 2) has no wheels for it yet.

```powershell
winget install Python.Python.3.12
```

Check it:

```powershell
python --version
```

### 2. Install FFmpeg

Required. Video import, previews, and exports all depend on it.

```powershell
winget install Gyan.FFmpeg
```

**Close and reopen PowerShell**, then confirm:

```powershell
ffmpeg -version
ffprobe -version
```

If `ffmpeg` is not recognized, add its `bin` folder to your PATH and reopen the
terminal.

### 3. Create a virtual environment and install

From the project folder:

```powershell
cd path\to\golf_swing_lab

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4. Verify the environment

```powershell
python -m golf_lab.diagnostics
```

This prints Python and package versions, FFmpeg version, CPU/memory/GPU info,
writable directories, free disk space, and whether optional cloud coaching is
enabled. It never prints secrets — it is safe to paste anywhere.

---

## Run the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

It opens at <http://localhost:8501>. To stop it, press `Ctrl+C` in the terminal.

To skip Streamlit's first-run email prompt:

```powershell
streamlit run app.py --server.headless true
```

---

## Using the Video Lab

1. Open **Video Lab → Import a swing**.
2. Choose an MP4 or MOV file.
3. Fill in club, camera view, handedness, and anything else you know.
   Camera view matters — later milestones only compute metrics the angle can
   actually support.
4. Click **Import swing**. Preview generation is the slow step; a progress bar
   shows where it is.
5. Switch to **Inspect frames**:
   - drag the slider for coarse positioning
   - **◀ Previous frame** / **Next frame ▶** step exactly one frame
   - frame number and timestamp are shown for the frame you are actually seeing
   - **Save frame to exports** writes a PNG or JPEG into the swing's
     `exports/` folder and offers a download

Your original file is copied into the swing folder and then marked read-only.
It is never edited, re-encoded, or overwritten.

For getting good footage in the first place, read
[docs/RECORDING_GUIDE.md](docs/RECORDING_GUIDE.md) — it matters more than any
feature here.

---

## Where your data lives

```
data/swings/<swing_id>/
    original.<ext>      your file, untouched and read-only
    preview.mp4         downscaled upright proxy for fast scrubbing
    thumbnail.jpg
    metadata.json       everything needed to reopen the swing
    exports/            saved frames and (later) rendered videos
data/logs/              rotating application log
```

Nothing under `data/` or `models/` is ever committed to Git. Delete a swing by
deleting its folder — the app deliberately does not delete your footage for you.

---

## Tests

```powershell
pytest
```

93 tests covering metadata extraction, frame-rate parsing, rotation handling,
frame accuracy, timestamp conversion, preview generation and orientation,
filename sanitization, serialization round-trips, cache keys, diagnostics
secret-safety, and the full import pipeline.

Test fixture videos are **generated** by FFmpeg at test time, so no private
golf footage ever enters the repository. Tests that need FFmpeg skip cleanly
if it is not installed.

---

## Privacy

- Everything runs locally. There is no network code path in this version.
- No `OPENAI_API_KEY` is read, needed, or requested.
- Optional cloud coaching arrives in Milestone 7. It will be off by default,
  will require explicit per-swing opt-in before anything is sent, will send
  selected frames and computed observations rather than whole videos, and can
  be disabled entirely. The key would be read only from an environment
  variable by the local Python backend — never exposed to the browser, never
  logged, printed, or displayed.

---

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | layering, data flow, caching, dependency decisions |
| [ROADMAP.md](docs/ROADMAP.md) | all ten milestones and their status |
| [DATA_MODEL.md](docs/DATA_MODEL.md) | on-disk formats, types, versioning |
| [RECORDING_GUIDE.md](docs/RECORDING_GUIDE.md) | how to film swings that are actually analysable |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | **read this** — what 2D video can and cannot tell you |

---

## Troubleshooting

**"FFmpeg was not found on your PATH"** — install it (step 2), then reopen your
terminal and restart the app. The Settings page has a **Re-check environment**
button if you installed it while the app was running.

**Import fails with a codec error** — the message includes FFmpeg's own stderr.
Usually the fix is re-exporting the clip as H.264 MP4.

**Swing shows "Needs review" after import** — the preview's frame count did not
match the original's, which normally means the source is variable-frame-rate.
Frame numbers may be slightly offset. Recording with a fixed frame rate avoids
this.

**Video appears sideways** — please file this with the output of
`python -m golf_lab.diagnostics` and the `ffprobe` output for the clip;
orientation is handled explicitly and a failure would be a real bug.

**Streamlit says a port is in use** — `streamlit run app.py --server.port 8502`.
