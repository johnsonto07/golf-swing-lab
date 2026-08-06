# Golf Swing Lab

A local, private golf swing analysis workspace. Runs entirely on your own
machine and opens in your browser. No account, no cloud, no API key.

**Current status: Milestones 1 and 2 complete.** You can import swing videos,
step through them frame by frame, and run local pose estimation with a skeleton
overlay and per-joint confidence. Swing phases, comparison, ball tracer, and
coaching are later milestones — see [docs/ROADMAP.md](docs/ROADMAP.md).

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
pip install -e ".[dev,pose]"
```

`[pose]` adds MediaPipe for pose estimation. Leave it out if you only want the
Video Lab — everything except Swing Analysis works without it.

**Install it as one command, exactly as written.** Installing bare
`mediapipe` afterwards will pull numpy 2 and OpenCV 5 and break the rest of the
stack. The extra re-pins numpy, scipy, and OpenCV for that reason. Confirm with:

```powershell
pip check
```

It should print `No broken requirements found.`

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

## Using Swing Analysis (pose overlay)

1. Open **Swing Analysis** and pick a swing.
2. Go to the **Model** tab and download a pose model. This is a **one-time
   download and the only time the app uses the internet.** Three are offered:
   Lite (~5.5 MB), Full (~9 MB, recommended), Heavy (~29 MB, most accurate but
   3–5× slower on CPU).
3. Go to **Run analysis** and click **Analyse this swing**. A progress bar
   shows where it is and **Cancel** stops it. A cancelled run saves nothing —
   a partial result is discarded rather than stored as if it were complete.
4. Go to **Skeleton overlay** to step through the result:
   - switch between **smoothed** and **raw** landmarks
   - toggle face landmarks and legs, adjust line thickness
   - per-joint visibility for the swing-relevant joints is listed per frame
   - **Save this frame with overlay** writes a PNG into `exports/`

### What the overlay is telling you

- **Faded joints are low-confidence.** They are dimmed rather than hidden on
  purpose: hiding them would make a shaky estimate look like a clean one with
  fewer joints. If a wrist is faint through the downswing, don't trust anything
  measured from that wrist.
- **"No pose was detected on this frame" means exactly that.** Frames where
  estimation failed are recorded as failed and left empty. They are never
  interpolated, and no measurement is derived from them.
- **Smoothing never crosses a gap.** Each run of detected frames is filtered on
  its own, so the app can't invent motion through frames where nothing was seen.
- **Raw is always kept.** `pose_raw.npz` is the evidence; `pose_smoothed.npz`
  is a convenience. Any number can be traced back to what was observed.

If a saved analysis is out of date — because the video changed or the analysis
version moved — the page says so **before** showing its numbers, rather than
presenting stale results as current.

---

## Where your data lives

```
data/swings/<swing_id>/
    original.<ext>      your file, untouched and read-only
    preview.mp4         downscaled upright proxy for fast scrubbing
    thumbnail.jpg
    metadata.json       everything needed to reopen the swing
    pose_raw.npz        landmarks exactly as the model produced them
    pose_smoothed.npz   filtered copy for display and velocity work
    pose_info.json      which model, which versions, how good the result was
    exports/            saved frames and (later) rendered videos
data/logs/              rotating application log
models/                 downloaded pose models + pose_models.json manifest
```

Nothing under `data/` or `models/` is ever committed to Git. Delete a swing by
deleting its folder — the app deliberately does not delete your footage for you.

---

## Tests

```powershell
pytest
```

244 tests covering metadata extraction, frame-rate parsing, rotation handling,
frame accuracy, timestamp conversion, preview generation and orientation,
filename sanitization, serialization round-trips, cache keys, diagnostics
secret-safety, the full import pipeline, and all of Milestone 2: the pose
`.npz` contract, gap-aware smoothing, overlay drawing, model management and
integrity, the inference loop, and staleness detection.

The UI acceptance tests run the real Streamlit page scripts and assert that
Next/Previous move exactly one frame, that the displayed frame number and
timestamp match, that saving a frame writes to `exports/`, that the original
file is untouched afterwards, that a frame with no pose says so, that a stale
analysis is flagged before its numbers are shown, and that **no page load ever
starts a model download**.

Test fixture videos are **generated** by FFmpeg at test time, so no private
golf footage ever enters the repository. Tests that need FFmpeg skip cleanly
if it is not installed. The entire pose pipeline is tested through a fake
backend, so the suite needs neither MediaPipe nor a downloaded model nor a
network connection.

---

## Privacy

- Everything runs locally. The **only** network access in the app is
  downloading a pose model, which happens once, on an explicit button press,
  from Google's published MediaPipe model storage. Your video never leaves your
  machine — inference, overlay, and storage are entirely local.
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

**"MediaPipe is not installed"** — run `pip install -e ".[dev,pose]"` in the
activated venv and restart the app.

**Pose estimation is slow** — it is CPU-bound and runs every frame. Switch to
the Lite model, or trim the clip before importing. A 4-second 120 fps clip is
480 frames of inference.

**Numpy or OpenCV errors after installing something new** — run `pip check`.
If it complains, reinstall with `pip install -e ".[dev,pose]"`, which restores
the verified combination. Installing `mediapipe` on its own is the usual cause.

**Pose model download fails** — it is the only step needing internet. Check
your connection and retry; a failed download leaves no partial file. If your
network blocks `storage.googleapis.com`, you can place the `.task` file in
`models/` manually, though it will then be flagged as having no recorded
checksum until you re-download it through the app.

**Windows "filename or extension is too long"** — the project path is too deep
for Windows' 260-character limit. Move the project somewhere short like
`C:\golf_swing_lab`, or create the venv at a short path (`python -m venv
C:\gsl-venv`). Inside the repo, `git config core.longpaths true` handles git's
side.
