# Roadmap

Status legend: ✅ done · 🔜 next · ⬜ planned

## ✅ Milestone 0 — Environment and architecture

- Environment inspected (OS, Python, FFmpeg, CPU, memory, GPU)
- Mutually compatible dependency versions selected and pinned
- Modular package structure created
- Structured logging (console + rotating file under `data/logs/`)
- Diagnostics available as a page and as `python -m golf_lab.diagnostics`
- `ARCHITECTURE.md`, `ROADMAP.md`, `LIMITATIONS.md`, `DATA_MODEL.md`,
  `RECORDING_GUIDE.md` written
- No OpenAI integration, no model downloads, no training

## ✅ Milestone 1 — Video Lab

- Local Streamlit app with the full six-page navigation
- MP4 / MOV upload with a swing metadata form
- Immutable original storage (copied once, then marked read-only)
- Metadata extraction via ffprobe with an OpenCV fallback
- Phone rotation handled correctly (tag *and* display matrix)
- Browser-compatible upright preview + thumbnail
- Frame slider, previous/next frame, first/last
- Consistent frame number and timestamp
- Save any frame as PNG or JPEG
- Per-swing directory storage, reopenable
- Clear error and progress reporting
- 113 automated tests, including UI acceptance tests that drive the real
  Streamlit pages

## 🔜 Milestone 2 — Pose overlay

- MediaPipe Pose Landmarker model management (`models/`, checksum, license, version)
- Frame-by-frame inference with progress and cancellation
- `pose_raw.npz` and `pose_smoothed.npz` stored separately; raw always retained
- Frames where estimation failed are recorded, **not** silently interpolated
- Skeleton overlay on the preview with per-frame confidence
- Pose caching keyed by video fingerprint + analysis version
- CPU by default; optional GPU delegate where supported

Suggested first slice: model download + inference on a single frame, then the
whole clip, then the overlay.

## ⬜ Milestone 1b — Recording-quality assessment (Pipeline B)

Small but high value before pose work compounds on bad footage. Reports
**separate** confidence per capability — a clip can be fine for body analysis
and useless for ball tracking:

- full golfer visible · feet/hands/ball visible · camera movement
- lighting · motion blur · resolution · frame rate
- face-on vs down-the-line suggestion (user-confirmable)

## ⬜ Milestone 3 — Swing phases and qualitative metrics

- Manual marking of P1–P9; automatic suggestions for address, top, impact, finish
- Audio-assisted impact suggestion; manual correction always wins
- Tempo (backswing/downswing ratio), head movement, hip sway/depth,
  hand height and depth
- Camera-view-gated metrics
- Qualitative classifications with low/medium/high confidence

## ⬜ Milestone 4 — Reference comparison

- Reference library with golfer, angle, club, handedness, source, licensing
- Synchronization by matched phase, not timestamp or percentage
- Hip-center and body-scale normalization; optional mirroring
- Side-by-side playback and ghost overlay
- Differences stated as observations, never automatically as faults
- Compare against a pro, your personal best, or your average

## ⬜ Milestone 5 — Manual ball tracer

- Impact confirmation, click-to-place ball, additional visible points
- Shot-shape and height presets seeding an editable spline
- Launch direction, apex, curvature, endpoint controls
- Growing tracer preview; never visible before the confirmed impact frame
- Final render at original quality with original audio remuxed

## ⬜ Milestone 6 — Assisted tracking

- Camera stabilization, ball candidate suggestions, optical-flow assistance
- Confidence scoring, manual correction, tracked vs estimated kept distinct

## ⬜ Milestone 7 — Coaching

- `LocalRuleCoachProvider` (offline) and `MockCoachProvider` (tests) first
- Up to three strengths, one priority, one possible effect, one drill
- Optional `OpenAICoachProvider` with strict structured output, explicit
  per-swing opt-in, backend-only key handling, and a global cloud kill switch

## ⬜ Milestone 8 — History and progress

- Tags, notes, trends, comparison with personal best, export management
- Analysis-version tracking so stale results are flagged rather than trusted

## ⬜ Milestone 9 — Custom detection research

- Labeling workflow, clubhead and ball datasets, YOLO experiments,
  held-out evaluation, model versioning, false-positive analysis

Only after everything above is stable, and only using corrected data collected
from real use.
