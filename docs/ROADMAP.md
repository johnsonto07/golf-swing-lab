# Roadmap

Status legend: ✅ done · 🔜 next · ⬜ planned · 🚧 blocked

## Blocked by [issue #1](https://github.com/johnsonto07/golf-swing-lab/issues/1) (GSL-1)

Preview frames do not map back to source frames on variable-frame-rate video.
Until a verified per-frame mapping exists, these specific features cannot be
built correctly, regardless of which milestone they sit in:

| 🚧 Blocked | Milestone |
|---|---|
| Reliable tempo ratios | 3 |
| Reliable phase durations in seconds | 3 |
| Frame-perfect source comparison | 3 / 4 |
| Timing-sensitive ball-tracer export | 5 |

Everything else in those milestones — phase *marking*, geometry metrics,
normalization, overlay work — is unaffected and can proceed.

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

## ✅ Milestone 2 — Pose overlay

**Complete and released as [v0.2.0](https://github.com/johnsonto07/golf-swing-lab/releases/tag/v0.2.0).**
Verified end to end on real 4K HEVC phone footage: 438/438 frames detected,
0.86 mean confidence. 272 local tests, CI green on Python 3.10–3.12.

The variable-frame-rate timeline problem found during that verification is
**not** unfinished Milestone 2 work — pose estimation, smoothing, the overlay,
and storage are all complete and correct. It is a separate, pre-existing
limitation of the preview proxy, tracked as
[GSL-1 / issue #1](https://github.com/johnsonto07/golf-swing-lab/issues/1),
and it gates specific *future* features (listed against Milestones 3 and 5)
rather than anything delivered here.

- MediaPipe Pose Landmarker model management with three selectable models
  (lite/full/heavy), recorded source, licence, version, size, and SHA-256
- Downloads are **explicit only** — `ensure_model` refuses to fetch unless
  told to, so opening a page never starts a network transfer
- Integrity is trust-on-first-use and says so: the hash of what was actually
  downloaded is recorded and verified on every later load, which catches
  truncation and corruption without pretending to prove authenticity
- Frame-by-frame inference in MediaPipe VIDEO mode (tracking state carries
  between frames), with throttled progress and per-frame cancellation
- `pose_raw.npz` and `pose_smoothed.npz` stored separately; raw always retained
- Frames where estimation failed are recorded as failed and hold NaN — **never
  interpolated**, and the UI says so on the frame itself
- Smoothing (Savitzky-Golay or moving average) never crosses a gap; each run of
  detected frames is filtered independently
- Skeleton overlay with per-joint confidence: low-confidence joints are drawn
  faded rather than hidden, so you can see *where* the model was unsure
- Pose caching keyed by video fingerprint + analysis version, with staleness
  surfaced as plain sentences before any numbers are shown
- CPU by default; optional GPU delegate with a clear fallback message

Deferred to Milestone 3, where it belongs with the measurements that need it:
body-scale normalization and any use of `world_landmarks`.

## ⬜ Milestone 1b — Recording-quality assessment (Pipeline B)

Small but high value before pose work compounds on bad footage. Reports
**separate** confidence per capability — a clip can be fine for body analysis
and useless for ball tracking:

- full golfer visible · feet/hands/ball visible · camera movement
- lighting · motion blur · resolution · frame rate
- face-on vs down-the-line suggestion (user-confirmable)

## 🔜 Milestone 3 — Swing phases and qualitative metrics

> **Blocked on [GSL-1](KNOWN_ISSUES.md#gsl-1)** for tempo and phase durations.
> Preview frames do not map back to source frames on variable-frame-rate video,
> so any duration measured today is computed on a resampled timeline. Phase
> *marking* can proceed; anything expressed in seconds or as a ratio must wait
> for a verified per-frame preview→source mapping.

### ✅ First vertical slice (delivered)

- `golf_lab/swing/` package: result statuses, phase types, detector protocol,
  camera-view metric registry, and the first detector
- **Address** and **top of backswing** detection from the hand path, with
  confidence and an explicit status
- Camera-view gating as a registry: a metric declares the views it supports
  and is refused on others *before* any computation runs
- Seven quality states, with the invariant that only `available` and
  `low_confidence` may carry a value — enforced in the constructors
- Derived results stored separately in `swing_analysis.json` with their own
  schema and detector version, so a detector change never invalidates or
  overwrites the pose landmarks
- Phases tab in the UI: detected phases with preview frame numbers, jump-to
  buttons, per-metric status and reason, and the VFR warning retained
- 107 tests covering the above

### ✅ Second vertical slice (delivered)

- **All seven phases detected** from the hand path: address, takeaway, top of
  backswing, downswing, impact region, follow-through, finish
- Impact reported as a **region**, never a frame — the clubhead is not tracked
  and at 30 fps it crosses the ball in under one frame interval; the region
  half-width scales with frame rate
- Range phases render as ranges in the UI, with the impact caveat inline
- Dependency-ordered detection: a missing phase blocks its dependents with a
  reason naming what was absent, reported as `insufficient_frames` rather than
  `detection_failed` so a cascade stays distinguishable from a real failure
- Detector version 1 → 2, which invalidates stored analyses without re-running
  MediaPipe
- 412 tests

### ⬜ Remaining in Milestone 3

- Manual phase correction that outranks the automatic suggestion
- Implement the declared-but-not-yet-computed metrics (lead-arm angle,
  hip line, hip depth, posture change, shoulder line)
- Audio-assisted impact suggestion
- Qualitative classifications with low/medium/high confidence
- 🚧 Tempo and phase durations in seconds — **blocked on GSL-1**

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

> **Blocked on [GSL-1](KNOWN_ISSUES.md#gsl-1)** for the final render. Export
> draws onto the *original* at full quality, which requires knowing the source
> frame for each traced preview frame. On variable-frame-rate clips that
> mapping does not currently exist.

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
