# Roadmap

Status legend: ✅ done · 🔜 next · ⬜ planned · 🚧 blocked

## Timing foundation — resolved

[Issue #1](https://github.com/johnsonto07/golf-swing-lab/issues/1) was filed as
"preview frames do not map back to source frames on VFR video". **Measurement
showed that diagnosis was wrong**: the clip behind it decodes 438 frames at a
constant 25.000 fps while its container claims 484 at 22.873, and the preview
pipeline preserves frame count and timestamps exactly (0.00000 s drift on a
genuinely variable-rate fixture).

Timing is now taken from measured presentation timestamps, so the features that
were blocked on it are unblocked:

| Previously blocked | Now |
|---|---|
| Tempo ratios | ✅ computed from measured timestamps |
| Phase durations in seconds | ✅ computed from measured timestamps |
| Frame-perfect source comparison | ✅ preview frame N is source frame N, verified |
| Timing-sensitive tracer export | foundation ready; export itself is Milestone 5 |

Media whose timestamps cannot be read at all still refuses durations and tempo.
That is a limit of the input, not a defect.

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

The "variable frame rate" problem reported during that verification turned out
to be inconsistent container metadata rather than a property of the video or a
fault in the preview — see the timing foundation section above. Pose
estimation, smoothing, the overlay and storage were correct throughout.

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

### ✅ Phase detection complete, with measured source timing

All seven phases — address, takeaway, top of backswing, downswing, impact
region, follow-through, finish — are located with an explicit confidence and
status, as preview-frame indices or ranges.

**Durations and tempo are now available**, computed from measured presentation
timestamps rather than preview frame arithmetic. Each states its timing basis,
and media whose timestamps cannot be read refuses them outright rather than
estimating from a nominal rate.

Preview frame indices remain visible beside the measured timestamps, so neither
is mistaken for the other.

### ⬜ Remaining in Milestone 3

- Manual phase correction that outranks the automatic suggestion
- Implement the declared-but-not-yet-computed metrics (lead-arm angle,
  hip line, hip depth, posture change, shoulder line)
- Audio-assisted impact suggestion; manual correction always wins
- Qualitative classifications with low/medium/high confidence

## ⬜ Milestone 4 — Reference comparison

- Reference library with golfer, angle, club, handedness, source, licensing
- Synchronization by matched phase, not timestamp or percentage
- Hip-center and body-scale normalization; optional mirroring
- Side-by-side playback and ghost overlay
- Differences stated as observations, never automatically as faults
- Compare against a pro, your personal best, or your average

## 🔜 Milestone 5 — Manual ball tracer

> The timing foundation this needs is in place: preview frame N is source
> frame N for the current pipeline, verified by measurement, and every frame
> carries a measured timestamp. Export still has to respect rotation, aspect
> ratio and audio, which is Milestone 5 work.

- ✅ Impact confirmation and ball points, with provenance kept distinct
- ✅ Shot-shape and height presets seeding an editable curve
- ⬜ Spline geometry: launch direction, apex, curvature, endpoint
- ⬜ Overlay rendering, growing, never before the confirmed impact frame
- ⬜ Click-to-place UI on the Ball Tracer page
- ⬜ Final render at original quality with original audio remuxed

Delivered in slices so the parts that can be tested headlessly land before the
part that needs a browser:

| Slice | Content | Status |
|---|---|---|
| 5a | `tracer/model.py`, `storage/tracer_repository.py` | done |
| 5b | spline geometry from points + controls | next |
| 5c | overlay rendering and the growing tracer | |
| 5d | interactive page: impact confirm, click-to-place | |
| 5e | export at original quality, audio remuxed | |

5a–5c carry no new dependency and are fully unit-testable. The only new
dependency in the milestone is the click-capture component in 5d — see
ARCHITECTURE.md, "Capturing a click on an image".

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
