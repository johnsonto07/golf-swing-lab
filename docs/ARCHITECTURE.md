# Architecture

## Product stance

Golf Swing Lab is a **semi-automatic** local video-analysis workspace. The
computer proposes; you confirm. Every important detection — swing phases, ball
positions, impact frame, camera view — is editable, and a manual correction
always wins over an automatic suggestion.

The input is ordinary 2D phone video, not motion capture. The architecture
therefore separates three things that are easy to conflate:

| Layer | Example | Precision |
|---|---|---|
| Internal computation | normalized wrist displacement `0.312` | numeric, full precision |
| User-facing observation | "hands moved outward during early transition" | qualitative |
| Interpretation | "may contribute to a more leftward swing direction" | hedged, with confidence |

User-facing output never invents precision that 2D video cannot support.

## Layering

```
pages/            Streamlit UI. Layout and user intent only.
  └── golf_lab/ui.py        shared widgets, session-state keys, page chrome
golf_lab/
  config.py                 paths, versions, tunables
  logging_config.py         structured logging to console + rotating file
  diagnostics.py            environment report (also `python -m golf_lab.diagnostics`)
  models/                   Pydantic types = the on-disk serialization contract
  video/                    decode, probe, preview, frame access, (later) render
  storage/                  swing directories, metadata.json, ingestion pipeline
  pose/      (Milestone 2)  landmark backends, smoothing, measurements, quality
  swing/     (Milestone 3)  phase detection/editing, qualitative rules, comparison
  tracer/    (Milestone 5)  impact, ball points, trajectory, overlay
  coaching/  (Milestone 7)  provider abstraction: local rules / mock / OpenAI
```

**Rule: no pipeline logic in `pages/`.** A page reads user intent, calls into
`golf_lab`, and renders the result. This keeps the logic testable without
Streamlit, which is why the test suite never imports Streamlit.

### Directories that exist now

`models/`, `video/`, `storage/`, `pose/`, and `swing/` are implemented.
`tracer/` holds its data model and storage (Milestone 5, slice 5a); its
geometry, rendering, and export are still to come. `coaching/` is described
here and in `ROADMAP.md` rather than created as an empty stub file — an empty
module that imports cleanly but does nothing is worse than no module, because
it hides which milestone you are on.

### The four stages, and why they are separate

It is easy to collapse "analysing a swing" into one step. It is four, and each
consumes the previous one's output without mutating it:

| Stage | Produces | Cost | Needs the model? |
|---|---|---|---|
| **Pose inference** | `pose_raw.npz` | tens of seconds | yes |
| **Smoothing** | `pose_smoothed.npz` | milliseconds | no |
| **Phase detection** | `swing_analysis.json` | milliseconds | no |
| **Metric extraction** | same file | milliseconds | no |

Only the first is expensive and irreproducible; its raw output is the evidence
everything downstream rests on. The other three are pure functions of stored
landmarks.

That is why derived results live in a *separate file* with their own schema and
detector version. Changing a phase-detection heuristic invalidates
`swing_analysis.json` and nothing else — re-running it never touches MediaPipe
and cannot degrade the landmarks it was computed from. Staleness is two-level:
derived results go stale when the detector version, the schema, or the pose
analysis they were derived from changes.

```
golf_lab/swing/
  results.py            ResultStatus + MetricResult; the "no value without a
                        status" rules, enforced in __post_init__
  phases.py             SwingPhase, PhaseResult, SwingPhases, and the
                        SwingPhaseDetector protocol
  geometry_detector.py  first detector: address + top from the hand path
  metric_registry.py    camera-view gating, landmark requirements, evaluation
golf_lab/storage/
  analysis_repository.py  swing_analysis.json + derived-result staleness
```

### The tracer is authored, not derived

```
golf_lab/tracer/
  model.py                BallPoint, PointSource, CurveControls, TracerSpec;
                          the impact rule and the shot-shape presets
golf_lab/storage/
  tracer_repository.py    tracer.json + staleness that reports, never discards
```

Every other artifact in the project can be regenerated from the video. A
tracer cannot — it is the only thing a user makes by hand. Two consequences
follow, and both are deliberate departures from how derived results behave:

- **Staleness is reported, not acted on.** A regenerated preview means stored
  coordinates may no longer line up, so the tracer is flagged and kept.
  Silently recomputing it, the correct move for `swing_analysis.json`, would
  here mean destroying work with no way to get it back.
- **Provenance is structural.** `confirmed`, `tracked`, and `estimated` points
  are stored as different things, so a save-and-reload can never let an
  invented point pass for an observed one. Correcting a tracked point makes it
  confirmed and drops the tracker's confidence, which described the
  algorithm's guess rather than the person's.

No rendered curve is stored: the spline is a pure function of the points and
controls, so persisting its output would create a second source of truth that
could disagree with them after an edit.

### How phases are located, and why impact is a region

All seven phases come from the path of the hands. Shoulder and hip rotation are
more meaningful biomechanically, but they project very differently face-on
versus down-the-line, so a detector built on them would have to be two
detectors. Hand height and hand speed read similarly from either angle.

```
address ──► takeaway ──► top ──► downswing ──► impact region
                                           └──► follow-through ──► finish
```

| Phase | How it is found |
|---|---|
| Address | last frame of the initial quiet period |
| Takeaway | first frame of sustained motion — the same measurement as address, read from the other side |
| Top | highest hands between address and peak hand speed |
| Impact region | frames where the hands return to address height |
| Downswing | between the top and the impact region |
| Follow-through | between the impact region and the finish |
| Finish | first sustained quiet after impact |

Two of these deserve their reasoning spelled out.

**The top is bounded by peak hand speed.** The finish usually has the hands
just as high as the top, so "highest hands in the clip" finds the wrong one on
about half of all swings. Peak hand speed is always the downswing, so searching
only between address and that point separates them. The cost is a real
limitation: on a clip containing *only* a backswing, peak speed falls inside
the backswing and the bound collapses, so no top is reported. The detector
refuses rather than guessing, and a test pins that.

**Impact is a range, never a frame.** The clubhead is not tracked, so the hands
only indicate roughly when the club came back through the ball — and at 30 fps
the clubhead crosses the ball in well under one frame interval. Reporting an
"impact frame" would be a precision the input cannot support. The half-width
scales with frame rate, because a 240 fps clip genuinely localises it better.

Three properties came out of tests failing against realistic synthetic swings:
the hand path is smoothed before differentiating (raw differences of noisy
landmarks measure jitter more than motion); motion must be *sustained* over
several frames to count (one frame over threshold is noise); and the
noise-floor term is capped relative to peak speed (on a clip moving from frame
one, a percentile of the speeds measures the swing, not the noise).

**Range endpoints may fall in a pose gap.** They are boundaries derived from
neighbouring phases — the downswing runs from the frame after the top to the
frame before impact — so a gap can land on one. The invariant that *is*
enforced is that a **point** phase (address, takeaway, top, finish) never sits
on a frame where no pose was observed.

### Camera-view gating lives in the registry, not the UI

A single 2D camera cannot see what it is not pointed at. Lateral hip sway is
meaningful face-on and meaningless down-the-line; spine angle is the reverse.
Computing one anyway yields a number that is not so much wrong as
*meaningless* — and a meaningless number is worse than a missing one, because
it looks like evidence.

So a metric declares the views it supports and the landmarks it needs, and
`metric_registry.evaluate` refuses on the wrong view **before any arithmetic
runs**. The refusal is a first-class result with a reason, not a silent skip.
Adding a metric means adding a spec and a compute function; it does not mean
touching a page.

### Confidence gating and the no-placeholder rule

Every phase and metric result carries an explicit `ResultStatus`:

`available` · `low_confidence` · `missing_landmarks` ·
`unsupported_camera_view` · `insufficient_frames` · `blocked_by_timing` ·
`detection_failed`

Each names a *different remedy*, which is why they are not one "unavailable".
Only `available` and `low_confidence` carry a value; the constructors raise if
anything else does. `low_confidence` is deliberately in the first group — it is
a real measurement to weigh, not a missing one, and it arrives with the reason
attached.

This is enforced rather than documented because the failure it prevents is
silent: `0.0` displayed for a head that was never located reads as "no
movement", and nothing downstream can tell the difference.

A detector also declares `supported_phases` and simply omits the rest. "Not
attempted by this detector" and "attempted and failed" are different facts, and
conflating them hides the second one.

### Measured timing is the source of truth

`video/timeline.py` extracts every frame's presentation timestamp once at
import and stores it as `timeline.json`. That artifact — not the container — is
the authority on frame count, frame rate and frame spacing.

This replaced a set of assumptions that were measurably wrong. A real phone clip
advertised 484 frames at 22.873 fps and actually decodes 438 at a constant
25.000, so both the count and the rate taken from its headers were false, and
timestamps computed from them were ~9% out.

Three things are now kept distinct that used to be conflated:

| | Decided from | Reported as |
|---|---|---|
| Frame count | decoded frames | the frame count, full stop |
| Frame spacing | measured timestamp gaps | constant / variable / unverified |
| Metadata quality | container vs decoded | "inconsistent metadata", never "VFR" |

`TimelineConfidence` records the trust state — measured, partially interpolated,
nominal only, unavailable — and `supports_durations` gates every duration and
tempo calculation on it. Nominal-only media is refused rather than estimated,
because a duration from an assumed rate looks precise and is wrong.

Phases remain **preview frame indices**; `swing/source_timing.py` converts them
to seconds through the timeline. Both are shown in the UI so neither is mistaken
for the other. Preview frame N is source frame N for the current
`-fps_mode passthrough` pipeline, verified at 0.00000 s drift on genuinely
variable-rate input — an identity that must be re-established by measurement if
that pipeline ever changes. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) (GSL-1).

### The `pose/` package and the optional-dependency rule

MediaPipe is a large optional dependency. Nothing in `golf_lab.pose` may import
it at module scope; `mediapipe_backend.py` imports it lazily inside the
constructor and is the only file that touches it at all.

That is not tidiness — it is what lets the landmark topology, the sequence
container, smoothing, the overlay, and storage all be imported and tested on a
machine that has never installed MediaPipe and never downloaded a model. The
test suite drives the entire inference pipeline through a `FakePoseBackend`
(`tests/conftest.py`), so progress, cancellation, failed-frame bookkeeping, and
staleness are all covered without a model file or a network connection.

```
pose/
  landmarks.py         33-point topology as plain data; no imports at all
  sequence.py          PoseSequence + the .npz on-disk contract
  backend.py           PoseBackend protocol + PoseFrameResult
  mediapipe_backend.py the ONLY module that imports mediapipe
  inference.py         drives a backend over a video; progress + cancellation
  smoothing.py         Savitzky-Golay / moving average, gap-aware
  overlay.py           skeleton drawing with confidence-based fading
  model_manager.py     download, checksum, licence, manifest
```

### Deviation from the proposed layout

Page files use numeric prefixes (`1_Video_Lab.py`, `2_Swing_Analysis.py`, …).
Streamlit orders sidebar navigation by filename, so unprefixed names would put
"Compare" above "Video Lab". The mapping to the requested names is direct.

## Data flow (implemented today)

```
upload  ─►  validate ─► copy original (then chmod read-only)
                          │
                          ├─► ffprobe ─► VideoMetadata (rotation-aware)
                          ├─► ffmpeg  ─► preview.mp4 (upright, H.264, faststart, silent)
                          ├─► ffmpeg  ─► thumbnail.jpg
                          └─► metadata.json  (atomic write)

inspect ─►  FrameReader(preview.mp4) ─► exact frame ─► st.image
                          └─► save_frame ─► exports/frame_000123.png
```

### Why a preview/proxy exists

1. Phone HEVC/MOV often will not play in a browser `<video>` element at all.
2. Scrubbing a 4K 120 fps clip by decoding the original is too slow to feel
   interactive.
3. Rotation gets baked into the preview's pixels, so OpenCV — which ignores
   container rotation metadata — sees an upright image.

The preview is generated with frame-sync passthrough so **frame index N in the
preview is frame index N in the original**. Import verifies this and marks the
swing `needs_review` with an explanation if the counts disagree (the usual
cause is variable-frame-rate source video).

The original is used *only* for final export. It is copied once, marked
read-only, and never rewritten.

## Orientation handling

Phones express rotation two ways, and they disagree in sign:

- a `rotate` tag on the video stream (clockwise), and/or
- a Display Matrix in `side_data_list` whose `rotation` is **counter-clockwise**

`extract_rotation_degrees` normalizes both to a clockwise angle and prefers the
tag when both are present. `VideoMetadata.width/height` are *display*
dimensions (swapped for 90°/270°); `coded_width/coded_height` are as stored.
`tests/test_metadata.py` and `tests/test_preview.py` pin this behaviour,
including an end-to-end check that OpenCV reads the generated preview upright.

## Frame accuracy

`FrameReader` keeps its own position pointer instead of trusting
`CAP_PROP_POS_FRAMES` for every access:

- forward jump ≤ 30 frames → `grab()` forward (cheap, exact)
- anything else → seek, then decode

Tests assert that random access returns pixel-identical frames to sequential
access, and that consecutive frames of an animated fixture always differ (which
would fail if the reader ever stalled or skipped).

## Caching strategy

| Cache | Scope | Key |
|---|---|---|
| `FrameCache` | in-process, 24 frames, LRU | file fingerprint + frame index + settings |
| `st.cache_resource` on `FrameReader` | Streamlit session | path + size + mtime |
| preview / thumbnail | on disk, permanent | swing directory |

File identity uses `name + size + mtime`, not a content hash — hashing
gigabytes on every page load is not viable, and replacement is the realistic
invalidation case.

Later milestones extend the same pattern: `pose_raw.npz` / `pose_smoothed.npz`
are computed once per swing and keyed by video fingerprint + analysis version.

## Expensive work is explicit

Streamlit reruns the entire script on every widget interaction. So:

- Import runs only on a form submit button, never on a widget change.
- The `FrameReader` is a cached resource; moving the slider does not reopen the file.
- Previous/Next use `on_click` callbacks, so state changes land *before* widgets
  re-instantiate. This is what makes single-frame stepping exact rather than
  off-by-one.
- Preview, Analyze, Save corrections, and Render are separate user actions by
  design — they are never chained implicitly.

## Environment and dependency decisions

Verified in a clean virtualenv before pinning:

| Package | Pin | Why this version |
|---|---|---|
| Python | 3.10–3.12 | MediaPipe (Milestone 2) has no wheels for 3.13 |
| streamlit | 1.41.1 | earliest release where `st.image` accepts `use_container_width` — the UI tests caught this failing on 1.39 |
| opencv-python | 4.10.0.84 | last 4.x line verified against numpy 1.x |
| numpy | **1.26.4** | pinned to 1.x deliberately — MediaPipe 0.10.x and OpenCV wheels are built against the numpy 1 ABI, and numpy 2 breaks them |
| scipy | 1.13.1 | compatible with numpy 1.26; provides `savgol_filter` for pose smoothing |
| pydantic | 2.9.2 | v2 API (`model_dump`, `model_validate`) used throughout |
| psutil | 6.0.0 | optional-by-design hardware probing for diagnostics |
| pytest | 8.3.3 | dev only |
| mediapipe | 0.10.14 (extra) | Milestone 2; **its transitive deps must be pinned too** — see below |

### Why `opencv-contrib-python`, not `opencv-python`

MediaPipe depends on `opencv-contrib-python`. Installing plain `opencv-python`
alongside it puts two distributions into the same `cv2` namespace, which
upstream explicitly warns against. Contrib is a strict superset, so the project
depends on it directly and there is exactly one `cv2`.

### Why the `pose` extra re-pins numpy, scipy, and opencv

Installing bare `mediapipe==0.10.14` resolves `opencv-contrib-python` to 5.x,
which drags in numpy 2 and silently breaks every wheel built against the numpy
1 ABI. This was observed in practice, not theorised: a plain
`pip install mediapipe==0.10.14` upgraded numpy to 2.5.1 and opencv to 5.0.0,
leaving two conflicting cv2 packages installed.

The `pose` extra therefore repeats `numpy`, `scipy`, and
`opencv-contrib-python` at their pinned versions. They are constraints, not
decoration: without them `pip install -e ".[pose]"` does not produce the
combination that was verified to import and run together. `pip check` passing
is part of the definition of a working environment here.

FFmpeg is an external system dependency, not a Python package. `find_ffmpeg`
never crashes when it is missing — pages show an actionable install message and
Settings reports it.

The numpy 1.x pin is the single most important line in `pyproject.toml`.
Upgrading it before MediaPipe is integrated will produce confusing binary
errors.

### Capturing a click on an image (Milestone 5)

Streamlit cannot report where on an image the user clicked, and the tracer's
core interaction is placing a ball by clicking it. Three options were weighed:

| Option | Verdict |
|---|---|
| `streamlit-image-coordinates` | **chosen** — a small JS component with no Python dependencies of its own, so it cannot reach the numpy 1 ABI |
| `streamlit-drawable-canvas` | richer, but heavier and has historically lagged Streamlit releases, which is real risk against the 1.41.1 pin |
| number inputs + rendered crosshair | no dependency at all, but turns the central interaction into type-and-check |

Having no Python dependencies is the deciding property, not the file size: the
failure this project keeps hitting is a transitive resolve that moves numpy or
opencv. A component that ships only JavaScript cannot cause it.

The decision is recorded here, but the dependency is **not yet added**. It gets
verified in a clean virtualenv against `streamlit==1.41.1`, with `pip check`,
before it is pinned — the same bar every other dependency cleared. Slices 5a-5c
deliberately need none of it, so the model, geometry, and rendering layers stay
testable headlessly whatever happens to the component.

## Error handling

- Every failure mode the user can hit produces a message that says what failed
  and what to do about it, not a stack trace.
- `FFmpegCommandError` carries the tail of stderr so codec problems are
  diagnosable rather than generic.
- A failed import removes its half-built directory; no orphan folders.
- `list_records` skips an unreadable `metadata.json` and logs it, so one bad
  file cannot hide the rest of your history.

## Privacy

No network code path exists in Milestone 0/1. No API key is read, and none is
required. `openai_integration_enabled()` reports only presence/absence as a
boolean and never returns, logs, or displays the value. When cloud coaching
arrives in Milestone 7 it will be opt-in per swing, backend-only, and will send
selected frames plus computed observations — never the whole video by default.

## Planned modules

| Module | Milestone | Responsibility |
|---|---|---|
| `video/audio.py` | 5–7 | extract audio, remux into exports, audio-peak impact hint |
| `video/renderer.py` | 5 | overlay-drawing render at original quality, then audio remux |
| `video/stabilization.py` | 6 | camera-motion compensation before ball tracking |
| `pose/backend.py` | 2 | model management: file, version, source, license, checksum, device |
| `pose/mediapipe_backend.py` | 2 | Pose Landmarker inference, CPU by default |
| `pose/smoothing.py` | 2 | configurable temporal smoothing; raw values always retained |
| `pose/measurements.py` | 3 | body-scale-normalized quantities |
| `pose/quality.py` | 1b/2 | Pipeline B recording-quality assessment, per-capability confidence |
| `swing/phase_detection.py` | 3 | suggest P1/P4/P7/P9 |
| `swing/phase_editor.py` | 3 | manual overrides that outrank suggestions |
| `swing/qualitative_rules.py` | 3 | numbers → hedged classifications + confidence |
| `swing/normalization.py` | 4 | hip-center and body-scale normalization |
| `swing/comparison.py` | 4 | phase-synchronized reference comparison |
| `tracer/*` | 5–6 | impact, ball points, spline trajectory, overlay |
| `coaching/provider.py` | 7 | provider interface; local rules and mock come first |
| `storage/reference_repository.py` | 4 | reference library incl. licensing notes |
