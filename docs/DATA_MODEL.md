# Data model

## On-disk layout

```
data/
  logs/
    golf_swing_lab.log          rotating, 2 MB × 4
  swings/
    <swing_id>/
      original.<ext>            immutable, read-only after import
      preview.mp4               upright H.264, browser-safe, silent
      thumbnail.jpg
      metadata.json             SwingRecord (implemented)
      timeline.json             measured per-frame timestamps
      pose_raw.npz              Milestone 2
      pose_smoothed.npz         Milestone 2
      pose_info.json            Milestone 2
      swing_analysis.json       Milestone 3
      tracer.json               Milestone 5 (model implemented)
      exports/                  every generated file lands here
```

`swing_id` format: `YYYYMMDD_HHMMSS_<8 hex chars>` — chronologically sortable
and collision-resistant, so directory listing order is import order.

Paths inside `metadata.json` are **relative to the swing directory**, so the
whole `data/` folder can be moved or copied to another machine intact.

## Types (`golf_lab/models/video.py`)

### `VideoMetadata`

| Field | Notes |
|---|---|
| `path` | absolute path at probe time |
| `container_format`, `codec_name`, `audio_codec_name`, `has_audio` | from ffprobe |
| `coded_width`, `coded_height` | dimensions **as stored** |
| `width`, `height` | **display** dimensions, after rotation |
| `rotation_degrees` | clockwise, snapped to 0/90/180/270 |
| `fps` | container's nominal rate; a fallback only — see `timeline.json` |
| `avg_frame_rate` | frames ÷ duration, i.e. the rate actually observed |
| `r_frame_rate` | the nominal rate the container advertises |
| `frame_count` | the container's `nb_frames`; **not** the decoded count |
| `frame_count_is_estimated` | true when derived from `duration × fps` |
| `duration_seconds` | stream duration, else container duration |
| `probe_source` | `"ffprobe"` or `"opencv"` |

**None of these fields is trusted for timing.** They are the container's
claims, and a real clip advertised 484 frames at 22.873 fps while decoding 438
at a constant 25.000. Frame count, frame rate and frame spacing all come from
`timeline.json`, measured from presentation timestamps.

`container_rates_disagree` is true when the two advertised rates differ by more
than 2% — a tolerance chosen so NTSC-style rates (60 vs 59.94) are not flagged.
It reports **inconsistent metadata**, not variable frame rate, and must not be
used as a rate classifier.

### `SwingRecord.preview_video` — the second timeline

A full `VideoMetadata` for the generated proxy, stored alongside the
original's — useful because the two containers can report different things.

**Every interactive frame number in the app refers to the preview**, and its
authoritative frame count and timing come from `timeline.json` rather than
from either container. `preview_frame_count` and `preview_fps` remain as
fallbacks for records written before timing was measured.

`container_metadata_is_inconsistent` is true when a file's own numbers
contradict each other or the decoded stream. It drives an informational notice
only; it is **not** a statement that the video is variable frame rate. See
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) (GSL-1).

### `SwingContext` — what the user reports, never guessed

`club`, `camera_view`, `handedness`, `shot_shape`, `typical_miss`,
`carry_yards`, `notes`.

`camera_view` matters structurally: later milestones refuse to compute a metric
the camera angle cannot support.

### `SwingRecord` — the persisted swing

`swing_id`, `original_filename` (verbatim, including spaces and parentheses),
`imported_at`, the three relative paths, `video`, `context`, `status`,
`status_detail`, `app_version`, `analysis_version`.

### Enums

- `CameraView`: `face_on` · `down_the_line` · `other` · `unknown`
- `Handedness`: `right` · `left`
- `ShotShape`: `straight` · `fade` · `draw` · `slice` · `hook` · `push` · `pull` · `unknown`
- `SwingStatus`: `not_processed` · `processing` · `ready` · `needs_review` ·
  `low_confidence` · `failed`

`status_detail` always explains a non-`ready` status in plain language.

## Versioning

- `app_version` — user-facing release.
- `analysis_version` — bump whenever a change could alter computed results.
  Milestone 8 uses it to flag stale saved analyses instead of silently
  trusting them.

Both are written into every `SwingRecord` at import.

## Example `metadata.json`

```json
{
  "swing_id": "20260805_143022_a3f9c1d2",
  "original_filename": "Range Session (7 iron) 3.MOV",
  "imported_at": "2026-08-05T14:30:22.481000Z",
  "original_relpath": "original.mov",
  "preview_relpath": "preview.mp4",
  "thumbnail_relpath": "thumbnail.jpg",
  "video": {
    "path": "…/original.mov",
    "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
    "codec_name": "hevc",
    "audio_codec_name": "aac",
    "has_audio": true,
    "coded_width": 1920, "coded_height": 1080,
    "width": 1080, "height": 1920,
    "rotation_degrees": 90,
    "fps": 119.88, "frame_count": 360,
    "duration_seconds": 3.003,
    "frame_count_is_estimated": false,
    "probe_source": "ffprobe"
  },
  "context": {
    "club": "7 Iron", "camera_view": "down_the_line",
    "handedness": "right", "shot_shape": "fade",
    "typical_miss": "block right", "carry_yards": 165.0,
    "notes": "slight headwind"
  },
  "status": "ready", "status_detail": "",
  "app_version": "0.1.0", "analysis_version": "1"
}
```

## `pose_raw.npz` / `pose_smoothed.npz` (Milestone 2, implemented)

Written by `golf_lab/pose/sequence.py`. Compressed `.npz`, loaded with
`allow_pickle=False` — these files are data, and a pickled array in one would
be arbitrary code execution on load.

```
format_version   int                                   currently 1
landmarks        float32 [n_frames, 33, 3]             x, y normalized to frame; z relative depth
world_landmarks  float32 [n_frames, 33, 3]             roughly metric, hip-centred
visibility       float32 [n_frames, 33]                landmark not occluded
presence         float32 [n_frames, 33]                landmark in frame at all
detected         bool    [n_frames]                    false = estimation failed
fps              float
frame_width      int
frame_height     int
smoothing        str                                   e.g. "savgol(window=7, polyorder=2)"
metadata_keys    str[]                                 free-form provenance, stored as
metadata_values  str[]                                 two parallel string arrays
```

Landmark order is MediaPipe's 33-point BlazePose topology, defined in
`golf_lab/pose/landmarks.py`. **It is part of the on-disk contract** — the
indices must not be rearranged without bumping `analysis_version`.

Undetected frames hold **NaN**, never zeros. Zero is a valid coordinate and
would silently place a joint in the top-left corner of the image. Nothing in
the codebase fills those gaps in: `pixel_coordinates()` returns `None` for an
undetected frame rather than an array of NaNs, so callers are forced to handle
"no pose here" explicitly instead of drawing garbage.

`format_version` is checked on load. A file written by a newer layout is
refused rather than misread.

Raw is always kept alongside smoothed, and smoothing never crosses a gap —
each unbroken run of detected frames is filtered independently, because
filtering across an undetected stretch would invent plausible-looking motion
through frames where nothing was seen.

### `pose_info.json` (Milestone 2, implemented)

Provenance for one stored analysis, written **last** so it acts as the marker
that a complete analysis exists:

| Field | Notes |
|---|---|
| `model_key`, `model_filename`, `model_sha256` | which weights produced this |
| `backend`, `device`, `mediapipe_version` | how it was computed |
| `video_fingerprint` | name + size + mtime hash of the analysed video |
| `frame_count`, `detected_count`, `detection_rate` | coverage |
| `mean_confidence`, `longest_gap_frames` | quality |
| `smoothing` | settings used, so the result is reproducible |
| `analysis_version`, `app_version`, `pose_format_version` | staleness |
| `created_at`, `elapsed_seconds` | when and how long |

`staleness_reasons()` compares the stored fingerprint and versions against
current ones and returns user-facing sentences. An empty list means the cached
result is current; anything else is displayed before the numbers are shown.
This is what prevents a skeleton computed from a since-replaced video being
presented as if it were current.

### `phases.json` (Milestone 3)

Each phase records `frame`, `source` (`auto` | `manual`), and `confidence`.
A `manual` entry always outranks an `auto` one.

### `tracer.json` (Milestone 5)

`impact_frame`, plus points tagged by layer: `confirmed` (you clicked it),
`tracked` (detected), `estimated` (interpolated). Plus style settings and the
chosen shape/height presets. The renderer must never draw before
`impact_frame`.

### `analysis.json` (Milestone 3/7)

A list of observations in the shape agreed for coaching:

```json
{
  "metric": "early_downswing_hand_path",
  "classification": "moves outward",
  "confidence": "medium",
  "evidence": "Normalized wrist position moved away from the body between P4 and P5.",
  "applicable_camera_view": "down_the_line",
  "applicable_phase": "P4-P5",
  "possible_implication": "May contribute to a more leftward swing direction.",
  "limitation": "Clubface and strike location were not measured."
}
```

## Git policy

`data/` and `models/` contents are git-ignored. No user footage, no model
weights, no exports are ever committed. Test fixtures are **generated** by
FFmpeg at test time rather than checked in, so no private golf video enters the
repository.


## `timeline.json` (measured source timing)

Written at import by `video/timeline.py`. The authority on frame count, frame
rate and frame spacing — the container's own numbers are recorded only for
comparison.

| Field | Notes |
|---|---|
| `schema_version`, `extractor`, `extractor_version` | provenance; a change invalidates stored timelines |
| `confidence` | `measured` / `degraded` / `nominal` / `unavailable` |
| `rate_classification` | `constant` / `variable` / `unverified`, from measured gaps |
| `measured_fps`, `measured_duration_seconds`, `frame_count` | decoded truth |
| `nominal_fps`, `container_frame_count` | the container's claims, for comparison only |
| `source_seconds[]`, `durations[]`, `methods[]` | per frame, stored column-wise |
| `source_fingerprint`, `preview_fingerprint` | staleness detection |

`methods[i]` is `measured`, `interpolated`, or `nominal`. Durations are refused
whenever either endpoint is `nominal`, because a duration derived from an
assumed frame rate looks precise and is wrong.

Frames serialize column-wise: a 4-minute 120 fps clip is ~29k frames, and a list
of objects would cost megabytes of punctuation.


## `tracer.json` (Milestone 5 — model implemented, UI and export pending)

Written by `tracer/model.py` through `storage/tracer_repository.py`. The only
**authored** artifact in the project: everything else can be recomputed from
the video, and this cannot. Re-running pose inference or phase detection never
touches it.

| Field | Notes |
|---|---|
| `schema_version` | a change flags a stored drawing rather than redrawing it |
| `impact_frame`, `impact_source` | preview frame index, and `user` / `detector` — set together or not at all |
| `points[]` | `frame`, `x`, `y`, `source`, optional `confidence` |
| `shape`, `height`, `handedness` | the presets that seeded the curve |
| `controls` | `launch_direction_degrees`, `apex_height`, `curvature`, `endpoint` |
| `preview_fingerprint` | staleness detection |

Coordinates are normalized 0–1 against the **preview** frame, matching
`pose_raw.npz`. Storing pixels would break silently if the preview were
regenerated at a different width.

`source` is `confirmed` (a person placed it), `tracked` (Milestone 6 found it),
or `estimated` (the curve filled a gap). These are stored separately and drawn
distinctly, so a reload can never make an invented point look observed.
Correcting a tracked point makes it `confirmed` and drops the tracker's
confidence, which described the algorithm's guess and not the person's.

Two rules live in the model rather than the renderer: nothing may be placed
before the impact frame is confirmed, and moving impact later **drops** points
stranded behind it rather than hiding them. A tracer drawn before the ball is
struck is a false claim about the video, not a style choice.

No rendered curve is stored. The spline is a pure function of `points` and
`controls`, so persisting its output would create a second source of truth that
could disagree after an edit.

Staleness is reported, never acted on. A regenerated preview means the stored
coordinates may not line up any more — but the drawing is hand-made, so it is
kept and flagged instead of discarded.
