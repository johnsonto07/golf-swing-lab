# Known issues

Tracked defects and limitations that are understood but not yet fixed. Each
has an ID that the code and UI reference directly, so a warning a user sees on
screen can be traced to a written explanation.

| ID | Title | Severity | Blocks |
|---|---|---|---|
| [GSL-1](#gsl-1) | Preview frames do not map back to source frames on VFR video | High | Milestone 3 tempo, Milestone 5 export |

---

## GSL-1

### Preserve source timestamps and map preview frames back to the source

**Status:** open ·
**Tracked at:** <https://github.com/johnsonto07/golf-swing-lab/issues/1> ·
**Found:** 2026-08-06, testing a real 4K HEVC phone clip ·
**Blocks:** Milestone 3 (tempo and phase durations), Milestone 5 (ball-tracer
export at original quality)

### What is wrong

Every interactive feature reads the generated `preview.mp4`, not the original.
`ARCHITECTURE.md` states the invariant that "frame index N in the preview is
frame index N in the original", enforced by encoding the preview with
`-fps_mode passthrough`.

**That invariant does not hold for variable-frame-rate sources.** A real phone
clip demonstrated it:

| | Original | Preview |
|---|---|---|
| Frames | 484 | 438 |
| Frame rate | 25 nominal / 22.87 average | 25.0 constant |
| Duration | 17.55 s | 17.52 s |

46 frames were lost, and the remaining ones were placed on an evenly spaced
timeline that the source never had.

Two consequences:

1. **Frame indices are not equivalent.** Preview frame 300 is not original
   frame 300, and the offset is not a constant — it depends on where the
   source frames were actually spaced.
2. **Timestamps drift.** `FrameReader.timestamp_for_frame` divides by a single
   frame rate. On the preview's 25 fps, frame 100 reports t = 4.000 s; on the
   source's 22.87 fps average the same frame is nearer t = 4.37 s. The error
   grows through the clip and is roughly 9% on this sample.

### What is affected

Not affected — these work on whatever image is on screen:

- pose estimation, landmark positions, the skeleton overlay
- per-joint and per-frame confidence
- saving frames and overlay PNGs

Affected — anything that converts frame index to real time, or needs to reach
back to the original file:

- **tempo** (backswing:downswing ratio) — a ratio of two durations, both wrong
- **phase durations** (P1–P9) — same
- **audio-assisted impact detection** — audio comes from the original, whose
  timeline differs from the preview the frames came from
- **ball-tracer export** — renders overlays onto the *original* at full
  quality, so it needs the correct source frame for each traced preview frame

### Why it is not fixed yet

A partial fix is worse than none. Scaling timestamps by
`original_fps / preview_fps` would make the numbers *look* right while still
being wrong, because VFR spacing is not uniform — the correction differs per
frame. That would produce plausible tempo ratios that are quietly incorrect,
which is precisely the failure mode this project's architecture is written to
avoid.

### What a real fix requires

1. Extract the actual presentation timestamp of every source frame:
   `ffprobe -select_streams v:0 -show_entries frame=pkt_pts_time -of json`
   (or `best_effort_timestamp_time`).
2. Extract the same for the generated preview.
3. Build and persist an explicit per-frame mapping —
   `preview_frame_index -> (source_frame_index, source_pts_seconds)` — stored
   alongside the swing, versioned like the pose data.
4. Have the UI report source time from the mapping, and keep showing preview
   frame numbers as preview frame numbers.
5. Make the ball-tracer renderer resolve target frames through the mapping
   rather than assuming index equality.
6. Verify on: a CFR clip (mapping must be the identity), the VFR clip that
   produced this issue, and a phone slow-motion clip.

Consider also writing the preview with `-vsync passthrough -copyts` so the
source timestamps survive into the proxy, which would make the mapping
recoverable rather than reconstructed.

### Acceptance criteria

- [ ] Per-frame preview→source mapping exists, is persisted, and is versioned
- [ ] Identity mapping verified on constant-frame-rate footage
- [ ] Source timestamp shown in the UI is correct on the VFR sample, verified
      against `ffprobe` frame timestamps
- [ ] Tempo and phase durations computed from source time, not preview index
- [ ] Ball-tracer export writes to the correct original frame
- [ ] `timeline_is_approximate` returns False once a swing has a valid mapping

### Current mitigation

`SwingRecord.timeline_is_approximate` is True for these clips. The Video Lab
and Swing Analysis pages show a warning, the swing is marked `needs_review` at
import, and `docs/LIMITATIONS.md` documents the restriction. Nothing in the app
currently computes tempo or phase durations, so no incorrect number is being
shown today — this issue must be resolved **before** Milestone 3 or Milestone 5
lands.
