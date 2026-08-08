# Limitations

Read this before trusting anything the application tells you.

## What this tool fundamentally is

A **2D video analysis workspace**. It measures pixels in a flat image and
normalizes them by apparent body size. It is not a launch monitor, not a 3D
motion-capture system, and not a substitute for a coach who can see you swing.

## Structural limits of single-camera 2D

- **No true 3D rotation.** Shoulder turn, hip turn, and spine angle cannot be
  measured accurately from one 2D camera. Anything reported about them is an
  apparent, projected quantity and will be labelled as such.
- **Perspective changes everything.** Moving the camera a metre closer, higher,
  or a few degrees around changes measured values without your swing changing
  at all. This is why comparing swings recorded from a consistent position
  matters more than any individual number.
- **Depth is invisible.** "Hand depth" down-the-line is a horizontal pixel
  offset, not a real distance.
- **No clubface data.** Face angle, strike location, spin, and dynamic loft are
  not measured and will never be inferred. Any statement about ball flight
  causes is therefore hedged.

## Limits in the current version

- Pose estimation and the skeleton overlay work; phase detection, comparison,
  tracer, coaching, and rendering do not exist yet.
- Recording-quality assessment (Pipeline B) is not implemented, so the app will
  not yet warn you that footage is unsuitable for a given analysis type.
- History shows imported swings but no trends.
- Swing deletion is intentionally not in the UI; remove folders yourself.
- **Timing is measured, not assumed** — see the section below on what is
  measured and what is refused.

## Frame timing: what is measured and what is assumed

Timing comes from the video's **own presentation timestamps**, extracted once at
import and stored in `timeline.json`. Container metadata is not trusted for it.

That distinction is not pedantry — it was a bug. A real phone clip advertised
484 frames at 22.873 fps; it actually decodes **438 frames at a constant 25.000
fps**. `nb_frames` was wrong and `avg_frame_rate`, being frames ÷ duration,
inherited the error. The clip was wrongly flagged variable-frame-rate and its
timestamps were ~9% out. Both problems came from believing the header instead
of the frames.

### Five things that are often conflated

| | What it means |
|---|---|
| **Container metadata** | The file's headers. Frequently wrong. Never used for timing. |
| **Decoded frame count** | Frames that actually decode. The source of truth. |
| **Presentation timestamps** | When each frame is shown. The basis for all durations. |
| **Measured constant-rate** | Gaps between timestamps agree within 2%. |
| **Genuine variable-rate** | Gaps really do vary. Durations still trustworthy — they are measured. |
| **Nominal-only fallback** | No timestamps readable. Durations refused. |

A file with inconsistent metadata is **not** thereby variable frame rate. The
app reports the two separately, because they have different remedies: bad
metadata needs nothing from you, while genuine VFR affects how evenly frames
are spaced in time.

### Timing trust states

Every timing-dependent number states which of these it rests on:

- **measured** — every frame timestamp read from the media. Durations and tempo
  available.
- **partially interpolated** — some frames had no readable timestamp and were
  estimated linearly between their nearest measured neighbours. Durations
  available, reported as low confidence.
- **nominal only** — no timestamps readable; timing derived from a nominal rate.
  **Durations and tempo are refused**, because a duration computed from an
  assumed rate looks precise and is wrong.
- **unavailable** — the file could not be probed at all.

### What this means for frame identity

The preview is generated with `-fps_mode passthrough`, which carries source
timestamps through unchanged. Verified on a genuinely variable-frame-rate
fixture: 60 frames in, 60 out, **0.00000 s** maximum drift. Preview frame N is
source frame N for this pipeline.

If the preview pipeline ever changes to re-encode with frame-rate conversion,
that identity must be re-established by measurement rather than assumed — see
GSL-1 in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Video handling caveats

- **Timestamps are measured, not derived.** Each frame's presentation time is
  read from the media at import. `frame_index / fps` is used only as a labelled
  fallback when no timestamps can be read, and durations are refused in that
  case.
- **Container frame counts are ignored.** `nb_frames` is recorded for comparison
  only; the decoded frame count is what the app uses.
- **Rotation is snapped to 90° steps.** Arbitrary rotation angles are not
  supported (they effectively do not occur in phone footage).
- **Codec support is FFmpeg's.** If your FFmpeg build cannot decode a clip, the
  app cannot either. The error message includes FFmpeg's own stderr.
- **10-bit / HDR HEVC** will be tone-mapped crudely by the preview encode.
  Colours in the preview may not match the original. Export uses the original.
- **Audio is not in the preview** by design. It stays in the original and will
  be remuxed at export time (Milestone 5).
- **Read-only originals are a guard rail, not security.** The chmod can fail on
  some filesystems; that is logged, not fatal.

## Frame rate and what it costs you

| Frame rate | Body analysis | Impact frame | Ball tracking |
|---|---|---|---|
| 30 fps | usable | ±1 frame ≈ 33 ms — coarse | not viable |
| 60 fps | good | ±17 ms | marginal |
| 120 fps | good | ±8 ms | workable in good light |
| 240 fps | good | ±4 ms | best, needs bright light |

At 30 fps the clubhead can travel several feet between frames. Impact will
often fall *between* two frames, and the ball may be invisible in every frame
after impact.

## Interpretation limits

The app will distinguish, and you should too:

| Kind of statement | Example |
|---|---|
| Observed difference | "Your hands are lower than the reference at the top." |
| Possible tendency | "Hands appear to move outward during early transition." |
| Confirmed fault | *requires human judgement — the app will not assert this* |
| Possible effect | "May contribute to a more leftward swing direction." |
| Causal claim | *the app will not make these* |

Differences from a professional are **not** automatically faults. Pros differ
from each other enormously, and some of your differences are your swing working
as intended.

## Ball tracer honesty

The tracer (Milestone 5) draws a smooth screen-space curve you shape by hand.
It is a visualization of the flight you saw. It is **not** measured, **not**
physically simulated, and **not** comparable to TrackMan or similar data.

## Cloud coaching (Milestone 7, not yet built)

- Fully optional; the app works completely offline without it.
- Will require explicit per-swing opt-in before any image leaves the machine.
- Will send selected key frames and computed observations — not the full video.
- The API key is read only from the `OPENAI_API_KEY` environment variable by
  the local Python backend, is never exposed to browser JavaScript, and is never
  logged, printed, or displayed.
- A setting will disable cloud analysis entirely.
- An LLM cannot see what was not measured. It will be constrained to explaining
  existing evidence, not inventing measurements.

## Reference footage and licensing

The application does not download professional footage. Reference swings must
be clips you own or that are explicitly licensed for this use. Licensing notes
are a required field in the reference library.
