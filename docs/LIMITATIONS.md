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

## Limits in the current version (Milestone 2)

- Pose estimation and the skeleton overlay work; phase detection, comparison,
  tracer, coaching, and rendering do not exist yet.
- Recording-quality assessment (Pipeline B) is not implemented, so the app will
  not yet warn you that footage is unsuitable for a given analysis type.
- History shows imported swings but no trends.
- Swing deletion is intentionally not in the UI; remove folders yourself.
- **Variable-frame-rate clips have an approximate timeline** — see the section
  immediately below. This is the most consequential limitation in the build.

## Variable frame rate: what is and is not reliable

Everything interactive — scrubbing, pose, the overlay — reads the generated
`preview.mp4`, never the original. For **constant-frame-rate** footage the two
are frame-for-frame equivalent and this section does not apply.

For **variable-frame-rate** footage (many phone "auto" modes, slow-motion
clips, most screen recordings) it does. FFmpeg cannot put unevenly spaced
frames into a browser-playable proxy without resampling them to a constant
rate. A real test clip went from 484 source frames at 22.87 fps average to 438
preview frames at a constant 25 fps.

The app detects this, marks the swing `needs_review`, and shows a warning on
both the Video Lab and Swing Analysis pages.

**Still reliable on these clips:**

- the pose overlay and every landmark position
- per-joint and per-frame confidence
- what the swing looks like at a given preview frame
- saved frames and overlay exports

**Not yet reliable on these clips:**

- **Tempo** (backswing:downswing ratio) — derived from durations that are
  measured on a resampled timeline
- **Phase durations** — how long P1→P4 or P4→P7 actually took
- **Frame-perfect comparison against the source** — preview frame N is not
  original frame N, and the offset varies through the clip
- **Any timestamp read as a time in your original file** — the displayed time
  is preview time, which drifted about 9% by the end of the test clip

None of these features are implemented yet, so the app is not currently showing
you a wrong number. The restriction exists so that they are not built on a
foundation that would make them wrong. Resolving it is a prerequisite for
Milestone 3 tempo work and Milestone 5 export — tracked as **GSL-1** in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

If you need trustworthy tempo today, record in a mode that produces constant
frame rate, and check `Variable frame rate: no` in the Video Lab's Original
file panel.

## Swing phases and metrics (Milestone 3, first slice)

Only **address** and **top of backswing** are detected. Both come from the hand
path, which is the one signal a single camera reads reliably from either angle.
The other phases are not attempted, and the UI says "not attempted by this
detector" rather than leaving a gap you might read as a failure.

**Metrics depend on the camera view, and this is enforced.** A metric declares
which views it supports and is refused on the others before any arithmetic
runs. Lateral hip sway measured down-the-line would be a number describing
mostly camera-axis motion — not wrong so much as meaningless, and meaningless
numbers look like evidence.

Currently computed:

| Face-on | Down-the-line |
|---|---|
| Head sway, hip sway, shoulder tilt | Head movement, spine angle |

Declared for a view but not yet computed (reported as such, never as a value):
lead-arm angle, hip line, hip depth, posture change, shoulder line.

**Every result carries a status.** `available`, `low_confidence`,
`missing_landmarks`, `unsupported_camera_view`, `insufficient_frames`,
`blocked_by_timing`, `detection_failed`. Only the first two carry a number —
enforced in code, not just intended. A missing measurement displays as "—",
never as `0.0`.

**Not offered at all yet:** tempo ratios and phase durations in seconds. Both
require real source timing and are blocked by GSL-1 below. The phase container
deliberately exposes no duration field, so nothing downstream can accidentally
compute one.

## Video handling caveats

- **Constant-frame-rate assumption.** Timestamps are computed as
  `frame_index / fps` on the preview's timeline. See the section above for what
  this costs on variable-frame-rate sources; import detects the case both by
  comparing preview and original frame counts and by comparing the container's
  nominal and average frame rates.
- **Estimated frame counts.** When the container does not store `nb_frames`,
  the count is derived from `duration × fps` and shown as "(estimated)".
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
