# Recording guide

Footage quality sets a hard ceiling on what any analysis can tell you. Ten
minutes of setup discipline is worth more than any feature in this app.

## Camera settings

- **Use your phone's built-in camera app.** Third-party and social apps
  re-encode, crop, and sometimes drop frames.
- **1080p at 120 fps** is the sweet spot for most people.
- **240 fps only in bright light.** High frame rates cut exposure time; in dim
  light you get a dark, noisy, motion-blurred clip that is worse than 120 fps.
- **30 fps still works** for body analysis, but impact timing is coarse and ball
  tracking is not viable. The app will tell you this.
- **No digital zoom.** It throws away real detail. Move the phone instead.
- **Lock focus and exposure** (tap and hold on most phones) so the camera does
  not hunt mid-swing.

## Camera position

- **Keep the camera completely still.** A tripod is ideal; anything stable is
  fine. Handheld footage makes ball tracking and any frame-to-frame comparison
  much harder.
- **Record face-on and down-the-line as separate clips.** A 45° compromise
  angle is unusable for both.
- **Be consistent between sessions.** Same distance, same height, same angle.
  Comparing today's swing to last month's only means something if the camera
  did not move. Mark your spot, or note the distance.

### Face-on

- Camera perpendicular to your target line, level with roughly your hip or
  chest height.
- Frame the full body plus a little headroom, with the ball visible.

### Down-the-line

- Camera directly behind you on the target line, at about hand height.
- You should see the ball, your whole body, and some space down the target line.

## Framing

- **Full body in frame for the entire swing**, including the finish. Cropping
  the head or feet removes the reference points that normalization depends on.
- Include the **club, the ball, and a little of the early ball flight** if you
  want to use the tracer later.
- Leave a little margin. Clipping a hand at the top costs you that measurement.

## Lighting

- Bright, even light. Outdoors on an overcast day is close to perfect.
- Avoid shooting into the sun or a bright window — you become a silhouette and
  pose estimation degrades badly.
- Indoors, more light beats a higher frame rate every time.

## Background and clothing

- A plain, contrasting background helps pose estimation and ball detection.
- Wear clothes that contrast with the background. Avoid clothing that matches
  it, and avoid very loose clothing that hides your body outline.

## Quick checklist

- [ ] Native camera app
- [ ] 1080p, 120 fps (240 only if bright)
- [ ] Tripod or stable rest, camera not moving
- [ ] No digital zoom
- [ ] Focus and exposure locked
- [ ] Full body, club, and ball in frame through the finish
- [ ] Face-on and down-the-line recorded separately
- [ ] Good, even light — not backlit
- [ ] Same camera position as last session

## What poor footage costs you

| Problem | Consequence |
|---|---|
| Camera moves | ball tracking unreliable; comparisons invalid |
| Feet or head cropped | body normalization and balance metrics unavailable |
| 30 fps | impact frame ±33 ms; ball often invisible after impact |
| Dim light | motion blur; low pose confidence |
| Backlit | silhouette; pose estimation fails or drifts |
| Digital zoom | lost detail that cannot be recovered |
| Inconsistent position | apparent "changes" that are just camera differences |

A future milestone will assess these automatically and report separate
confidence for body analysis, club tracking, and ball tracking — a clip can be
perfectly fine for one and useless for another.
