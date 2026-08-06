# Local model weights

This directory holds local model files (MediaPipe Pose Landmarker, and later
optional YOLO detectors for club/ball).

Nothing in this directory is committed to Git (see `.gitignore`); only this
README is tracked.

## How models get here

`golf_lab/pose/model_manager.py` manages them. Downloading is the **only**
step in this app that touches the network, so it never happens on its own:
`ensure_model()` refuses to download unless explicitly told to, and the
Download button on the Swing Analysis page is the only path that passes that
flag. Opening a page cannot start a transfer.

Downloads go to a `.part` temporary file and are only moved into place once
the whole body has arrived, so an interrupted download cannot leave a
half-written `.task` that fails later inside MediaPipe with an unreadable
error.

## Available models

| Key | File | Size | Notes |
|---|---|---|---|
| `lite` | `pose_landmarker_lite.task` | ~5.5 MB | fastest; loses fast-moving wrists near the top |
| `full` | `pose_landmarker_full.task` | ~9 MB | **default** — holds arms and hips at full swing speed |
| `heavy` | `pose_landmarker_heavy.task` | ~29 MB | most accurate; 3–5× the CPU time of Full |

All three are Google's MediaPipe Pose Landmarker float16 bundles, licensed
under the **Apache License 2.0** (<https://www.apache.org/licenses/LICENSE-2.0>).

## `pose_models.json`

Written on download, recording per model: filename, SHA-256, size, source URL,
model version, and licence.

Integrity handling is **trust-on-first-use**, and the manifest says so in an
`integrity` field. Google does not publish per-file checksums for these
bundles, so a pinned hash in the source would be security theatre. What is
recorded is the hash of whatever was actually downloaded, verified on every
later load — that reliably catches a truncated or corrupted file, but it
cannot by itself prove the first download was authentic. `verify_model()`
reports both cases distinctly, and a model present without a recorded checksum
is flagged rather than assumed fine.
