# Local model weights

This directory holds local model files (MediaPipe Pose Landmarker, and later
optional YOLO detectors for club/ball). No models are downloaded as part of
Milestone 0 or Milestone 1 — the Video Lab does not require pose estimation.

Nothing in this directory is committed to Git (see `.gitignore`); the folder
itself is kept via `.gitkeep`-style tracking of this README only.

When Milestone 2 adds pose estimation, a model-management module
(`golf_lab/pose/backend.py`) will document, for each model placed here:

- Filename
- Version
- Source / download URL
- License
- Checksum (when available)
- Expected input size
- Execution device it was validated on
