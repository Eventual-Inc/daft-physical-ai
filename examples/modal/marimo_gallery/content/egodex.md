# EgoDex hand tracking

The EgoDex demo is a local-first example. It reads a small LeRobot-format
dataset, runs MediaPipe hand tracking on CPU, draws predicted keypoints against
EgoDex ground truth, and reports detection/PCK metrics.

Unlike the Modal-hosted DROID app, this path is meant to run on a laptop once
the local inference environment is installed.

## Run locally

```bash
uv run --with av --with mediapipe --with scipy --with opencv-python --with matplotlib \
  python examples/04_episode_operations/hand_tracking/demo.py
```

The native LeRobot reader ships with Daft v0.7.17+, so the regular project
dependencies cover it - only the inference extras above are additional.

## Notebook path

```bash
source .venv/bin/activate
jupyter lab examples/04_episode_operations/hand_tracking/demo.ipynb
```

The committed markdown and image are already rendered, so researchers can inspect
the output before running the inference stack.
