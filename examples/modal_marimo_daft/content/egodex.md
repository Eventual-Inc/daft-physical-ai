# EgoDex hand tracking

The EgoDex demo is a local-first example. It reads a small LeRobot-format
dataset, runs MediaPipe hand tracking on CPU, draws predicted keypoints against
EgoDex ground truth, and reports detection/PCK metrics.

Unlike the Modal-hosted DROID app, this path is meant to run on a laptop once
the local inference environment is installed.

## Run locally

```bash
source .venv/bin/activate
uv pip install --prerelease=allow --extra-index-url https://nightly.daft.ai \
  -U daft av mediapipe scipy opencv-python matplotlib
python examples/egodex_handtracking_lite/demo.py
```

The Daft nightly is currently needed for the native LeRobot reader. Once that
reader ships in a released Daft version, the nightly install step can go away.

## Notebook path

```bash
source .venv/bin/activate
jupyter lab examples/egodex_handtracking_lite/demo.ipynb
```

The committed markdown and image are already rendered, so researchers can inspect
the output before running the inference stack.
