# daft-physical-ai

Physical-AI data annotation on [Daft](https://docs.daft.ai), starting with hand
tracking. The annotation methods run as Daft UDFs, so they slot into any Daft
pipeline and execute lazily, batched, and distributed.

> **Status:** `track_hands` is implemented for both methods - MediaPipe (CPU, 2D)
> and WiLoR (GPU, 3D); CLI + demo are next. How each method was verified is in
> [TESTING.md](TESTING.md).

## API

The package operates on an image column and returns a hand-pose column - it
doesn't know about LeRobot, EgoDex, or Modal, so it composes with any Daft
DataFrame. A LeRobot dataset is the natural source: Daft's native reader
([`daft.datasets.lerobot`](https://docs.daft.ai/en/stable/datasets/lerobot/),
added in [Daft #7090](https://github.com/Eventual-Inc/Daft/pull/7090)) decodes
each camera into an image column with `load_video_frames`.

```python
import daft
from daft.datasets import lerobot
from daft_physical_ai.hands import track_hands

# one row per frame; the camera key is decoded into an image column.
# egodex-test is a tiny EgoDex sample (3 episodes / 632 frames) in LeRobot v3 format.
df = lerobot.read("pepijn223/egodex-test", load_video_frames="observation.image")

# wilor -> GPU, 3D MANO keypoints, both hands (MANO weights user-supplied)
df = df.with_column("hands", track_hands(df["observation.image"], method="wilor", mano_path="MANO_RIGHT.pkl"))

# mediapipe -> CPU, 2D only, permissive license, no weights to supply
df = df.with_column("hands", track_hands(df["observation.image"], method="mediapipe"))

df.write_parquet("annotated/")
```

`track_hands` is also exported at the top level
(`from daft_physical_ai import track_hands`). Any image column works - the LeRobot
reader is just the most convenient source for robot data.

Install the method you need as an extra: `pip install daft-physical-ai[mediapipe]`
(CPU, 2D) or `pip install daft-physical-ai[wilor]` (GPU, 3D). WiLoR also needs a
CUDA `torch` build and `chumpy` from git
(`pip install 'chumpy @ git+https://github.com/mattloper/chumpy'`, omitted from
the extra because PyPI metadata can't carry direct references), plus a
user-supplied `MANO_RIGHT.pkl` (research-gated). See [TESTING.md](TESTING.md).

> **Note:** `daft.datasets.lerobot` is merged but not yet in a published Daft
> release (latest is v0.7.16; the reader lands in the next one). Until then it's
> available from Daft `main`.

One unified output schema regardless of method
(`kp3d` is null for MediaPipe):
`list[struct{ handedness, confidence, kp2d, kp3d? }]`.

## Example

A complete walkthrough - read a dataset, run `track_hands` (MediaPipe), show the
keypoints, and score against EgoDex ground truth - in three equivalent forms:

- **[examples/demo.md](examples/demo.md)** - read it start to finish; code and outputs inline, nothing to run.
- **[examples/demo.ipynb](examples/demo.ipynb)** - runnable notebook (outputs included).
- **[examples/demo.py](examples/demo.py)** - plain script.

Generate your own (other methods, a Modal GPU runtime, with/without eval) with the
`daft-physical-ai` CLI - run it with no arguments for an interactive walkthrough,
or pass flags (`--method wilor --runtime modal --mano-path ... --no-input`).

## Development

```bash
uv sync                      # set up env + install deps
uv run pre-commit install    # install lint/format hooks
uv run pytest tests/ -v      # run the test suite
```

## Versioning

Versions are derived from git tags via `hatch-vcs`. Tag releases as `v0.1.0`,
`v0.2.0`, etc.

## Publishing

Publishing a GitHub release triggers `.github/workflows/publish-package.yml`,
which builds a wheel and sdist with `uv build` and uploads both to PyPI via
[trusted publishing](https://docs.pypi.org/trusted-publishers/). Configure the
trusted publisher on PyPI for this repository before the first release.
