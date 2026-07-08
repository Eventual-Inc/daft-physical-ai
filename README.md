# daft-physical-ai

Physical-AI data processing on [Daft](https://github.com/Eventual-Inc/Daft), starting with hand
tracking. The methods run as Daft UDFs, so they slot into any Daft
pipeline and execute lazily, batched, and distributed.

## API

The package operates on a Daft image column and returns a hand-pose column. A
LeRobot dataset is a natural source: Daft's native reader
`daft.datasets.lerobot` (added in [Daft #7090](https://github.com/Eventual-Inc/Daft/pull/7090))
decodes each camera into an image column with `load_video_frames`.

```python
import daft
from daft.datasets import lerobot
from daft_physical_ai.hands import track_hands

# one row per frame; the camera key is decoded into an image column.
# egodex-test is a tiny EgoDex sample (3 episodes / 632 frames) in LeRobot v3 format.
df = lerobot.read("pepijn223/egodex-test", load_video_frames="observation.image")

# pick a method (each returns the same schema):
# mediapipe -> CPU, 2D only, permissive license, no weights to supply
# wilor     -> GPU, 3D MANO keypoints (MANO weights user-supplied)
df = df.with_column("hands", track_hands(df["observation.image"], method="mediapipe"))

df.write_parquet("annotated/")
```

Install the method you need as an extra: `pip install daft-physical-ai[mediapipe]`
(CPU, 2D), `pip install daft-physical-ai[wilor]` (GPU, 3D), or
`pip install daft-physical-ai[all]` for both. WiLoR additionally needs a CUDA
`torch` build and `chumpy` from git
(`pip install 'chumpy @ git+https://github.com/mattloper/chumpy'`, omitted from the
extra because PyPI metadata can't carry direct references), plus a user-supplied
`MANO_RIGHT.pkl` ([research-gated](docs/mano.md)).

> **Note:** the LeRobot reader with batched video decode
> ([Daft #7184](https://github.com/Eventual-Inc/Daft/pull/7184)) is not yet in a
> stable Daft release (latest is v0.7.17). Until it is, this repo resolves `daft`
> from the nightly index - `uv sync` handles it via the `daft-nightly` index in
> `pyproject.toml`.

## Output schema

One unified output schema regardless of method: each frame yields a list of
0-2 detected hands. A single hand value (MediaPipe):

```python
{
    "handedness": "right",        # "left", "right", or "unknown"
    "confidence": 0.979,
    "kp2d": [[1412.1, 1111.1],    # 21 image-space [x, y] keypoints
             [1357.9, 1075.9],
             ...],
    "kp3d": None,                 # 21 [x, y, z] keypoints, or null for 2D-only methods
}
```

The Daft type is `list[struct{ handedness: string, confidence: float32, kp2d:
list[list[float32]], kp3d: list[list[float32]] }]`, defined as `HANDS_DTYPE` in
`daft_physical_ai/hands/schema.py`.

## Example

A complete walkthrough - read a dataset, run `track_hands` (MediaPipe), draw the
keypoints, and score against EgoDex ground truth:

![track_hands keypoints](examples/demo_keypoints.png)

Available in three equivalent forms:

- **[examples/demo.md](examples/demo.md)** - read it start to finish; code and outputs inline.
- **[examples/demo.ipynb](examples/demo.ipynb)** - runnable notebook (outputs included).
- **[examples/demo.py](examples/demo.py)** - plain script.

Generate your own (other methods, a Modal GPU runtime, with/without eval) with the
`daft-physical-ai` CLI - run it with no arguments for an interactive walkthrough,
or pass flags:

```bash
# No flags - interactive walkthrough that asks a few questions
daft-physical-ai

# --no-input skips all prompts; flags supply the answers, the rest use defaults
daft-physical-ai --method mediapipe --output-dir my-demo --no-input
daft-physical-ai --method wilor --runtime modal --mano-path ./MANO_RIGHT.pkl --no-input
```

The bare `daft-physical-ai` command works once the package is installed. Until
it's published to PyPI, run it from a clone instead:

```bash
uv sync                          # installs the daft-physical-ai console script
uv run daft-physical-ai          # generate a demo (prefix the commands above with `uv run`)
```

To *run* a generated demo you also need its inference stack (`uv sync` already
brings the nightly Daft with the LeRobot reader). Install the extras into the
venv, then run from the activated venv - not `uv run`, which re-syncs the env
and would drop them:

```bash
source .venv/bin/activate
uv pip install -U av mediapipe scipy opencv-python matplotlib jupyterlab
jupyter lab hand-tracking-demo/demo.ipynb
```

Once the LeRobot reader lands in a stable Daft release (> v0.7.17), the nightly
pin in `pyproject.toml` goes away and this collapses to
`pip install daft-physical-ai`.

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
