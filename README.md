# daft-physical-ai

Physical-AI data processing on [Daft](https://github.com/Eventual-Inc/Daft):
hand tracking and reward scoring. The methods run as Daft UDFs, so they slot
into any Daft pipeline and execute lazily, batched, and distributed.

Available on [PyPI](https://pypi.org/project/daft-physical-ai/):

```bash
pip install "daft-physical-ai[mediapipe]"
```

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

## Raw EgoDex releases

For Apple's original EgoDex HDF5+MP4 release, use the extension's lazy reader:

```python
from daft_physical_ai.datasets import egodex

episodes = egodex.raw("/data/egodex", tasks="fold_towel").limit(2)
poses = egodex.trajectory(episodes, fields=["transforms/leftHand", "transforms/rightHand"])
frames = egodex.camera_frames(poses, width=224, height=224, sample_interval_seconds=1.0)
```

EgoDex is CC-BY-NC-ND, so the package does not download, extract, or redistribute it. Download and extract the archives from the [official EgoDex repository](https://github.com/apple/ml-egodex), then point `raw()` at your copy. See [the runnable example](examples/egodex_raw_hdf5_video.py).

Install the method you need as an extra: `pip install "daft-physical-ai[mediapipe]"`
(CPU, 2D), `pip install "daft-physical-ai[wilor]"` (GPU, 3D), or
`pip install "daft-physical-ai[all]"` for both. WiLoR additionally needs a CUDA
`torch` build and `chumpy` from git
(`pip install 'chumpy @ git+https://github.com/mattloper/chumpy'`, omitted from the
extra because PyPI metadata can't carry direct references), plus a user-supplied
`MANO_RIGHT.pkl` ([research-gated](docs/mano.md)).

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

## Reward scoring

Score episodes with a reward model
([Robometer-4B](https://huggingface.co/robometer/Robometer-4B)) - per-frame
task progress (0-1) plus success probability, written back as a dataset column.
Use it to filter failed or stalled episodes before BC training, as dense reward
for RL post-training, or to catch mislabeled tasks.

```python
from daft_physical_ai.rewards import score_rewards

# one row per episode: task text, length, and where its frames live in the video
df = df.with_column(
    "rewards",
    score_rewards(
        df["task"], df["length"], df["from_ts"], df["to_ts"], df["video_path"],
        url="http://localhost:8001",   # any running Robometer eval server
        max_frames=8,                  # frames sampled per episode
    ),
)
```

Scoring is a pure HTTP call: the package never imports the model - you bring a
running [Robometer eval server](https://github.com/robometer/robometer) and
pass its URL. `daft-physical-ai rewards` scaffolds a complete demo plus the two
server scripts to run one yourself (`run_robometer_server.py` for any NVIDIA
GPU, `modal_eval_server.py` for [Modal](https://modal.com)). The output type,
defined as `REWARD_DTYPE` in `daft_physical_ai/rewards/schema.py`:

```
struct {
    reward_score:       list[float64]                          # per-frame task progress, 0-1
    robometer_success:  list[float64]                          # per-frame success probability
    reward_frames:      list[struct{index, timestamp_s}]       # which frames were scored
}
```

## Example

A complete walkthrough - read a dataset, run `track_hands` (MediaPipe), draw the
keypoints, and score against EgoDex ground truth:

![track_hands keypoints](examples/demo_keypoints.png)

Available in three equivalent forms:

- **[examples/demo.md](examples/demo.md)** - read it start to finish; code and outputs inline.
- **[examples/demo.ipynb](examples/demo.ipynb)** - runnable notebook (outputs included).
- **[examples/demo.py](examples/demo.py)** - plain script.

Generate your own (other methods, a Modal GPU runtime, with/without eval) with the
`daft-physical-ai hands` command - run it with no flags for an interactive
walkthrough, or pass flags:

```bash
# No flags - interactive walkthrough that asks a few questions
uvx daft-physical-ai hands

# --no-input skips all prompts; flags supply the answers, the rest use defaults
uvx daft-physical-ai hands --method mediapipe --output-dir my-demo --no-input
uvx daft-physical-ai hands --method wilor --runtime modal --mano-path ./MANO_RIGHT.pkl --no-input
```

`uvx` runs the CLI without installing anything (scaffolding needs no inference
deps). If the [PyPI package](https://pypi.org/project/daft-physical-ai/) is
already installed (`pip install daft-physical-ai`), plain `daft-physical-ai hands`
works too; from a clone of this repo, `uv sync` installs it (`uv run daft-physical-ai`).

Each capability is its own subcommand - `daft-physical-ai hands` and
`daft-physical-ai rewards` so far (`daft-physical-ai` with no arguments lists
what's available). The `rewards` scaffold also writes the Robometer server
scripts next to the demo, so one directory holds everything: score the
episodes, and serve the model locally or on Modal.

To *run* a generated demo you also need its inference stack. `uvx` covers that
too - one line, nothing installed:

```bash
uvx --from jupyterlab --with "daft-physical-ai[mediapipe]" --with matplotlib --with scipy \
  jupyter-lab hand-tracking-demo/demo.ipynb
```

(`scipy` is only needed if the demo includes the ground-truth eval.)

In a clone, `uv sync` already brings a Daft with the LeRobot reader; install the
extras into the venv, then run from the activated venv - not `uv run`, which
re-syncs the env and would drop them:

```bash
source .venv/bin/activate
uv pip install -U av mediapipe scipy opencv-python matplotlib jupyterlab
jupyter lab hand-tracking-demo/demo.ipynb
```

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
