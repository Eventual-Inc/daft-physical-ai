# daft-physical-ai

Physical-AI data annotation, episode analysis, and policy evals on
[Daft](https://github.com/Eventual-Inc/Daft): hand tracking, a canonical
one-row-per-step episode table, and benchmark comparison over rollout parquet.
The annotation methods run as Daft UDFs, so they slot into any Daft pipeline
and execute lazily, batched, and distributed. Runnable recipes for the whole
workflow - read, inspect, transform, operate, label, write, train, evaluate -
are indexed in [examples/](examples/README.md).

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

> **Note:** `daft.datasets.lerobot` is merged but not yet in a published Daft
> release (latest is v0.7.16; the reader lands in the next one). Until then it's
> available from Daft `main`.

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

## Episode tables

For rollout analysis and dataset normalization, `daft_physical_ai.episodes`
defines a portable Arrow/parquet contract: one row per step, with episode-level
metadata denormalized onto every row. That shape makes failure analysis a single
Daft scan:

```python
import daft
from daft_physical_ai.episodes import Episode, Step, write_episode

# Build or ingest an Episode, then write the canonical parquet part.
episode = Episode(
    episode_id="libero_spatial/0/0/openvla",
    source="rollout",
    instruction="put the bowl on the plate",
    steps=(Step(timestep=0),),
    success=False,
    terminal_failure="unlabeled",
    model="openvla-demo",
    policy_type="openvla",
)
write_episode(episode, "data/rollouts", run_id="demo")

df = daft.read_parquet("data/rollouts/*.parquet")
failures = df.where(df["success"] == False)
```

## Policy evals

`daft_physical_ai.evals` is the analysis half of the eval loop over that
contract. `episode_id` names the evaluation *spec*
(`suite/task_id/init_state_id/seed`), not the attempt - so two policies that
ran the same benchmark join into a paired, per-spec comparison, and a run can
be checked against the published protocol without re-simulating anything:

Real benchmark rollouts ship in-repo
([examples/08_policy_evals/data/](examples/08_policy_evals/data/README.md):
OpenVLA and VLA-JEPA on LIBERO-Spatial, 100 episodes each, ~2 MB), so this
runs offline after a clone:

```python
import daft
from daft_physical_ai.evals import compare_policies, success_rates, validate_run

df = daft.read_parquet("examples/08_policy_evals/data/rollouts/*.parquet")
success_rates(df).show()                                  # openvla 84%, vla_jepa 99%
paired = compare_policies(df, "openvla", "vla_jepa")      # same specs, side by side
report = validate_run(df, suite="libero_spatial", policy_type="openvla", trials_per_task=10)
assert report.ok  # task coverage, trials/task, seed 7, step caps - or named issues
```

Rollout *generation* (simulators, policy checkpoints, GPU images) stays in a
separate harness and lands here as parquet in the episode schema. The runnable
walkthroughs live in [examples/08_policy_evals/](examples/08_policy_evals/):

```bash
uv run python examples/08_policy_evals/success_rates.py      # scoreboard, per task
uv run python examples/08_policy_evals/compare_policies.py   # paired per-spec diffs
uv run python examples/08_policy_evals/validate_protocol.py  # protocol check (CI-able)
uv run python examples/08_policy_evals/mine_failures.py --no-plot
```

## Example

A complete walkthrough - read a dataset, run `track_hands` (MediaPipe), draw the
keypoints, and score against EgoDex ground truth:

![track_hands keypoints](examples/04_episode_operations/hand_tracking/demo_keypoints.png)

Available in three equivalent forms:

- **[examples/04_episode_operations/hand_tracking/demo.md](examples/04_episode_operations/hand_tracking/demo.md)** - read it start to finish; code and outputs inline, nothing to run.
- **[examples/04_episode_operations/hand_tracking/demo.ipynb](examples/04_episode_operations/hand_tracking/demo.ipynb)** - runnable notebook (outputs included).
- **[examples/04_episode_operations/hand_tracking/demo.py](examples/04_episode_operations/hand_tracking/demo.py)** - plain script.

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

To *run* a generated demo you also need its inference stack - including the LeRobot
reader, which currently ships only in nightly Daft. Install it into the venv, then run
from the activated venv - not `uv run`, which re-syncs the env and would drop the nightly:

```bash
source .venv/bin/activate
uv pip install --prerelease=allow --extra-index-url https://nightly.daft.ai \
  -U daft av mediapipe scipy opencv-python matplotlib jupyterlab
jupyter lab hand-tracking-demo/demo.ipynb
```

Once the LeRobot reader lands in a released Daft (> v0.7.16), the nightly step goes
away and this collapses to `pip install daft-physical-ai`.

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
