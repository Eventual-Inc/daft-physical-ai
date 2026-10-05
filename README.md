# daft-physical-ai

Robotics and physical-AI data on [Daft](https://github.com/Eventual-Inc/Daft).
Dataset readers return Daft DataFrames and operations are Daft expressions, so
everything stays lazy, batched, and scales from a laptop to a cluster on Daft's
own runner. There is no separate engine or job format to learn.

```bash
pip install daft-physical-ai
```

## Quickstart

Read a LeRobot dataset, detect hands in every frame, and write the result:

```python
from daft.datasets import lerobot
from daft_physical_ai.hands import track_hands

# A tiny EgoDex sample in LeRobot v3 format: 3 episodes, 632 frames.
df = lerobot.read("pepijn223/egodex-test", load_video_frames="observation.image")
df = df.with_column("hands", track_hands(df["observation.image"], method="mediapipe"))
df.write_parquet("annotated/")
```

`pip install "daft-physical-ai[mediapipe]"` adds the CPU hand-tracking model.

## Datasets

Daft reads the common robotics formats natively. This package adds readers for
datasets whose release layout needs more than a generic reader.

| Dataset | Format | API | Guide |
|---|---|---|---|
| Any LeRobot v3 dataset | Parquet + MP4 | `daft.datasets.lerobot` (in Daft) | [Daft docs](https://docs.daft.ai) |
| DROID | HDF5 + MP4 | `daft.datasets.droid` (in Daft) | [example](examples/droid_episode_index.py) |
| Any MCAP / HDF5 / video | - | `daft.read_mcap`, `daft.functions` (in Daft) | [Daft docs](https://docs.daft.ai) |
| [EgoDex](https://github.com/apple/ml-egodex) | HDF5 + MP4 | `daft_physical_ai.datasets.egodex` | [docs/egodex.md](docs/egodex.md) |
| [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) | MCAP | `daft_physical_ai.datasets.abc` | [docs/abc.md](docs/abc.md) |
| [HIW-500](https://huggingface.co/datasets/BitRobot/HIW-500) | MCAP (ROS 2) | `daft_physical_ai.datasets.hiw500` | [docs/hiw500.md](docs/hiw500.md) |
| [REASSEMBLE](https://researchdata.tuwien.ac.at/records/0ewrv-8cb44) | HDF5 | `daft_physical_ai.datasets.reassemble` | [docs/reassemble.md](docs/reassemble.md) |
| [PX OmniSharing](https://huggingface.co/datasets/paxini/Omnisharing_DB_SampleData) | HDF5 (tactile) | `daft_physical_ai.datasets.omnisharing` | [docs/omnisharing.md](docs/omnisharing.md) |

Each reader starts with `raw()`, a catalog that lists episodes without opening
them. Functions such as `messages()`, `trajectory()`, or `camera_frames()` then
read only the topics, fields, and time windows you ask for.

## Operations

| Operation | API | Runs on | Guide |
|---|---|---|---|
| Hand tracking: 2D (MediaPipe) or 3D MANO (WiLoR) keypoints per frame | `daft_physical_ai.hands.track_hands` | CPU / GPU | [docs/hands.md](docs/hands.md) |
| Reward scoring: per-frame task progress and success probability (Robometer) | `daft_physical_ai.rewards.score_rewards` | your Robometer server | [docs/rewards.md](docs/rewards.md) |
| Motion trimming: dead frames from proprioception, without decoding video | `daft_physical_ai.proprio`, `daft_physical_ai.trim` | CPU | [docs/trim.md](docs/trim.md) |
| Hand-pose features and scenario queries (grasp, lift, pinch, ...) | `daft_physical_ai.datasets.common.ego_centric` | CPU | [example](examples/pose_features_numpy.py) |

## Examples

[examples/](examples/) has a runnable script per dataset and an executed
walkthrough (notebook, markdown, and script) per operation. The
`daft-physical-ai` CLI generates your own version of each walkthrough, including
a Modal GPU variant; see [docs/cli.md](docs/cli.md).

## Development

```bash
uv sync                      # set up env + install deps
uv run pre-commit install    # install lint/format hooks
uv run pytest tests/ -v      # run the test suite
```

Tests that read real remote data are marked `integration`; see
[TESTING.md](TESTING.md).
