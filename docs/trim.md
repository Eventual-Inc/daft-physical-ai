# Motion trimming

Find the dead frames in an episode - the operator setting up before anything
happens, the tail after the task is done - without decoding video. The robot's
own joint positions sit in parquet next to the mp4, so a still arm is a columnar
scan away. On DROID that is 12 GB of proprioception against 400 GB of video.

```python
from daft import col
from daft.datasets import lerobot
from daft_physical_ai.proprio import motion_scale, motion_energy, is_active
from daft_physical_ai.trim import trim_windows

STATE = "observation.state.joint_position"

episodes = lerobot.read_episodes("hf://datasets/lerobot/droid_1.0.1")
frames = lerobot.load_episode_frames(episodes, "hf://datasets/lerobot/droid_1.0.1")

scale = motion_scale(frames, STATE, dims=7)   # one pass: the typical per-dim step
frames = (
    frames
    .with_column("motion_energy", motion_energy(col(STATE), dims=7, scale=scale))
    .with_column("is_active", is_active(col("motion_energy")))
)
```

That gives two views of the same answer, for two kinds of consumer:

```python
# sampling frames (VLA training): drops the head, the tail, and interior pauses
frames.where(col("is_active"))

# decoding a video slice: one contiguous window per episode, interior pauses kept
trim_windows(frames, fps=15)
```

`trim_windows` returns one row per episode - `start_frame`, `end_frame`,
`start_ts`, `end_ts`, `kept_frames`, `trim_fraction`, and `never_active` for
aborted takes where the arm never moves at all. Pass `from_ts=` (the episode's
`videos/{key}/from_timestamp`) to get timestamps that are absolute inside the
shared mp4, which is what `score_rewards` and any decoder want.

Unlike `track_hands` and `score_rewards`, `trim_windows` takes a DataFrame and
returns a DataFrame: a window is an aggregation across an episode's rows, not a
value each row can carry. `motion_energy` and `is_active` are ordinary
expressions.

## Example

[examples/trim/](../examples/trim/) is the executed walkthrough - one DROID shard
streamed from Hugging Face, scored per frame, reduced to trim windows, and
plotted. No GPU, no server.

Generate your own (different dataset, state column, shard count):

```bash
daft-physical-ai trim     # interactive
daft-physical-ai trim --dataset my/dataset --dims 6 --no-input
```
