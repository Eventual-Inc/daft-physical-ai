"""Motion trimming demo - dead frames in lerobot/droid_1.0.1

Robot episodes open with the operator setting up and end after the task is done - dead frames that cost decode time, VLM tokens, and training steps. This demo finds them **without decoding video**: the robot's own joint positions live in parquet next to the mp4, and a still arm is a columnar scan away.

Two outputs, for two kinds of consumer: a per-frame `is_active` flag (for training that samples frames - drops interior pauses too) and one contiguous trim window per episode (for anything that decodes a video slice).
"""

# Setup
#
# Install with `pip install daft-physical-ai matplotlib`, then import.

import daft
from daft import col
from daft.datasets import lerobot

from daft_physical_ai.proprio import is_active, motion_energy, motion_scale
from daft_physical_ai.trim import trim_windows

# Configure
#
# The dataset, its state column, and how many of its data files to read. Everything streams from Hugging Face - the video is never touched.

DATASET = "lerobot/droid_1.0.1"
STATE = "observation.state.joint_position"  # the robot's own joint positions, one list per frame
DIMS = 7
FPS = 15
SHARDS = 1  # data files to read (DROID has 156; one is ~1,000 episodes)

# Build the frame DataFrame
#
# One row per frame: episode metadata from Daft's LeRobot reader joined to the per-frame parquet.

root = f"hf://datasets/{DATASET}"
episodes = lerobot.read_episodes(root).select(
    "episode_index",
    col("tasks").list_join("; ").alias("task"),
    "dataset_from_index",
)
shards = [f"{root}/data/chunk-000/file-{i:03d}.parquet" for i in range(SHARDS)]
frames = episodes.join(daft.read_parquet(shards), on="episode_index")

# droid_1.0.1 quirk: some episodes carry an orphan second recording under the
# same episode_index. A canonical row sits exactly at dataset_from_index +
# frame_index; orphans never do, so one filter drops them.
frames = frames.where(col("index") == col("dataset_from_index") + col("frame_index"))
frames = frames.select("episode_index", "task", "frame_index", STATE)

# Score the motion
#
# `motion_energy` measures how much the arm moved since the previous frame - each joint's change, normalized by that joint's typical step, combined into one number. `is_active` thresholds it, requiring 3 consecutive frames of motion so a single noisy frame doesn't count. Idle frames sit near zero; real motion is orders of magnitude above.

scale = motion_scale(frames, STATE, dims=DIMS)  # one pass: the typical per-dim step
frames = (
    frames
    .with_column("motion_energy", motion_energy(col(STATE), dims=DIMS, scale=scale))
    .with_column("is_active", is_active(col("motion_energy")))
)

# Reduce to trim windows
#
# One row per episode: the span from the first sustained motion to the last, padded 0.25s on each side. Episodes where the arm never moves at all - aborted takes - keep their full span and get flagged `never_active` instead of being trimmed to nothing.

windows = trim_windows(frames, fps=FPS)
windows.sort("trim_fraction", desc=True).show(5)

# See one episode
#
# The motion-energy curve for the most-trimmed episode, with the kept window shaded. The flat stretch outside the window is the operator not yet doing anything.

import matplotlib.pyplot as plt

worst = windows.where(~col("never_active")).sort("trim_fraction", desc=True).limit(1).to_pylist()[0]
ep = worst["episode_index"]
curve = (
    frames.where(col("episode_index") == ep)
    .sort("frame_index")
    .select("frame_index", "motion_energy")
    .to_pydict()
)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(curve["frame_index"], curve["motion_energy"], lw=1.2)
ax.axhline(0.1, ls="--", lw=1, color="tab:red", label="threshold")
ax.axvspan(worst["start_frame"], worst["end_frame"], color="tab:green", alpha=0.15, label="kept window")
ax.set_yscale("log")
ax.set_xlabel("frame")
ax.set_ylabel("motion energy")
ax.set_title(f"episode {ep}: {worst['trim_fraction']:.0%} trimmed")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# What it saves
#
# Both views over everything scanned. To trim the video itself, pass `from_ts=` (the episode's `videos/{key}/from_timestamp`) to `trim_windows` and the window comes back as absolute timestamps a decoder - or `daft_physical_ai.rewards.score_rewards` - can seek to.

totals = frames.agg(
    col("is_active").cast(daft.DataType.int64()).sum().alias("active"),
    col("frame_index").count().alias("total"),
).to_pydict()
kept = windows.agg(col("kept_frames").sum().alias("kept")).to_pydict()["kept"][0]

total, active = totals["total"][0], totals["active"][0]
print(f"{total} frames scanned")
print(f"window view:    keeps {kept}  (drops {1 - kept / total:.1%})")
print(f"per-frame view: keeps {active}  (drops {1 - active / total:.1%})")
