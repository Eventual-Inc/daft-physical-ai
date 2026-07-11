"""Index a LeRobot v3 dataset without decoding a single video frame.

`daft.datasets.lerobot` (Daft >= 0.7.17) reads a LeRobot dataset lazily:
`read_episodes` gives one row per episode straight from the metadata,
`read_tasks` the task table, and `read` one row per frame with episode
metadata broadcast on - video stays undecoded until you ask for it with
``load_video_frames``. The dataset here is a tiny EgoDex sample (3 episodes /
632 frames) in LeRobot v3 format.
"""

from __future__ import annotations

import argparse

from daft import col
from daft.datasets import lerobot


def main() -> int:
    parser = argparse.ArgumentParser(description="Episode/task/frame views of a LeRobot dataset.")
    parser.add_argument("--dataset", default="pepijn223/egodex-test", help="HF repo id or path")
    parser.add_argument("--min-length", type=int, default=100, help="episode-length filter to demo")
    args = parser.parse_args()

    episodes = lerobot.read_episodes(args.dataset)
    index = episodes.select("episode_index", "tasks", "length").sort("episode_index").to_pydict()
    print(f"{args.dataset}: {len(index['episode_index'])} episodes")
    for episode_index, tasks, length in zip(index["episode_index"], index["tasks"], index["length"]):
        print(f"  episode {episode_index}: {length:4d} frames  {tasks[0][:70]}")

    tasks = lerobot.read_tasks(args.dataset).to_pydict()
    print(f"\n{len(next(iter(tasks.values()), []))} distinct tasks in the task table")

    long_episodes = episodes.where(col("length") >= args.min_length)
    frames = lerobot.read(args.dataset).join(
        long_episodes.select("episode_index"), on="episode_index", how="semi"
    )
    print(
        f"frames in episodes with >= {args.min_length} steps: {frames.count_rows()} "
        f"(selected without touching any video)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
