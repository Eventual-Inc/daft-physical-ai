"""Merge two LeRobot recording sessions into one table.

Yesterday's teleop and today's land as two LeRobot datasets, both numbering
episodes from zero. `episode_index` and the global frame `index` collide, so
the second session is re-indexed before concat.

This demo reads the same tiny v3 dataset twice as session A and session B
(a second public v3 dataset is not available yet). The mechanics are those of
merging distinct recordings from one rig. Task strings ride on every frame, so
task identity survives without a `task_index` remap.
"""

from __future__ import annotations

import argparse

from daft import col, lit
from daft.datasets import lerobot


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge two LeRobot sessions with re-indexed episodes.")
    parser.add_argument("--session-a", default="pepijn223/egodex-test")
    parser.add_argument("--session-b", default="pepijn223/egodex-test")
    args = parser.parse_args()

    frames_a = lerobot.read(args.session_a)
    frames_b = lerobot.read(args.session_b)

    extent = frames_a.agg(
        (col("episode_index").max() + lit(1)).alias("n_episodes"),
        (col("index").max() + lit(1)).alias("n_frames"),
    ).to_pydict()
    episode_offset, frame_offset = extent["n_episodes"][0], extent["n_frames"][0]

    merged = frames_a.concat(
        frames_b.with_column("episode_index", col("episode_index") + lit(episode_offset)).with_column(
            "index", col("index") + lit(frame_offset)
        )
    )

    lengths = (
        merged.groupby("episode_index")
        .agg(col("frame_index").count().alias("frames"))
        .sort("episode_index")
        .to_pydict()
    )
    print(f"merged: {len(lengths['episode_index'])} episodes / {sum(lengths['frames'])} frames")
    for episode_index, frames in zip(lengths["episode_index"], lengths["frames"]):
        source = "A" if episode_index < episode_offset else "B"
        print(f"  episode {episode_index} (session {source}): {frames} frames")

    n_frames = merged.count_rows()
    n_distinct = merged.select("index").distinct().count_rows()
    assert n_frames == n_distinct, "global frame index must stay unique after the merge"
    print(f"\nglobal frame index unique after re-indexing: {n_distinct}/{n_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
