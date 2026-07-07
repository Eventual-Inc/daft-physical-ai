"""Pose-feature tracks from hand state - pure NumPy, one pass per episode.

The pattern: aggregate each episode's ``observation.state`` rows into one
(N, 48) array with a Daft groupby, then run the vectorized geometry
(`daft_physical_ai.pose`) once over the whole episode - curl, pinch, palm
orientation per frame, and forward-difference rates on top. No frame explode,
no window functions, no model.

Runs against a tiny public EgoDex sample in LeRobot v3 format via
``daft.datasets.lerobot`` (Daft >= 0.7.17).
"""

from __future__ import annotations

import argparse

import numpy as np
from daft import col
from daft.datasets import lerobot

from daft_physical_ai.pose import EpisodeFeatureComputer


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-episode pose-feature tracks from hand state.")
    parser.add_argument("--dataset", default="pepijn223/egodex-test", help="LeRobot v3 repo id or path")
    args = parser.parse_args()

    frames = lerobot.read(args.dataset)
    episodes = (
        frames.groupby("episode_index")
        .agg(
            col("tasks").any_value(),
            col("frame_index").list_agg().alias("frame_idxs"),
            col("observation.state").list_agg().alias("states"),
            col("timestamp").list_agg().alias("timestamps"),
        )
        .sort("episode_index")
        .to_pydict()
    )

    computer = EpisodeFeatureComputer()
    print(f"{'ep':3s} {'frames':>6s} {'fps':>4s} {'curl L/R (m)':>14s} {'min pinch L/R':>14s} {'palm-up %':>10s}  task")
    for i, episode_index in enumerate(episodes["episode_index"]):
        order = np.argsort(np.asarray(episodes["frame_idxs"][i]))
        state = np.asarray(episodes["states"][i], dtype=np.float64)[order]
        timestamps = np.asarray(episodes["timestamps"][i], dtype=np.float64)[order]
        fps = (len(timestamps) - 1) / max(timestamps[-1] - timestamps[0], 1e-9)

        tracks = computer.compute(state=state)
        palm_up_fraction = float(np.mean((np.asarray(tracks["palm_up_L"]) > 0.7) | (np.asarray(tracks["palm_up_R"]) > 0.7)))
        print(
            f"{episode_index:<3d} {tracks['num_frames']:>6} {fps:4.0f} "
            f"{np.mean(tracks['curl_L']):7.3f}/{np.mean(tracks['curl_R']):5.3f} "
            f"{np.min(tracks['pinch_L']):7.3f}/{np.min(tracks['pinch_R']):5.3f} "
            f"{palm_up_fraction:>9.0%}  {episodes['tasks'][i][0][:48]}"
        )

    print(
        "\nEach row is one pass of pure NumPy over the whole episode; the same tracks"
        "\nfeed the scenario queries in examples/04_episode_operations/pose_query_segments.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
