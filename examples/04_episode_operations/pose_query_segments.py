"""Find the moments: scenario queries over pose-feature tracks.

Scenario predicates are plain NumPy masks over an episode's feature tracks -
``grasping`` fires while the fingers close fast, ``lifting`` while the wrist
rises - and matching frames stitch into ``[start, end]`` segments. This runs
the two state-only scenarios on a public EgoDex sample (LeRobot v3, via
``daft.datasets.lerobot``); the skeleton-gated scenarios (``writing_grip``,
``hammer_grip``, ``in_hand``, ``twisting``, ``reaching``) additionally need
the 204-D skeleton stream from the raw EgoDex HDF5.
"""

from __future__ import annotations

import argparse

import numpy as np
from daft import col
from daft.datasets import lerobot

from daft_physical_ai.pose import EpisodeFeatureComputer, scenario_mask, segments_of


def main() -> int:
    parser = argparse.ArgumentParser(description="Grasp/lift segment queries over pose tracks.")
    parser.add_argument("--dataset", default="pepijn223/egodex-test", help="LeRobot v3 repo id or path")
    parser.add_argument("--scenarios", default="grasping,lifting", help="comma-separated state-only scenarios")
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
    for i, episode_index in enumerate(episodes["episode_index"]):
        order = np.argsort(np.asarray(episodes["frame_idxs"][i]))
        state = np.asarray(episodes["states"][i], dtype=np.float64)[order]
        timestamps = np.asarray(episodes["timestamps"][i], dtype=np.float64)[order]
        fps = (len(timestamps) - 1) / max(timestamps[-1] - timestamps[0], 1e-9)

        tracks = computer.compute(state=state)
        by_tag = {
            tag: {name.removesuffix(f"_{tag}"): np.asarray(values) for name, values in tracks.items() if name.endswith(f"_{tag}")}
            for tag in ("L", "R")
        }

        print(f"episode {episode_index} ({tracks['num_frames']} frames): {episodes['tasks'][i][0][:60]}")
        for scenario in (s.strip() for s in args.scenarios.split(",")):
            mask = scenario_mask(scenario, by_tag)
            segments = segments_of(np.flatnonzero(mask).tolist())
            spans = ", ".join(f"{a / fps:.1f}-{b / fps:.1f}s" for a, b in segments) or "-"
            print(f"  {scenario:9s} {int(mask.sum()):3d} frames  {spans}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
