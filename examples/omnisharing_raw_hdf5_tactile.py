"""Read selected OmniSharing DF-2 signals without downloading whole episodes."""

from __future__ import annotations

import argparse

import daft

from daft_physical_ai.datasets import omnisharing


def build(dataset_uri: str, *, limit: int = 1, side: str = "lefthand") -> daft.DataFrame:
    episodes = omnisharing.raw(dataset_uri, stage="DF-2").limit(limit)
    metadata = omnisharing.episode_metadata(episodes)
    trajectories = omnisharing.trajectory(metadata, f"observation/{side}/joints")
    return omnisharing.tactile(trajectories, side, split_by_sensor=True).select(
        "episode_key",
        "capture_key",
        "instruction",
        "frame_count",
        f"observation/{side}/joints",
        f"observation/{side}/tactile_sensors",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_uri", help="Hugging Face repo id or local/remote release directory")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--side", choices=omnisharing.SIDES, default="lefthand")
    parser.add_argument("--layout", action="store_true")
    args = parser.parse_args()

    if args.layout:
        episodes = omnisharing.raw(args.dataset_uri, stage="DF-2").limit(args.episodes)
        omnisharing.describe(episodes).show(60)
    else:
        build(args.dataset_uri, limit=args.episodes, side=args.side).show()


if __name__ == "__main__":
    main()
