from __future__ import annotations

import daft
from daft.datasets import droid


def build_episode_index() -> daft.DataFrame:
    """Build a lazy DROID episode index using released Daft APIs."""
    episodes = droid.raw()

    successful_episodes = episodes.where(daft.col("success") == daft.lit(True))

    return successful_episodes.select(
        "uuid",
        "scene_id",
        "building",
        "current_task",
        "success",
        "trajectory_length",
        "wrist_video",
        "ext1_video",
        "ext2_video",
    )


def main() -> None:
    episode_index = build_episode_index()

    # Inspect the lazy plan before materializing remote data.
    episode_index.explain(show_all=True)


if __name__ == "__main__":
    main()
