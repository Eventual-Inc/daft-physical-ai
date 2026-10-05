"""Build a lazy DROID episode index with Daft's native reader.

`daft.datasets.droid.raw()` catalogs episodes without decoding video. Filter
and project first; materialize later. This script only prints the plan.
"""

from __future__ import annotations

import daft
from daft.datasets import droid


def build_episode_index() -> daft.DataFrame:
    episodes = droid.raw()
    successful = episodes.where(daft.col("success") == daft.lit(True))
    return successful.select(
        "uuid",
        "scene_id",
        "building",
        "current_task",
        "success",
        "trajectory_length",
        "wrist_cam_video",
        "ext1_cam_video",
        "ext2_cam_video",
    )


def main() -> None:
    build_episode_index().explain(show_all=True)


if __name__ == "__main__":
    main()
