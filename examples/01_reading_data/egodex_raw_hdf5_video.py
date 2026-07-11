"""Inspect an extracted EgoDex release with Daft, without converting it first.

Download and extract EgoDex yourself under its CC-BY-NC-ND terms, then run:

    uv run python examples/egodex_raw_hdf5_video.py /data/egodex --task fold_towel

The reader catalogs HDF5 metadata lazily. It reads only the pose fields and
video frames requested below, after the task and episode filters have applied.
"""

from __future__ import annotations

from collections.abc import Sequence

import daft

from daft_physical_ai.datasets import egodex

FIELDS = ("camera/intrinsic", "transforms/leftHand", "transforms/rightHand")


def main(
    path: str,
    *,
    task: str | None = None,
    limit: int = 2,
    fields: Sequence[str] = FIELDS,
) -> daft.DataFrame:
    episodes = egodex.raw(path, tasks=task).limit(limit)
    trajectories = egodex.trajectory(episodes, fields=fields)
    frames = egodex.camera_frames(trajectories, width=224, height=224, sample_interval_seconds=1.0)
    return frames.select("task", "episode_id", "metadata", "camera/intrinsic", "video_frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read raw EgoDex HDF5 trajectories and video with Daft.")
    parser.add_argument("path", help="Root of an extracted EgoDex release")
    parser.add_argument("--task", help="Optional task directory to inspect")
    parser.add_argument("--episodes", type=int, default=2, help="Maximum episodes to inspect")
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="HDF5 trajectory field to read; repeat for multiple fields",
    )
    parser.add_argument("--limit", type=int, default=8, help="Limit the number of rows to display")
    parser.add_argument(
        "--select",
        nargs="+",
        metavar="COL",
        help="Column names to display; defaults to all",
    )
    parser.add_argument("--show-rows", type=int, default=8, help="Number of rows to show in the printed table")
    args = parser.parse_args()
    df = main(args.path, task=args.task, limit=args.episodes, fields=args.fields or FIELDS)

    out = df.select(*args.select) if args.select else df
    out.limit(args.limit).collect().show(args.show_rows)
