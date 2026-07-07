"""Normalize LIBERO demonstrations into the canonical step-row table.

Downloads the original robomimic-style demo HDF5 for each requested
LIBERO-Spatial task from `yifengzhu-hf/LIBERO-datasets` on Hugging Face,
parses it with `daft_physical_ai.ingest.Hdf5Ingestor`, and writes one
canonical parquet part per demonstration - signals only (actions, 8-dim
proprio state, eef position, gripper), no image bytes, so ~500 MB of HDF5
per task becomes ~1 MB of queryable step rows.

WARNING: each task file is 0.5-0.75 GB (camera frames are embedded in the
HDF5); all 10 libero_spatial tasks total ~6.2 GB of one-time download
(cached by huggingface_hub). Start with one task or pass --limit.

Run (needs the [hdf5] extra + huggingface_hub):

    uv run --with h5py --with huggingface_hub \
      python examples/02_episode_data/normalize_libero_demos.py \
      --tasks pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
      --limit 5

    # everything (the command that produced the committed slice):
    uv run --with h5py --with huggingface_hub \
      python examples/02_episode_data/normalize_libero_demos.py --tasks all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from daft_physical_ai.episodes import assert_emits_schema, write_episode
from daft_physical_ai.ingest import Hdf5Ingestor

HF_REPO = "yifengzhu-hf/LIBERO-datasets"

# The 10 libero_spatial tasks (HDF5 file stems minus `_demo`), task_id order.
LIBERO_SPATIAL_TASKS = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="LIBERO demo HDF5 -> canonical step-row parquet.")
    parser.add_argument(
        "--tasks",
        default=LIBERO_SPATIAL_TASKS[2],
        help="comma-separated task names (file stem minus `_demo`), or `all`",
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--limit", type=int, help="demos per task (default: all ~50)")
    parser.add_argument("--out-dir", type=Path, default=Path("libero-demos"))
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download  # example-only dependency

    tasks = list(LIBERO_SPATIAL_TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    print(f"{len(tasks)} task file(s), ~0.5-0.75 GB each to download (huggingface_hub caches them)")

    ingestor = Hdf5Ingestor()
    total_episodes = 0
    total_steps = 0
    for task in tasks:
        hdf5_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{args.suite}/{task}_demo.hdf5",
            repo_type="dataset",
        )
        n_episodes = 0
        for episode in ingestor.load(hdf5_path, limit=args.limit):
            assert episode.task_name == task  # the demos<->rollouts join key
            out = write_episode(episode, args.out_dir, run_id="libero-demos")
            assert_emits_schema(out)
            n_episodes += 1
            total_steps += episode.num_steps
        total_episodes += n_episodes
        print(f"  {task}: {n_episodes} demos")

    print(f"\n{total_episodes} demonstrations / {total_steps} step rows -> {args.out_dir}/")
    print('read them:  daft.read_parquet("' + str(args.out_dir / "*.parquet") + '")')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
