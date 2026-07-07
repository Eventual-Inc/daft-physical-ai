"""Hand a curated Daft dataframe to a training loop - no export detour.

The end of the no-GPU half of the loop: curate demonstrations with the same
query stage 06 writes (successes, motion-trimmed), then stream it straight
into PyTorch with ``to_torch_dataloader``. Scalar columns arrive as tensors;
list columns arrive as equal-length Python lists (the fixed-size cast
guarantees that), so one ``torch.as_tensor`` stacks them to ``(batch, dim)``.

Run with torch as an ephemeral dependency:

    uv run --with torch python examples/07_training_handoff/curated_dataset_to_torch_dataloader.py

The next step after this one is the actual fine-tune (a lerobot policy on a
GPU) - out of scope here on purpose; this handoff is the boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import daft
from daft import col

from daft_physical_ai.curation import sft_view
from daft_physical_ai.operations import motion_trim

DEMOS = Path(__file__).resolve().parents[1] / "02_episode_data" / "data" / "libero_spatial_demos"


def main() -> int:
    parser = argparse.ArgumentParser(description="Curated step rows -> torch dataloader.")
    parser.add_argument("--demos", default=str(DEMOS / "*.parquet"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=3, help="batches to pull for the demo")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed - run via: uv run --with torch python", __file__)
        return 1

    demos = daft.read_parquet(args.demos)
    curated = sft_view(demos, trim_spans=motion_trim(demos))

    f32 = daft.DataType.float32()
    batches = (
        curated.select(
            col("action").cast(daft.DataType.fixed_size_list(f32, 7)),
            col("state").cast(daft.DataType.fixed_size_list(f32, 8)),
            col("gripper_state"),
            col("curation_weight"),
        )
        .shuffle()
        .to_torch_dataloader(batch_size=args.batch_size)
    )

    for i, batch in enumerate(batches):
        if i >= args.batches:
            break
        actions = torch.as_tensor(batch["action"])  # (batch, 7)
        states = torch.as_tensor(batch["state"])  # (batch, 8)
        weights = batch["curation_weight"]  # already a tensor
        print(
            f"batch {i}: actions={tuple(actions.shape)} {actions.dtype}, "
            f"states={tuple(states.shape)} {states.dtype}, weights={tuple(weights.shape)}"
        )

    print("\nA training step consumes these directly: loss(policy(states), actions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
