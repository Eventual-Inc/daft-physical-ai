# 07 - Training handoff

Hand a curated Daft dataframe to a training loop without an export detour.

- `curated_dataset_to_torch_dataloader.py` - the same curated view stage 06
  writes (successes, motion-trimmed), streamed into PyTorch with
  `to_torch_dataloader`: `(batch, 7)` float32 actions and `(batch, 8)` states
  per batch. Run via `uv run --with torch python ...` (torch stays an
  example-only dependency).

This is the deliberate boundary of the repo: the next step is the actual
fine-tune (a lerobot policy on a GPU), which consumes exactly this handoff.
