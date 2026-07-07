# 06 - Writing data

Persist curated and annotated data in training-ready layouts. Curated
datasets are *views*: the manifest names which episodes made the cut and why;
media stays wherever the source dataset keeps it.

- `write_curated_manifest.py` - writes three artifacts from the committed
  data via `daft_physical_ai.curation`: motion-trimmed SFT step rows, the
  per-episode manifest (kept `[start_step, end_step]` windows), and the
  (chosen, rejected) preference pairs from the rollout comparison. Outputs
  are deterministic from committed inputs and gitignored - regenerate anytime.

Planned: LeRobot/Zarr writers as Daft support lands.
