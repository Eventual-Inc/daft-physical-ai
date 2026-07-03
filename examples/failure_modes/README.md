# Failure-mode mining with canonical episode rows

This example builds synthetic rollout failures, writes them to the canonical
one-row-per-step parquet schema, reads them back with Daft, and labels re-grasp
failure loops from the per-step gripper/object-height signals.

It is intentionally CPU-only and synthetic. The point is to prove the workflow:

1. normalize episodes into the shared schema,
2. read the parquet glob with Daft,
3. filter failures,
4. classify failure modes from step-level signals.

Run it locally:

```bash
uv run --with matplotlib python examples/failure_modes/regrasp_demo.py
```

Skip plot generation when you only want the Daft/parquet smoke test:

```bash
uv run python examples/failure_modes/regrasp_demo.py --no-plot
```
