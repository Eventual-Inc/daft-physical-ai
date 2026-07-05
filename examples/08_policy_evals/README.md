# 08 - Policy evals

Benchmark reproduction and failure mining over canonical rollout parquet -
the analysis half of the eval loop. Rollout *generation* (simulators, policy
checkpoints, GPUs) runs in a separate harness and lands here as parquet;
`episode_id` names the evaluation spec (`suite/task_id/init_state_id/seed`),
so any harness that writes the schema gets this analysis for free.

- `mine_failures.py` - writes synthetic OpenVLA / VLA-JEPA rollout failures
  into the schema, scans the parquet glob with Daft, and labels
  slip-then-regrasp loops from per-step gripper/object-height signals.
  CPU-only and synthetic on purpose: it proves the workflow end-to-end.

```bash
uv run python examples/08_policy_evals/mine_failures.py --no-plot   # smoke test
uv run --with matplotlib python examples/08_policy_evals/mine_failures.py
```

The package side lives in `daft_physical_ai.evals`: `success_rates` /
`compare_policies` / `failure_counts` (grouped by policy so shared specs never
chimera), `detect_regrasp`, and `validate_run` (checks a run against the
published LIBERO protocol: 50 trials/task, seed 7, per-suite step caps).

Planned: `success_rates.py`, `compare_policies.py`, and `validate_protocol.py`
over *hosted* rollout parquet from real benchmark runs - reproduce the
benchmark analysis with no GPU, sim, or checkpoints installed.
