# 08 - Policy evals

Benchmark reproduction and failure mining over canonical rollout parquet -
the analysis half of the eval loop. Rollout *generation* (simulators, policy
checkpoints, GPUs) runs in a separate harness and lands here as parquet;
`episode_id` names the evaluation spec (`suite/task_id/init_state_id/seed`),
so any harness that writes the schema gets this analysis for free.

**Real benchmark data ships in this directory** ([data/](data/README.md)):
OpenVLA and VLA-JEPA on LIBERO-Spatial, 100 episodes each, seed 7 - so every
example below runs offline on CPU after a clone.

- `success_rates.py` - the scoreboard, overall and per task. On this data:
  OpenVLA 84%, VLA-JEPA 99%.
- `compare_policies.py` - pair both policies on the same specs. VLA-JEPA wins
  15 layouts OpenVLA loses (and loses none OpenVLA wins) - and every OpenVLA
  loss ran to the 250-step cap while VLA-JEPA finished the same layout in
  78-141 steps: the failures are stalls, not quick mistakes.
- `validate_protocol.py` - check the parquet against the published protocol
  (task coverage, trials/task, seed, step caps); exits non-zero on deviation,
  so it drops into CI. This dataset is the 10-trials/task quick variant; the
  canonical run is 50.
- `mine_failures.py` - label slip-then-regrasp loops from per-step
  gripper/object-height signals (synthetic rollouts, proves the labeling
  workflow end-to-end).

```bash
uv run python examples/08_policy_evals/success_rates.py
uv run python examples/08_policy_evals/compare_policies.py
uv run python examples/08_policy_evals/validate_protocol.py
uv run python examples/08_policy_evals/mine_failures.py --no-plot
```

The package side lives in `daft_physical_ai.evals`: `success_rates` /
`compare_policies` / `failure_counts` (grouped by policy so shared specs never
chimera), `detect_regrasp`, and `validate_run`.
