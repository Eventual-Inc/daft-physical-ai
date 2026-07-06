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
- `label_failures.py` - name every failure from per-step gripper/eef signals
  (`evals.classify_failure`), no simulator or VLM. Verdict on this data: 16
  of 17 failures are slip-then-regrasp fumble loops averaging ~11 grasp
  attempts before the cap; labels write only to an optional sidecar.
- `acquisition_map.py` - rank where failures concentrate into the "collect
  these next" table. Top of the real plan: task 5 (bowl on ramekin),
  re_grasp on init states 1, 3, 4, 8.
- `validate_protocol.py` - check the parquet against the published protocol
  (task coverage, trials/task, seed, step caps); exits non-zero on deviation,
  so it drops into CI. This dataset is the 10-trials/task quick variant; the
  canonical run is 50.
- `mine_failures.py` - the synthetic end-to-end demo: writes rollouts through
  the schema, mines them with Daft, plots the hero trace.

```bash
uv run python examples/08_policy_evals/success_rates.py
uv run python examples/08_policy_evals/compare_policies.py
uv run python examples/08_policy_evals/label_failures.py
uv run python examples/08_policy_evals/acquisition_map.py
uv run python examples/08_policy_evals/validate_protocol.py
uv run python examples/08_policy_evals/mine_failures.py --no-plot
```

The package side lives in `daft_physical_ai.evals`: `success_rates` /
`compare_policies` / `failure_counts` (grouped by policy so shared specs never
chimera), `classify_failure` / `label_failures`, `detect_regrasp`, and
`validate_run`.
