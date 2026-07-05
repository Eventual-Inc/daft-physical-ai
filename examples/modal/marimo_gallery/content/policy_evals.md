# Policy evals

Success rate is a useful scoreboard, but it does not tell a researcher why a
policy failed - or how two policies differ on the exact same task layouts. The
canonical episode table keeps both answers in reach: per-step signals (gripper
state, end-effector position, object pose, action, media paths) plus the
evaluation spec (`suite/task_id/init_state_id/seed` as `episode_id`), all in
one Daft-readable parquet layout.

Because `episode_id` names the spec rather than the attempt, two policies that
ran the same benchmark join into a paired comparison:

```python
import daft
from daft_physical_ai.evals import compare_policies, success_rates

df = daft.read_parquet("failure-mode-demo/rollouts/*.parquet")
success_rates(df).show()
paired = compare_policies(df, "openvla", "vla_jepa")
```

The local demo writes synthetic OpenVLA and VLA-JEPA rollout failures into the
schema, scans the parquet glob with Daft, and labels slip-then-regrasp loops
from object height and gripper state.

Run the CPU-only smoke path:

```bash
uv run python examples/08_policy_evals/mine_failures.py --no-plot
```

Generate the hero plot locally:

```bash
uv run --with matplotlib python examples/08_policy_evals/mine_failures.py
```

This is the analysis half of the eval loop. Rollout *generation* - LIBERO
simulation, OpenVLA, VLA-JEPA, Modal GPU apps - runs in a separate harness and
lands here as parquet; `daft_physical_ai.evals.validate_run` then checks a run
against the published protocol (50 trials/task, seed 7, per-suite step caps)
without re-simulating anything.
