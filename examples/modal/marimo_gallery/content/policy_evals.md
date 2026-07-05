# Policy evals

Success rate is a useful scoreboard, but it does not tell a researcher why a
policy failed - or how two policies differ on the exact same task layouts. The
canonical episode table keeps both answers in reach: per-step signals (gripper
state, end-effector position, object pose, action, media paths) plus the
evaluation spec (`suite/task_id/init_state_id/seed` as `episode_id`), all in
one Daft-readable parquet layout.

Real benchmark rollouts ship in the repo - OpenVLA and VLA-JEPA on
LIBERO-Spatial, 100 episodes each, about 2 MB of parquet - so the analysis
runs offline on CPU after a clone. Because `episode_id` names the spec rather
than the attempt, the two policies join into a paired comparison:

```python
import daft
from daft_physical_ai.evals import compare_policies, success_rates

df = daft.read_parquet("examples/08_policy_evals/data/rollouts/*.parquet")
success_rates(df).show()                              # openvla 84%, vla_jepa 99%
paired = compare_policies(df, "openvla", "vla_jepa")  # same layouts, side by side
```

On this data the paired view is sharper than the scoreboard: VLA-JEPA wins 15
layouts OpenVLA loses (and loses none OpenVLA wins), and every OpenVLA loss
ran to the 250-step cap while VLA-JEPA finished the same layout in 78-141
steps - the failures are stalls, not quick mistakes.

Run the walkthroughs:

```bash
uv run python examples/08_policy_evals/success_rates.py
uv run python examples/08_policy_evals/compare_policies.py
uv run python examples/08_policy_evals/validate_protocol.py
uv run python examples/08_policy_evals/mine_failures.py --no-plot
```

This is the analysis half of the eval loop. Rollout *generation* - LIBERO
simulation, OpenVLA, VLA-JEPA, Modal GPU apps - runs in a separate harness and
lands here as parquet; `daft_physical_ai.evals.validate_run` then checks a run
against the published protocol (trials per task, seed 7, per-suite step caps)
without re-simulating anything.
