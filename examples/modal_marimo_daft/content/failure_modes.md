# Failure-mode mining

Success rate is a useful scoreboard, but it does not tell a researcher why a
policy failed. The canonical episode table keeps the per-step signals needed to
ask better questions: gripper state, end-effector position, object pose, action,
media paths, and episode outcome in one Daft-readable parquet layout.

The local demo writes synthetic OpenVLA and VLA-JEPA rollout failures into that
schema, scans the parquet glob with Daft, and labels slip-then-regrasp loops from
object height and gripper state.

```python
import daft

df = daft.read_parquet("failure-mode-demo/rollouts/*.parquet")
failures = df.where(df["success"] == False).sort(["episode_id", "step_idx"])
```

Run the CPU-only smoke path:

```bash
uv run python examples/failure_modes/regrasp_demo.py --no-plot
```

Generate the hero plot locally:

```bash
uv run --with matplotlib python examples/failure_modes/regrasp_demo.py
```

This is the first clean slice of the VLA failure-mode workflow. The heavier
pieces - LIBERO simulation, OpenVLA, VLA-JEPA, and Modal GPU rollout apps - can
attach to the same episode schema later without changing the analysis surface.
