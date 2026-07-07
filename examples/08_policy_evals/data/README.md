# LIBERO-Spatial rollout parquet (OpenVLA vs VLA-JEPA)

Real benchmark rollouts, one row per step in the
[`daft_physical_ai.episodes`](../../../daft_physical_ai/episodes/schema.py)
schema - 23,283 rows across 200 episodes, ~2 MB. Small enough to live in-repo
so every example in this directory runs offline after a clone.

| | |
|---|---|
| Suite | `libero_spatial` (10 tasks x 10 init states, seed 7) |
| Policies | `openvla` (100 episodes) - `vla_jepa` (100 episodes) |
| Protocol | 10 trials/task quick variant of the published protocol (canonical is 50; same seed, settle steps, and step caps) |
| Generated | 2026-07, on Modal A100-40GB by the VLA-JEPA harness (in-process closed loop, LIBERO `OffScreenRenderEnv`) |
| Checkpoints | `openvla/openvla-7b-finetuned-libero-spatial` - `lerobot/VLA-JEPA-LIBERO` |

`rollouts/openvla.parquet` and `rollouts/vla_jepa.parquet` are byte-faithful
compactions of the harness's per-episode parts (100 parts each, concatenated
in spec order, schema and rows unchanged).

Read it:

```python
import daft

df = daft.read_parquet("examples/08_policy_evals/data/rollouts/*.parquet")
```

Known quirks, preserved as generated:

- `model` is an empty string on the `openvla` rows (the harness didn't stamp
  its default checkpoint id); the checkpoint is the one listed above. Group by
  `policy_type`.
- `frame_path` / `wrist_path` / `video_path` reference the harness's Modal
  volume (`/outputs/...`) - the media itself is not included here.
- `embedding` is null (the embedding pass was not run).
- `terminal_failure` is `"unlabeled"` everywhere - labeling failures from the
  per-step signals is the point of the examples in this directory.
