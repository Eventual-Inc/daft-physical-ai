# LIBERO-Spatial demonstrations, canonical step rows (signals only)

The full official LIBERO-Spatial demonstration suite normalized into the
[`daft_physical_ai.episodes`](../../../daft_physical_ai/episodes/schema.py)
one-row-per-step schema - **500 demos (10 tasks x 50), 62,250 step rows,
~4.7 MB** - so the curation/training-handoff examples run offline after a
clone.

| | |
|---|---|
| Source | [`yifengzhu-hf/LIBERO-datasets`](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) - the original robomimic-style HDF5 release (~6.2 GB with embedded camera frames) |
| Extracted | actions (7-DoF EEF delta), 8-dim proprio state, `eef_pos`, `gripper_state` (qpos differencing), rewards/dones, instruction |
| Not extracted | camera frames (media stays in the source HDF5; `frame_path`/`wrist_path`/`video_path` are null) |
| Join key | `(suite, task_name)` - matches the rollout data in `examples/08_policy_evals/data/` exactly |

Regenerate with (one-time ~6.2 GB download, cached by `huggingface_hub`):

```bash
uv run --with h5py --with huggingface_hub \
  python examples/02_episode_data/normalize_libero_demos.py --tasks all --out-dir /tmp/libero-demos
```

then compact one parquet per task (zstd), as committed here.

Known quirks, preserved as parsed:

- `success` is `true` on all 500 demos (they are curated expert demonstrations).
- `task_id` and `init_state_id` are null - the HDF5 release doesn't carry them;
  join on `task_name`, which is derived from the file stem (the release's
  `problem_info.domain_name` is `"robosuite"`, not the suite, and its
  `problem_name` is a scene label - both kept out of the join path).
- `policy_type` is `"hdf5"` and `model` is `"libero_demo"` (demonstrations,
  not policy rollouts).
- `gripper_state` spans ~0.001-0.08 m, the same scale as the rollout data.
