# Examples

Runnable physical-AI data recipes on Daft, numbered as the workflow a
researcher actually runs. Stages 01-06 are the standard curation journey
(read, inspect, transform, operate, label, write); 07-08 close the loop into
training and evaluation - and the failures mined in 08 are exactly what the
next curation pass goes looking for. The whole loop up to the training step
runs offline on the real data committed in this tree: 500 LIBERO-Spatial
demonstrations (stage 02) and 200 OpenVLA/VLA-JEPA benchmark rollouts
(stage 08).

Every example runs first-try on a clean environment. Directories marked
*planned* are reserved stages with their target scripts named in their READMEs.

| # | Stage | What it covers | Status |
|---|---|---|---|
| 01 | [Reading data](01_reading_data/) | Robot datasets into Daft: DROID and LeRobot v3 today; raw EgoDex HDF5+video next | `droid_episode_index.py` · `lerobot_episode_index.py` |
| 02 | [Episode data](02_episode_data/) | LIBERO demonstrations normalized into canonical step rows (full suite in-repo, ~4.7 MB); LeRobot session merging | `normalize_libero_demos.py` · `merge_lerobot_datasets.py` + [`data/`](02_episode_data/data/README.md) |
| 03 | [Transforms](03_transforms/) | Pose-feature tracks two ways on a public EgoDex sample: per-episode NumPy arrays, and the same rates as in-plan Daft window expressions | `pose_features_numpy.py` · `pose_rates_in_dag.py` |
| 04 | [Episode operations](04_episode_operations/) | Hand tracking on EgoDex; the motion-trim / no-noops audit; pose scenario queries stitched into segments | [`hand_tracking/`](04_episode_operations/hand_tracking/) · `motion_trim.py` · `pose_query_segments.py` |
| 05 | [Inference](05_inference/) | Model-backed labeling with Daft AI functions: structured episode labels, VLM failure classification, semantic search | planned |
| 06 | [Writing data](06_writing_data/) | Curated training artifacts as views: SFT step rows + manifest + preference pairs | `write_curated_manifest.py` |
| 07 | [Training handoff](07_training_handoff/) | Curated Daft dataframes streamed into PyTorch - `(batch, 7)` actions, `(batch, 8)` states | `curated_dataset_to_torch_dataloader.py` |
| 08 | [Policy evals](08_policy_evals/) | Benchmark reproduction, signal-based failure labeling, and the acquisition map over real LIBERO rollouts | `success_rates.py` · `compare_policies.py` · `validate_protocol.py` · `label_failures.py` · `acquisition_map.py` · `mine_failures.py` |

Deployment recipes live outside the numbered journey:

- [`modal/marimo_gallery/`](modal/marimo_gallery/) - the FastAPI + Marimo demo
  site running Daft, locally or hosted on Modal.

The importable package (`daft_physical_ai`) grows only where examples repeat a
helper: `episodes` (the canonical step-row schema every stage shares), `hands`,
and `evals` today.
