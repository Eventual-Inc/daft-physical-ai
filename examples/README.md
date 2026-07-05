# Examples

Runnable physical-AI data recipes on Daft, numbered as the workflow a
researcher actually runs. Stages 01-06 are the standard curation journey
(read, inspect, transform, operate, label, write); 07-08 close the loop into
training and evaluation - and the failures mined in 08 are exactly what the
next curation pass in 02-03 goes looking for.

Every example is expected to run first-try on a clean environment against
public data. Directories marked *planned* are reserved stages with their
target scripts named in their READMEs.

| # | Stage | What it covers | Status |
|---|---|---|---|
| 01 | [Reading data](01_reading_data/) | Robot datasets into Daft: DROID today; LeRobot, raw EgoDex HDF5+video next | `droid_episode_index.py` |
| 02 | [Episode data](02_episode_data/) | Episode-level views over step rows: stats, indexes, dataset merging | planned |
| 03 | [Transforms](03_transforms/) | Deterministic NumPy features as Daft UDFs: pose geometry, quality checks, frame embeddings | planned |
| 04 | [Episode operations](04_episode_operations/) | Packaged robotics ops over episodes: hand tracking today; motion trim, pose queries next | [`hand_tracking/`](04_episode_operations/hand_tracking/) |
| 05 | [Inference](05_inference/) | Model-backed labeling with Daft AI functions: structured episode labels, VLM failure classification, semantic search | planned |
| 06 | [Writing data](06_writing_data/) | Training/analysis-ready outputs: canonical rollout parquet, LeRobot writers when Daft ships them | planned |
| 07 | [Training handoff](07_training_handoff/) | Curated Daft dataframes into `to_torch_dataloader` | planned |
| 08 | [Policy evals](08_policy_evals/) | Benchmark reproduction and failure mining over rollout parquet | `mine_failures.py` |

Deployment recipes live outside the numbered journey:

- [`modal/marimo_gallery/`](modal/marimo_gallery/) - the FastAPI + Marimo demo
  site running Daft, locally or hosted on Modal.

The importable package (`daft_physical_ai`) grows only where examples repeat a
helper: `episodes` (the canonical step-row schema every stage shares), `hands`,
and `evals` today.
