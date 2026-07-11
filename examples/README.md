# Examples

Runnable physical-AI data recipes on Daft, numbered as the workflow a
researcher actually runs: read datasets, inspect episode data, transform,
run episode operations, label with inference, write outputs, hand off to
training, analyze policy evals. Stages land incrementally; directories
marked *planned* are reserved, with their target scripts named here.

| # | Stage | What it covers | Status |
|---|---|---|---|
| 01 | [Reading data](01_reading_data/) | Robot datasets into Daft: DROID metadata, LeRobot v3 episode/task/frame views, raw EgoDex HDF5+video | `droid_episode_index.py` · `lerobot_episode_index.py` · `egodex_raw_hdf5_video.py` |
| 02 | [Episode data](02_episode_data/) | Episode-level views and dataset combination | `merge_lerobot_datasets.py`; normalization lands with the episode contract |
| 03 | Transforms | Deterministic NumPy features as episode passes and in-plan expressions | planned |
| 04 | [Episode operations](04_episode_operations/) | Packaged robotics ops over episodes | [`hand_tracking/`](04_episode_operations/hand_tracking/); motion trim and pose queries planned |
| 05 | Inference | Model-backed labeling with Daft AI functions | planned |
| 06 | Writing data | Curated training artifacts as views | planned |
| 07 | Training handoff | Curated dataframes into `to_torch_dataloader` | planned |
| 08 | Policy evals | Benchmark reproduction and failure mining over rollout parquet | planned |

Every landed example runs first-try on a clean environment against public
data (EgoDex raw reading expects a locally extracted release - see its
docstring).
