# 01 - Reading data

Get robot datasets into Daft. Daft's native readers do the heavy lifting;
these scripts show the minimal, copyable pattern per source.

- `droid_episode_index.py` - `daft.datasets.droid.raw()`: filter successful
  episodes and project an operational episode index, lazily.
- `lerobot_episode_index.py` - `daft.datasets.lerobot` (Daft >= 0.7.17):
  episode/task/frame views of a LeRobot v3 dataset, filtered without decoding
  any video.
- `egodex_raw_hdf5_video.py` - raw EgoDex episodes from a locally extracted
  release via `daft_physical_ai.datasets.egodex`: lazy `hdf5_file` /
  `video_file` access, no conversion step.
