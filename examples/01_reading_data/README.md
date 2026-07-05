# 01 - Reading data

Get robot datasets into Daft. Daft's native readers do the heavy lifting;
these scripts show the minimal, copyable pattern per source.

- `droid_episode_index.py` - `daft.datasets.droid.raw()`: filter successful
  episodes and project an operational episode index, lazily.

Planned: `lerobot_episode_index.py` (native `daft.datasets.lerobot`, needs the
next Daft release), `egodex_raw_hdf5_video.py` (raw HDF5 + `video_file`).
