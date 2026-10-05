# EgoDex (raw HDF5 + MP4 release)

For Apple's original EgoDex HDF5+MP4 release, use the extension's lazy reader:

```python
from daft_physical_ai.datasets import egodex

episodes = egodex.raw("/data/egodex", tasks="fold_towel").limit(2)
poses = egodex.trajectory(episodes, fields=["transforms/leftHand", "transforms/rightHand"])
frames = egodex.camera_frames(poses, width=224, height=224, sample_interval_seconds=1.0)
```

EgoDex is CC-BY-NC-ND, so the package does not download, extract, or redistribute it. Download and extract the archives from the [official EgoDex repository](https://github.com/apple/ml-egodex), then point `raw()` at your copy. See [the runnable example](../examples/egodex_raw_hdf5_video.py).
