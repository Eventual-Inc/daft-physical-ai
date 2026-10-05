# PX OmniSharing with Daft

[PX OmniSharing DB](https://huggingface.co/datasets/paxini/Omnisharing_DB_SampleData) is PaXini's omnimodal embodied-AI dataset. It records force and tactile sensing alongside vision: each episode captures a person wearing an instrumented exoskeleton glove, with 15 tactile pads per hand, about a dozen RGB cameras, stereo RGBD cameras, hand joints and poses, optional object poses, audio, and a natural-language instruction.

`daft_physical_ai.datasets.omnisharing` catalogs OmniSharing HDF5 episodes lazily and reads only the modalities you select. Episodes in the public sample are 292 MB to 4.12 GB each, so the reader range-reads the datasets you ask for instead of downloading whole files. The public dataset is CC-BY-NC-SA 4.0, so the package does not download or redistribute it.

## Start with the episode catalog

```python
import numpy as np
from daft import col

from daft_physical_ai.datasets import omnisharing

episodes = omnisharing.raw("paxini/Omnisharing_DB_SampleData", stage="DF-2")
one = episodes.limit(1)
```

`raw()` parses filenames only and opens no HDF5 file. It exposes two identities:

- `episode_key` is the root-relative path without `.hdf5`/`.h5`. It is unique.
- `capture_key` is `episode_{index}_{time}_{room}_{personnel}` without the stage suffix, so the DF-1, DF-2, and DF-2R files of one capture share it.

Filter the catalog before reading modalities. Every materializer below returns a lazy DataFrame with a fixed schema, and planning one opens no file.

## Pipeline stages

The [OmniSharing toolkit](https://github.com/px-DataCollection/px_omnisharing_dataprocess_kit) emits four stages. `raw()` derives the `stage` column from the filename suffix.

| Stage | Format | Suffix | Support |
| ----- | ------ | ------ | ------- |
| DF-1 | HDF5 | *(none)* | Catalog only: encoder and tactile streams are unparsed |
| **DF-2** | HDF5 | `_glove` | Supported. Parsed glove data with bimanual and object poses |
| DF-2R | HDF5 | `_dh13`, `_mano`, ... | Supported. Retargeted to a dexterous hand model (`hand_model` column) |
| DF-3 | LeRobot v2.1 | *(none)* | Not read here. Convert to LeRobot v3, then use `daft.datasets.lerobot` |

The public sample release contains DF-2 only. DF-2R changes widths: joints become 17 wide and tactile 3750. The reader takes widths and sensor layouts from the file, so the same code reads both stages.

Materializing a DF-1 episode fails with `OmniSharing DF-1 episodes are catalog-only`.

## Episode layout

Each episode is one HDF5 file:

```text
episode_{index}_{HHMMSS}_{room}_{personnel}_glove.hdf5
└── dataset                          # attrs: generated_time, data_id
    ├── meta                         # attrs: vendor + free-form task labels
    ├── action                       # leads observation by one frame
    │   └── {left,right}hand
    │       ├── joints/data          # (n, 29) float32, attrs: joint_names
    │       └── handpose/data        # (n, 7) float32, group attrs: order
    └── observation
        ├── aligned_timestamp        # (n,) int64, microseconds: the reference clock
        ├── audio                    # (samples, 1) float64 PCM, attrs: samplerate, txt
        ├── image                    # attrs: checked_cam_name, e.g. "Camera4"
        │   ├── RGB_Camera{i}        # H.265/HEVC Annex-B, 1920x1200; ids are non-contiguous
        │   │   ├── data, timestamp
        │   │   └── intrinsics, extrinsics
        │   └── RGBD_{i}             # Matroska, 1280x720
        │       ├── {color,left,right}/data
        │       ├── timestamp, {left,right}_timestamp   # one clock per eye
        │       ├── aligned_depth    # (n, H, W) uint16, not on every RGBD camera
        │       └── inner_extrinsic  # JSON: left-eye-to-colour transform
        ├── {left,right}hand
        │   ├── joints/data          # (n, 29)
        │   ├── handpose/data        # (n, 7)
        │   └── tactile/data         # (n, 3465), attrs: sensor_names, sensor_lengths
        └── obj{i}/data              # (n, 17), optional
```

Each camera `data` dataset holds a whole encoded stream for the episode: 1.4-3.5 MB per stream in the pinned 207-frame sample episode.

The layout varies between releases, so inspect it instead of assuming it:

```python
layout = omnisharing.describe(one)
layout.select("episode_key", "h5path", "kind", "shape", "dtype", "chunks", "compression").show()
```

`describe()` is a thin composition over `daft.functions.hdf5_metadata`. For attributes at a known path, use `daft.functions.hdf5_attrs` directly. The reader itself uses only the public `hdf5_file`, `hdf5_metadata`, and `Hdf5File.open()` surface.

## Three things that are easy to get wrong

### 1. `episode_index` is not unique

The index restarts per capture group, so the same value appears under different `(room_id, personnel_id)` pairs. In the public sample, 191 `episode_index` values are duplicated. Join and deduplicate on `episode_key`, or on `capture_key` to match one capture across stages:

```python
# Both of these exist in part_01:
#   episode_1217_212953_93_110056_glove.hdf5
#   episode_1217_213630_115_110092_glove.hdf5
episodes.select("episode_key", "capture_key", "episode_index").show()
```

### 2. Hand-pose quaternions are `qw`-first

`handpose` is laid out `[x, y, z, qw, qx, qy, qz]`, exported as `omnisharing.HANDPOSE_ORDER`. `trajectory()` checks the `order` attribute and fails if a file disagrees. SciPy's `Rotation.from_quat` expects `[qx, qy, qz, qw]` and gives wrong rotations if fed this layout directly:

```python
row = omnisharing.trajectory(one, "observation/lefthand/handpose").to_pylist()[0]
pose = np.asarray(row["observation/lefthand/handpose"])
xyz, quat_wxyz = pose[:, :3], pose[:, 3:]
quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # the order Rotation.from_quat expects
```

### 3. Cameras do not share the observation clock

Hands and `aligned_timestamp` share one clock. RGB cameras run on their own clocks, start at different moments, and produce more frames; RGBD eyes keep separate `left_timestamp` and `right_timestamp` clocks. Camera ids are also non-contiguous. Aligning by frame index is wrong. Use `frames(align_cameras=...)`, which matches each observation frame to the nearest camera timestamp and reports the residual:

```python
omnisharing.cameras(one).select("camera", "stream", "codec", "clock_length").show(30)

per_frame = omnisharing.frames(
    one,
    fields="observation/lefthand/joints",
    align_cameras=["RGB_Camera0", ("RGBD_0", "left")],
)
per_frame.select("frame_index", "RGB_Camera0/frame_index", "RGB_Camera0/timestamp_residual_us").show()
```

In one sampled episode, observation frame 0 aligns to `RGB_Camera0` frame 10, because that camera began recording earlier. Index alignment would offset every frame by 10.

## Metadata and trajectories

```python
metadata = omnisharing.episode_metadata(one)
metadata.select("instruction", "frame_count", "audio_sample_rate", "camera_names", "task_labels_json").show()

trajectory = omnisharing.trajectory(
    one,
    fields=["observation/lefthand/joints", "observation/lefthand/handpose"],
)
trajectory.select("observation/lefthand/joints", "observation/lefthand/joint_names").show()
```

Trajectory selectors use DROID-style slash paths, and output tensors appear in request order. Each `.../joints` field also adds a `.../joint_names` list column read from the `joint_names` attribute. Joint order differs between DF-2 (29 glove joints, `J1J`, `J2J`, ...) and DF-2R (17 retargeted joints), so index joints by name, not position.

## Tactile, audio, and objects

```python
flat = omnisharing.tactile(one, sides="lefthand")
split = omnisharing.tactile(one, sides="lefthand", split_by_sensor=True)
clip = omnisharing.audio(one, mono=True, max_seconds=2.0)
tracked = omnisharing.objects(one)
```

The `(n, 3465)` tactile vector is 15 pads concatenated, with widths in the `sensor_lengths` attribute (`palm_sensor1` is 219 wide, a fingertip such as `M6L` 378). Flat mode returns `observation/lefthand/tactile` plus `observation/lefthand/tactile_layout`, a list of `{name, offset, width}` structs. Split mode returns `observation/lefthand/tactile_sensors`, a list whose structs also hold each pad's Float32 tensor. Both schemas are fixed, so heterogeneous widths need no schema probe.

The published layout calls the audio a compressed stream. It is raw Float64 PCM, and the spoken instruction is its `txt` attribute, which `episode_metadata()` returns as `instruction`. Audio columns are `audio/waveform`, `audio/sample_rate`, and `audio/sample_count`. `max_seconds` slices the dataset before reading; it is ignored when the file has no `samplerate`. Missing audio returns an empty Float64 tensor and null metadata.

Object tracks come from an optional pose-estimation stage, and many episodes have none. `objects()` returns `n_objects` and `objects`, a list with one struct per `obj{i}` group: index, name, id, layout, and the `(frames, 17)` Float32 pose. Episodes without objects get an empty list; there are no per-object columns to pad.

## Camera inventory, payloads, and frames

```python
inventory = omnisharing.cameras(one)
payloads = omnisharing.camera_payloads(one, ["RGB_Camera0", ("RGBD_0", "color")])
decoded = omnisharing.camera_frames(
    one,
    ["RGB_Camera0", ("RGBD_0", "color")],
    start_time=0,
    end_time=1,
    width=224,
    height=224,
    sample_interval_seconds=0.25,
    max_frames=4,
)
```

`cameras()` returns one row per stream with codec, clock, intrinsics, extrinsics, and `is_checked_camera`, which is true for the camera named by the episode's `checked_cam_name` attribute (stored as `"Camera4"`, matching `RGB_Camera4` exactly).

Camera selection is explicit. RGBD cameras need a `(camera, stream)` selector where the stream is `color`, `left`, or `right`. `camera_payloads()` returns the encoded bytes and sniffed codec of every requested stream from one file session.

`camera_frames()` uses the same eight-field list-of-struct schema as `daft.functions.video_frames`: `frame_index`, `frame_time`, `frame_time_base`, `frame_pts`, `frame_dts`, `frame_duration`, `is_key_frame`, and `data`. RGB payloads are HEVC Annex-B; RGBD payloads are Matroska. Each payload holds the whole stream, so `max_frames` caps decoded frames per stream after the time and key-frame filters. It defaults to 1; pass `None` to decode every selected frame. Missing optional streams return empty lists, while present unsupported or malformed payloads fail.

## Depth and stereo calibration

```python
depth = omnisharing.depth_frames(one, ["RGBD_0", "RGBD_1"], frame_indices=[0, 10])
stereo = omnisharing.stereo_extrinsics(one, ["RGBD_0", "RGBD_1"])
```

`aligned_depth` is UInt16 depth registered to the colour image. It is absent from the published layout and not present on every RGBD camera; in a sampled episode only `RGBD_0` had it. A frame is 720x1280, about 1.8 MB, so `frame_indices` defaults to `0` and each requested index is read separately, in request order. Missing depth returns an empty UInt16 tensor.

The depth unit is not declared in the files: `aligned_depth` has no attributes. Values are returned unchanged.

Episode lengths vary, so an index present in one episode may be missing from another. By default each episode reads the requested indices it has and lists them in `<camera>/depth_frame_indices`; check that column instead of assuming it matches `frame_indices`. Pass `strict=True` to fail on a short episode instead.

Stereo output uses `<camera>/left_to_color`, the 4x4 transform from the left eye into the colour frame, and `<camera>/calibration_date`. It is internal to one camera; the `extrinsics` from `cameras()` instead place the camera relative to a reference rig (`RGBD_0` for RGBD cameras, `RGB_Camera6` for RGB ones). Missing calibration is typed-empty. A present calibration is validated: malformed JSON, a missing matrix, or a non-4x4 matrix fails the read.

## Aligned frame rows

```python
per_frame = omnisharing.frames(
    one,
    fields=["observation/lefthand/joints", "action/lefthand/joints"],
    align_cameras=["RGB_Camera0", ("RGBD_0", "left")],
    include_columns=["episode_key", "capture_key"],
)
```

`frames()` requires an explicit signal whitelist, because one tactile row is 3465 floats. It broadcasts only `include_columns` and emits `frame_index`, `timestamp_us`, each requested vector, and `<stream>/frame_index` plus `<stream>/timestamp_residual_us`. Residuals are signed `camera_timestamp - timestamp_us`; ties choose the earlier frame. Per-eye clocks are used when present, with the RGBD camera clock as fallback.

`action[i]` is the state at `i + 1`, so `frames()` shifts action rows by one and repeats the final action:

| Frame | 0 | 1 | ... | n-2 | n-1 |
| --- | --- | --- | --- | --- | --- |
| observation | s0 | s1 | ... | s(n-2) | s(n-1) |
| action | s1 | s2 | ... | s(n-1) | s(n-1) |

## Errors

A read that cannot be completed, such as a DF-1 episode, a missing required dataset, or a layout that fails validation, raises `OmniSharingReadError` inside Daft's execution error. The message names the episode path and the cause:

```text
OmniSharingReadError: hf://datasets/.../episode_1203_..._glove.hdf5: KeyError: Required OmniSharing tactile dataset is missing: dataset/observation/lefthand/tactile/data
```

## Installation and licensing

```bash
pip install daft-physical-ai
```

The package depends on `daft[hdf5,video]>=0.7.20`, which brings h5py and PyAV. Review the [dataset card](https://huggingface.co/datasets/paxini/Omnisharing_DB_SampleData) and the [toolkit license](https://github.com/px-DataCollection/px_omnisharing_dataprocess_kit) before use; the toolkit's binary components are restricted to research and education.
