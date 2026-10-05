# REASSEMBLE

[REASSEMBLE](https://tuwien-asl.github.io/REASSEMBLE_page/) (RSS 2025,
[arXiv:2502.05086](https://arxiv.org/abs/2502.05086)) is a multimodal dataset of
contact-rich assembly and disassembly on the NIST Assembly Task Board 1,
recorded with a Franka arm: 149 recording sessions, 4,551 action demonstrations
(4,035 successful), 781 minutes. Each recording has three RGB cameras, an event
camera, three microphones, proprioception, force/torque, and two levels of
action-segment labels.

`daft_physical_ai.datasets.reassemble` reads the original HDF5 release
directly. Each function reads one part of the file:

| Function | Reads | Output rows |
| --- | --- | --- |
| `raw()` | File paths only | One per recording |
| `segments()` | `segments_info` (high- and low-level labels) | One per action or skill |
| `robot_state()` | `robot_state/*` + `timestamps/*`, native rates | One per recording |
| `audio()` | `*_audio` MP3 bitstreams, decoded with PyAV | One per microphone |
| `events()` | `events` + `timestamps/events`, windowed | One per recording |
| `camera_frames()` | Embedded MP4 blobs, decoded with PyAV | One per frame |
| `remote_catalog()` | The TU Wien zip's central directory | One per recording |
| `download()` | Selected zip members, over HTTP range requests | - |
| `lerobot_sidecars()` | The LeRobot port's `audio/` and `events/` side files | One per episode |

No extra is needed: `h5py` (`daft[hdf5]`) and PyAV (`daft[video]`) are already
dependencies.

## Getting the data

The authors publish the release on TU Wien research data
([record 0ewrv-8cb44](https://researchdata.tuwien.ac.at/records/0ewrv-8cb44),
DOI [10.48436/0ewrv-8cb44](https://doi.org/10.48436/0ewrv-8cb44)) under
**CC-BY-4.0**. The record has four files:

| File | Size | Contents |
| --- | --- | --- |
| `data.zip` | 58.9 GB | 149 `<YYYY-MM-DD-HH-MM-SS>.h5` recordings |
| `poses.zip` | 89 KB | Camera and board poses per recording (JSON) |
| `splits.zip` | 1 KB | `train_split1.txt` (111) and `test_split1.txt` (37) |
| `README.txt` | 8 KB | Layout notes and known issues |

Extracted, the recordings total 264 GB: 102 MB to 3.6 GB each, median 1.7 GB.

**You cannot read the HDF5 files in place over HTTPS.** The server supports
HTTP range requests (`Accept-Ranges: bytes`, `206 Partial Content`), but every
`.h5` member of `data.zip` is deflate-compressed, so there is no byte range to
seek into. Extract recordings first, then point `raw()` at the extracted copies
on local disk or in any store Daft reads (S3, GCS, HTTP), where range reads
work as usual.

You do not have to download the whole archive. Range requests do let
`remote_catalog()` read the zip's central directory, and `download()` stream
single members out of it. Each member is fetched with one range request,
inflated, and CRC-checked:

```python
from daft_physical_ai.datasets import reassemble

catalog = reassemble.remote_catalog()  # recording, size_bytes, compressed_bytes
catalog.sort("compressed_bytes").show(3)

reassemble.download(["2025-01-11-14-43-37", "2025-01-10-15-39-56"], "/data/reassemble")
```

The smallest recording, `2025-01-11-14-43-37`, is a 24 MB transfer (102 MB
extracted) and took 7.5 s in testing.

## HDF5 layout

Checked against real files and the authors' loader
([`REASSEMBLE/io.py`](https://github.com/TUWIEN-ASL/REASSEMBLE/blob/main/REASSEMBLE/io.py)):

```text
<YYYY-MM-DD-HH-MM-SS>.h5
|-- hama1, hama2, hand              scalar |V blob: MP4 (H.264), 640x480 @ 30 fps
|-- capture_node-camera-image      scalar |V blob: MP4 render of the event camera, 346x260, ~6 fps
|-- hama1_audio, hama2_audio       (N,) int64: MP3 bytes, one byte per element (16 kHz stereo)
|-- hand_audio                     (N,) int64: MP3 bytes (48 kHz mono)
|-- events                         (N, 3) int64: x, y, polarity
|-- robot_state/
|   |-- measured_force, measured_torque              (N, 3)  ~1000 Hz
|   |-- joint_positions, joint_velocities, joint_efforts  (N, 7)  ~970 Hz
|   |-- gripper_positions, gripper_velocities, gripper_efforts  (N, 2)  ~970 Hz
|   |-- pose                                         (N, 7)  x, y, z, qw, qx, qy, qz  ~580 Hz
|   |-- velocity                                     (N, 6)  ~580 Hz
|   `-- compensated_base_force, compensated_base_torque  (N, 3)  ~465 Hz
|-- timestamps/<stream>            (N,) float64 Unix seconds, one array per stream above
`-- segments_info/<i>/             start, end (Unix s), success, text, index
    `-- low_level/<j>/             start, end, success, text
```

The release README differs from the files in a few places. The files use the
spelling `compensated_base_torque` (the README has `compenseted_`).
`measured_torque` has 3 columns, not 7. The event-camera video is named
`capture_node-camera-image`. The skill group is `low_level`, lowercase. The
gripper velocity and effort streams are not listed in the README. Audio is
stored as int64 arrays of MP3 bytes, not as byte strings. Recordings with a
missing camera omit that dataset entirely: `2025-01-10-16-17-40` has no `hand`.

Every stream keeps its own clock. The release does not resample or synchronize
anything, and audio has no per-sample timestamps.

## Usage

```python
from daft import col

from daft_physical_ai.datasets import reassemble

episodes = reassemble.raw("/data/reassemble", recordings="2025-01-11-14-43-37")

# Hierarchical labels. Times are Unix seconds on the same clock as every stream.
actions = reassemble.segments(episodes)  # segment_index, text, start, end, duration, success, low_level
skills = reassemble.segments(episodes, level="low")  # segment_text, skill_index, text, ...

# Force/torque and joints at their native rates: (N, k) float64 tensors plus timestamps.
state = reassemble.robot_state(episodes, fields=["measured_force", "measured_torque", "joint_positions"])

# Windows take Unix seconds, so segment bounds plug in directly.
pick = actions.where(col("text").startswith("Pick")).limit(1).to_pydict()
window = {"start_time": pick["start"][0], "end_time": pick["end"][0]}
ft = reassemble.robot_state(episodes, fields=["measured_force"], **window)
ev = reassemble.events(episodes, **window)  # num_events, events (N, 3), timestamps (N,)
frames = reassemble.camera_frames(episodes, ["hand", "hama1"], width=224, height=168, **window)

# Audio: sample_rate, channels, num_samples, samples (channels, N) float32.
sound = reassemble.audio(episodes)
mp3 = reassemble.audio(episodes, "hand", decode=False)  # raw MP3 bytes
```

`robot_state()` reads only the windowed slice of each stream. `events()` finds
its window by bisecting `timestamps/events` (about 22 element reads per bound on
a 2.5M-event recording), then reads only that slice: a 0.5 s window is about
25k events.
`camera_frames()` seeks to the keyframe before `start_time` and stops at
`end_time`. Each frame carries its own timestamp from `timestamps/<camera>`.

[`examples/reassemble_contact_segments.py`](../examples/reassemble_contact_segments.py)
runs all of this on one recording. It joins `segments()` with `robot_state()`
to get the force/torque sample count and peak force per action, and downloads
the smallest recording first if the directory is empty.

## The LeRobot port

[`robot-lev/reassemble`](https://huggingface.co/datasets/robot-lev/reassemble)
is a LeRobot v3.0 conversion of the same 149 recordings: 149 episodes,
1,433,406 frames at 30 fps, cameras `hand`, `hama1`, `hama2` (AV1, 480x640)
and `event_cam` (AV1, 260x346), 86 GB in total. Daft reads the LeRobot part
natively:

```python
import os

from daft import col
from daft.datasets import lerobot
from daft.io import HuggingFaceConfig, IOConfig

io_config = IOConfig(hf=HuggingFaceConfig(token=os.environ["HF_TOKEN"]))

frames = lerobot.read("robot-lev/reassemble", io_config=io_config)  # one row per 30 fps frame
episodes = lerobot.read_episodes("robot-lev/reassemble", io_config=io_config)  # length, tasks, video offsets
tasks = lerobot.read_tasks("robot-lev/reassemble", io_config=io_config)  # 69 high-level action strings

clip = lerobot.read(
    "robot-lev/reassemble", io_config=io_config, load_video_frames="observation.images.hand"
).where(col("episode_index") == 21)
```

The dataset is public, but without a token the Hub's tree API returned
`429 Too Many Requests` during testing. Daft's glob does not accept an
`@revision` in `hf://` paths, so `daft.datasets.lerobot` reads the default
branch.

The port keeps two modalities outside the frame stream, in side folders that
`daft.datasets.lerobot` does not read:

```text
audio/episode_XXXXXX/{hama1,hama2,hand}.wav
events/episode_XXXXXX.npz      # events (N, 3) int64, timestamps (N,) float64
meta/splits.json               # episode -> original recording name and split
```

`lerobot_sidecars()` catalogs them, one row per episode: `episode_index`,
`recording`, `split`, three lazy WAV file columns, and a lazy `events_file`.
`audio()` and `events()` accept that catalog and return the same schemas as
for the HDF5 release. `recording` joins the port to `raw()`.

```python
sidecars = reassemble.lerobot_sidecars()  # hf://datasets/robot-lev/reassemble
episode = sidecars.where(col("recording") == "2025-01-11-14-43-37")  # episode_index 21
sound = reassemble.audio(episode)
ev = reassemble.events(episode, start_time=1736603020.6, end_time=1736603021.1)
```

**The port's WAV files are not playable audio.** The conversion copied each
MP3 *byte* into one 32-bit PCM sample under a 16 kHz mono header. Every
sample value lies between 0 and 255, and played back the files are noise.
Durations are also wrong: `hand.wav` for episode 21 claims 11.7 s for a 23.4 s
recording. The bytes are intact. `audio()` recovers the original MP3 from the
WAV samples and decodes it. On episodes 21 and 77 the recovered MP3 matched
the HDF5 release byte for byte; on episode 21 it also decoded to identical
samples. If a future revision of the port stores real PCM, `audio()` reads it
as PCM. Daft's own `audio_file()` would read the current WAVs as noise.

The port's event NPZ files match the HDF5 `events` and `timestamps/events`
arrays exactly (checked on episode 21, 2,551,295 events). An NPZ is
compressed, so `events()` reads it whole before applying a time window.

## Original release vs the LeRobot port

The port resamples every sensor onto a 30 fps grid by nearest neighbour. For
recording `2025-01-11-14-43-37` (port episode 21), the port has 696 frames,
while the HDF5 has 23,240 `measured_force` samples (33x more), 22,549 joint
samples, and 13,401 pose samples.

| | Original HDF5 | LeRobot port |
| --- | --- | --- |
| Force/torque | ~1000 Hz measured, ~465 Hz base | 30 Hz, nearest-neighbour |
| Joints, gripper | ~970 Hz | 30 Hz |
| EE pose, velocity | ~580 Hz | 30 Hz |
| Joint velocity and effort | Yes | Yes, inside the 36-D `observation.state` |
| Gripper velocity and effort | Yes | Dropped (only gripper position) |
| Timestamps | Absolute Unix seconds, per stream | Episode-relative float32 on one 30 fps clock |
| High-level labels | Text, start, end, success | Per-frame `task_index`, `segment.index`, `segment.success` |
| Low-level skills (Grasp, Lift, Align, ...) | Text, start, end, success | Dropped |
| RGB video | H.264, 30 fps | Re-encoded to AV1 (crf 30) |
| Event camera render | ~6 fps | Resampled to 30 fps |
| Raw events | Yes | Yes, lossless (`events/*.npz`) |
| Audio | MP3: 16 kHz stereo, 48 kHz mono | Mislabelled WAVs (MP3 bytes; recoverable with `audio()`) |
| Missing camera | Dataset absent | Black frames |
| Camera and board poses | `poses.zip` | Not included |
| Download size | 58.9 GB zip (264 GB extracted) | 86 GB |

Use the port for 30 fps policy training on frames, joints, and actions. Use the
original release when the work depends on force/torque or proprioceptive
dynamics above 15 Hz (the 30 fps Nyquist limit), on low-level skill
segmentation, on exact per-stream timing, or on first-generation video.

## Known issues in the release

From the TU Wien record:

| Recording | Issue |
| --- | --- |
| `2025-01-10-15-28-50` | Hand camera missing at the beginning |
| `2025-01-10-16-17-40` | Hand camera missing (no `hand` dataset) |
| `2025-01-10-17-10-38` | Hand camera missing at the beginning |
| `2025-01-10-17-54-09` | No empty action at the beginning |
| `2025-01-11-14-22-09` | No empty action at the beginning |
| `2025-01-11-14-45-48` | F/T not valid for the last action |
| `2025-01-11-15-27-19` | F/T not valid for the last action |
| `2025-01-11-15-35-08` | F/T not valid for the last action |
| `2025-01-13-11-16-17` | Gripper broke for the last action |
| `2025-01-13-11-18-57` | Pose not available for the last action |

## Citation

```bibtex
@misc{sliwowski2025reassemble,
  title         = {REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly},
  author        = {Sliwowski, Daniel Jan and Jadav, Shail and Stanovcic, Sergej and Orbik, J\k{e}drzej and Heidersberger, Johannes and Lee, Dongheui},
  year          = {2025},
  eprint        = {2502.05086},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48436/0ewrv-8cb44},
  url           = {https://researchdata.tuwien.ac.at/records/0ewrv-8cb44}
}
```
