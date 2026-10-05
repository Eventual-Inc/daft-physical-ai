# ABC-130k

[ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) is a robot
teleoperation dataset of 130,703 bimanual YAM episodes (more than 3,590 hours).
Each episode is one MCAP file with asynchronous state, action, gripper,
instruction, calibration, and compressed-video streams. Some episodes also have
a separate MCAP of free-form subtask annotations.

`daft_physical_ai.datasets.abc` reads it in separate stages, so you can bound the
work before touching large episode files:

| Function | Reads |
| --- | --- |
| `raw()` | Object listing only: paths and sizes |
| `metadata()` | MCAP header, footer, and summary (range reads) |
| `messages()` | Message payloads for chosen topics and time window |
| `annotations()` | `/subtask-annotation` events, decoded |
| `camera_frames()` | Foxglove `CompressedVideo` topics, decoded to RGB |

## Setup

```bash
pip install "daft-physical-ai[abc]"
```

The `abc` extra adds the `mcap` reader and `huggingface_hub`, which `raw()` uses
to list the Hub repository. PyAV, used by `camera_frames()`, is already a
dependency.

The dataset is gated: accept the conditions on the
[dataset page](https://huggingface.co/datasets/XDOF/ABC-130k) (approval is
automatic). Then either set `HF_TOKEN` or run `hf auth login`. When the path is
`hf://` and `io_config` carries no token, the token is picked up from those
sources. Pass `io_config` yourself for other stores or explicit credentials:

```python
import os

from daft.io import HuggingFaceConfig, IOConfig

io_config = IOConfig(hf=HuggingFaceConfig(token=os.environ["HF_TOKEN"], use_xet=True))
```

## Episode layout

```text
data/
|-- train/
|   `-- <task_slug>/
|       `-- episode_<uuid>/
|           |-- episode.mcap
|           `-- annotation.mcap  # optional
`-- val/
    `-- <task_slug>/
        `-- episode_<uuid>/
            `-- episode.mcap
```

Some format documentation names task directories `task=<task_slug>`. `raw()`
accepts both forms.

## Catalog: `raw()`

```python
from daft_physical_ai.datasets import abc

episodes = abc.raw(split="train", tasks="clip_the_socks_to_the_hanger")
sample = episodes.sort("episode_size").limit(4)
sample.select("task_slug", "episode_id", "episode_size", "annotated").show()
```

`path` defaults to `hf://datasets/XDOF/ABC-130k`. Append `@<revision>` to pin a
commit. A local mirror or object-store root also works. Pass `split` and
`tasks` to `raw()` so the listing itself is narrow; a later `.where()` does not
shrink the listing.

On Hugging Face, `raw()` lists through the Hub's paginated recursive-tree API,
not one request per episode directory, and does the listing when you call it.
Other stores use Daft's lazy glob, which raises when executed if nothing
matches. No MCAP file is opened either way.

| Column | Description |
| --- | --- |
| `split` | `train` or `val`, from the path |
| `task_slug` | Task directory name |
| `episode_id` | Episode UUID |
| `episode_dir` | Episode directory |
| `episode_path`, `episode_size` | `episode.mcap` path and size in bytes |
| `annotation_path`, `annotation_size` | `annotation.mcap` path and size, or null |
| `annotated` | Whether an annotation MCAP exists |
| `episode_mcap`, `annotation_mcap` | Lazy `daft.File` references (annotation nullable) |

## Summaries: `metadata()`

```python
abc.metadata(sample).select(
    "episode_id", "task_name", "message_count", "message_start_time", "topics", "video_topics"
).show()
```

For each catalog row, `metadata()` range-reads the MCAP summary with the `mcap`
package. Message payloads are not downloaded. It keeps the catalog columns and
adds:

| Column | Description |
| --- | --- |
| `session_id`, `operator_id`, `task_name` | From the episode metadata record, when present |
| `duration_seconds` | Reported duration, or derived from message times |
| `message_count`, `chunk_count` | Indexed MCAP statistics |
| `message_start_time`, `message_end_time` | Unix nanoseconds |
| `topics`, `video_topics` | All channel topics; topics with `foxglove.CompressedVideo` schema |
| `indexed` | Whether chunk indexes are present |
| `episode_metadata_json` | The raw metadata record name and key/value map |

ABC releases have used both an `episode-metadata` record and a legacy
`session-metadata` record with different key spellings. Both are handled. If an
MCAP has no summary section, the whole file is read in sequence and `indexed`
is `False`.

## Messages: `messages()`

```python
one = episodes.sort("episode_size").limit(1)
start = abc.metadata(one).select("message_start_time").to_pydict()["message_start_time"][0]

states = abc.messages(
    one,
    topics=["/left-arm-state", "/right-arm-state"],
    start_time=start,
    end_time=start + 2_000_000_000,
)
states.select("episode_id", "topic", "log_time", "data").show()
```

`messages()` collects the catalog's paths, so filter and limit the catalog
first. Each file then becomes a `daft.read_mcap` scan, with the topic and time
filters passed to the reader. `start_time` is inclusive and `end_time` is
exclusive, both in MCAP `log_time` units (Unix nanoseconds). Output has
`source_path`, `topic`, `log_time`, `publish_time`, `sequence` (all `int64`),
raw protobuf `data` as binary, and `split`, `task_slug`, `episode_id`,
`episode_dir`, `file_kind`.

The default topics are the eight arm and gripper streams. They exclude cameras,
so a bare call never reads video:

| Topic | Meaning |
| --- | --- |
| `/left-arm-state`, `/right-arm-state` | Observed joint state and end-effector pose |
| `/left-arm-action`, `/right-arm-action` | Commanded joint position and pose |
| `/left-ee-state`, `/right-ee-state` | Observed gripper aperture (`0` closed, `1` open) |
| `/left-ee-action`, `/right-ee-action` | Commanded gripper aperture |

Streams run on independent clocks. `messages()` does not resample, join, or
align them, so don't assume matching row counts or indexes. Align on `log_time`.

Released Daft (up to v0.7.25) reads MCAP in Python. It has no `source_path`
column and returns `data` as `str(bytes)`. Daft main's unreleased native reader
adds `source_path` and uses unsigned integers and binary `data`. `messages()`
converts both to the schema above, so code written against it keeps working
when the native reader ships.

## Annotations: `annotations()`

```python
labels = abc.annotations(episodes.where(episodes["annotated"]).limit(4))
labels.select("episode_id", "log_time", "timestamp_ns", "label").show()
```

Reads `/subtask-annotation` from `annotation.mcap` and decodes each protobuf
payload into `timestamp_ns` and `label`. Each event starts a subtask that runs
until the next event, or to the end of the episode. Labels are free-form text.
Episodes without an annotation file give no rows. Malformed payloads give null
fields.

## Camera frames: `camera_frames()`

```python
frames = abc.camera_frames(
    one,
    cameras=["top", "left_wrist"],
    start_time=start,
    end_time=start + 1_000_000_000,
    width=224,
    height=224,
)
frames.select("episode_id", "camera", "topic", "format", "log_time", "is_key_frame", "data").show()
```

| Alias | Topics |
| --- | --- |
| `top` | Every top stream present: `/top-camera`, `/top-left-camera`, `/top-right-camera` |
| `top_mono` | `/top-camera` |
| `top_left` | `/top-left-camera` |
| `top_right` | `/top-right-camera` |
| `left_wrist` | `/left-wrist-camera` |
| `right_wrist` | `/right-wrist-camera` |

Each (episode, topic) pair gets its own PyAV decoder. The codec comes from the
`format` field inside each `CompressedVideo` message, not from the camera name;
ABC has both H.264 and H.265 streams. A topic is read from its start up to
`end_time`. For H.264/H.265, frames before the last keyframe preceding
`start_time` are skipped without decoding. Frames before `start_time` are
dropped from the output.

The time window bounds decoding but not reading: a window late in a long
episode still reads that topic's earlier chunks. Keep the catalog small. Output
columns: episode identity, `camera`, `topic`, `log_time`, `publish_time`,
`sequence`, `timestamp_ns`, `frame_id`, `format`, `is_key_frame`, `width`,
`height`, and RGB image `data`.

## Recommended order

1. Pass `split` and `tasks` to `raw()`.
2. Choose a bounded sample using `episode_size` and `annotated`.
3. Call `metadata()` on that sample when you need topics or time bounds.
4. Pass explicit topics and time windows to `messages()` and `annotations()`.
5. Decode cameras last, with a time window.

See the [ABC-130k format specification](https://huggingface.co/datasets/XDOF/ABC-130k/blob/main/docs/YAM_DATA_FORMAT.md)
for the message schemas, and [`examples/abc_episode_messages.py`](../examples/abc_episode_messages.py)
for a runnable script.
