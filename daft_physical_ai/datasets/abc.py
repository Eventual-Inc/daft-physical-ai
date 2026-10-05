"""Lazy access to `ABC-130k <https://huggingface.co/datasets/XDOF/ABC-130k>`_ robot episodes stored as MCAP.

ABC-130k is a gated Hugging Face dataset (accept the conditions on the dataset
page; approval is automatic) of bimanual YAM teleoperation episodes. Each
episode directory holds an ``episode.mcap`` with asynchronous state, action,
gripper, and compressed-video streams, plus an optional ``annotation.mcap`` with
free-form subtask labels::

    data/{train,val}/<task_slug>/episode_<uuid>/{episode.mcap,annotation.mcap}

The API keeps the expensive stages separate so you can bound work before
touching large episode files: :func:`raw` lists objects (paths and sizes only),
:func:`metadata` range-reads MCAP summaries, :func:`messages` and
:func:`annotations` read message payloads through :func:`daft.read_mcap`, and
:func:`camera_frames` decodes Foxglove ``CompressedVideo`` streams with PyAV.

Requires the ``abc`` extra (``pip install 'daft-physical-ai[abc]'``), which adds
the ``mcap`` reader and ``huggingface_hub`` for catalog listing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import daft
from daft.datatype import DataType
from daft.expressions import col, lit
from daft.functions import file, regexp_extract, regexp_replace, when

from daft_physical_ai.datasets import _mcap

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.expressions import Expression
    from daft.io import IOConfig

HF_DATASET = "hf://datasets/XDOF/ABC-130k"

# The eight arm/gripper state and action streams. The default excludes camera,
# calibration, and instruction topics so a bare messages() call never reads video.
DEFAULT_MESSAGE_TOPICS: tuple[str, ...] = (
    "/left-arm-state",
    "/right-arm-state",
    "/left-arm-action",
    "/right-arm-action",
    "/left-ee-state",
    "/right-ee-state",
    "/left-ee-action",
    "/right-ee-action",
)

ANNOTATION_TOPIC = "/subtask-annotation"

CAMERA_TOPICS: dict[str, tuple[str, ...]] = {
    "top": ("/top-camera", "/top-left-camera", "/top-right-camera"),
    "top_mono": ("/top-camera",),
    "top_left": ("/top-left-camera",),
    "top_right": ("/top-right-camera",),
    "left_wrist": ("/left-wrist-camera",),
    "right_wrist": ("/right-wrist-camera",),
}

_TOPIC_TO_CAMERA: dict[str, str] = {
    "/top-camera": "top_mono",
    "/top-left-camera": "top_left",
    "/top-right-camera": "top_right",
    "/left-wrist-camera": "left_wrist",
    "/right-wrist-camera": "right_wrist",
}

_VIDEO_SCHEMA = "foxglove.CompressedVideo"
# Foxglove CompressedVideo.format values -> FFmpeg decoder names.
_VIDEO_CODECS: dict[str, str] = {"h264": "h264", "h265": "hevc", "hevc": "hevc", "vp9": "vp9", "av1": "av1"}

_CATALOG_COLUMNS: tuple[str, ...] = (
    "split",
    "task_slug",
    "episode_id",
    "episode_dir",
    "episode_path",
    "episode_size",
    "annotation_path",
    "annotation_size",
    "annotated",
    "episode_mcap",
    "annotation_mcap",
)

_IDENTITY_COLUMNS: tuple[str, ...] = ("split", "task_slug", "episode_id", "episode_dir")

_SPLIT_PATTERN = r"/data/(train|val)/"
_TASK_PATTERN = r"/data/(?:train|val)/(?:task=)?([^/]+)/episode_"
_EPISODE_ID_PATTERN = r"/episode_([^/]+)/"
_EPISODE_DIR_PATTERN = r"/[^/]+\.mcap$"
_FILE_KIND_PATTERN = r"/(episode|annotation)\.mcap$"

_METADATA_DTYPE = DataType.struct(
    {
        "session_id": DataType.string(),
        "operator_id": DataType.string(),
        "task_name": DataType.string(),
        "duration_seconds": DataType.float64(),
        "message_count": DataType.int64(),
        "message_start_time": DataType.int64(),
        "message_end_time": DataType.int64(),
        "chunk_count": DataType.int64(),
        "topics": DataType.list(DataType.string()),
        "video_topics": DataType.list(DataType.string()),
        "indexed": DataType.bool(),
        "episode_metadata_json": DataType.string(),
    }
)

_ANNOTATION_DTYPE = DataType.struct({"timestamp_ns": DataType.int64(), "label": DataType.string()})

_FRAME_DTYPE = DataType.struct(
    {
        "log_time": DataType.int64(),
        "publish_time": DataType.int64(),
        "sequence": DataType.int64(),
        "timestamp_ns": DataType.int64(),
        "frame_id": DataType.string(),
        "format": DataType.string(),
        "is_key_frame": DataType.bool(),
        "width": DataType.int64(),
        "height": DataType.int64(),
        "data": DataType.image("RGB"),
    }
)


def _require_mcap() -> Any:
    return _mcap.require_mcap("ABC-130k", "abc")


# Shared MCAP helpers (see _mcap.py), kept under their original private names.
_normalize_names = _mcap.normalize_names
_normalize_messages = _mcap.normalize_messages
_open_mcap = _mcap.open_mcap
_resolve_hf_io_config = _mcap.resolve_hf_io_config


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def _hf_listing(
    root: str,
    *,
    split: str | None,
    tasks: tuple[str, ...] | None,
    include_annotations: bool,
    io_config: IOConfig | None,
) -> DataFrame:
    """List ABC objects with the Hub's paginated recursive-tree endpoint.

    The generic ``hf://`` glob walks each episode directory separately, which is
    slow and hits Hub rate limits on ABC's deep layout.
    """
    _mcap.require_hf_hub("ABC-130k", "abc")
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.utils import EntryNotFoundError

    repo_id, revision = _mcap.parse_hf_root(root, example=HF_DATASET)
    api = HfApi(token=_mcap.hf_token(io_config), library_name=_mcap.LIBRARY_NAME)

    if tasks is None:
        root_groups = [(f"data/{split}" if split is not None else "data",)]
    else:
        splits = (split,) if split is not None else ("train", "val")
        root_groups = [(f"data/{s}/{task}", f"data/{s}/task={task}") for s in splits for task in tasks]

    wanted = {"episode.mcap", "annotation.mcap"} if include_annotations else {"episode.mcap"}
    paths: list[str] = []
    sizes: list[int] = []
    for search_roots in root_groups:
        for search_root in search_roots:
            matched = False
            try:
                for entry in api.list_repo_tree(
                    repo_id,
                    path_in_repo=search_root,
                    recursive=True,
                    revision=revision,
                    repo_type="dataset",
                ):
                    if isinstance(entry, RepoFile) and entry.path.rsplit("/", 1)[-1] in wanted:
                        paths.append(f"{root}/{entry.path}")
                        sizes.append(entry.size)
                        matched = True
            except EntryNotFoundError:
                pass
            if matched:
                break

    if not paths:
        return _mcap.empty_listing()
    return daft.from_pydict({"path": paths, "size": sizes})


def _glob_listing(
    root: str,
    *,
    split: str | None,
    tasks: tuple[str, ...] | None,
    include_annotations: bool,
    io_config: IOConfig | None,
) -> DataFrame:
    filename = "*.mcap" if include_annotations else "episode.mcap"
    split_dir = split or "*"
    if tasks is None:
        patterns: str | list[str] = f"{root}/data/{split_dir}/**/{filename}"
    else:
        # ``*<task>`` matches both ``<task>`` and the ``task=<task>`` layout in one
        # listing; raw() filters on the parsed slug afterwards for an exact match.
        patterns = [f"{root}/data/{split_dir}/*{task}/**/{filename}" for task in tasks]
    return daft.from_glob_path(patterns, io_config=io_config)


def raw(
    path: str = HF_DATASET,
    *,
    split: Literal["train", "val"] | None = None,
    tasks: str | Sequence[str] | None = None,
    include_annotations: bool = True,
    io_config: IOConfig | None = None,
) -> DataFrame:
    r"""Catalog ABC-130k episodes as an episode-level DataFrame.

    Lists objects and parses paths only; no MCAP is opened. ``path`` defaults to
    the gated Hugging Face dataset (append ``@<revision>`` to pin one), and may
    also be a local mirror or another object store. On Hugging Face the listing
    goes through the Hub's recursive-tree API and is consumed when ``raw()`` is
    called; other stores use Daft's lazy glob, which raises at execution when
    nothing matches. Later transforms are lazy either way.

    Args:
        path: Dataset root.
        split: Optional ``"train"`` or ``"val"`` restriction.
        tasks: Optional task slug or slugs. They narrow the listing prefix, so
            unrelated tasks are not traversed. ``task=<slug>`` is accepted.
        include_annotations: Also list the optional ``annotation.mcap`` objects.
        io_config: Storage configuration. For ``hf://`` paths without a token,
            ``HF_TOKEN`` or the ``hf auth login`` credential is used.

    Returns:
        One row per episode: ``split``, ``task_slug``, ``episode_id``,
        ``episode_dir``, ``episode_path``, ``episode_size``, ``annotation_path``,
        ``annotation_size``, ``annotated``, and lazy ``daft.File`` references
        ``episode_mcap`` and (nullable) ``annotation_mcap``.
    """
    if split not in (None, "train", "val"):
        raise ValueError("split must be one of: 'train', 'val', or None")
    selected_tasks = _normalize_names(tasks, name="tasks")
    if selected_tasks is not None:
        if any("/" in task or "*" in task for task in selected_tasks):
            raise ValueError("tasks must be exact task slugs, without '/' or glob wildcards")
        selected_tasks = tuple(dict.fromkeys(task.removeprefix("task=") for task in selected_tasks))

    root = path.rstrip("/")
    io_config = _resolve_hf_io_config(io_config, (root,))
    listing = _hf_listing if root.startswith("hf://") else _glob_listing
    objects = listing(
        root, split=split, tasks=selected_tasks, include_annotations=include_annotations, io_config=io_config
    )

    def pick(kind: str, column: str) -> Expression:
        return when(col("__kind") == kind, col(column)).otherwise(lit(None)).any_value(ignore_nulls=True)

    catalog = objects.select(
        "path",
        "size",
        regexp_extract(col("path"), _FILE_KIND_PATTERN, 1).alias("__kind"),
        regexp_extract(col("path"), _SPLIT_PATTERN, 1).alias("split"),
        regexp_extract(col("path"), _TASK_PATTERN, 1).alias("task_slug"),
        regexp_extract(col("path"), _EPISODE_ID_PATTERN, 1).alias("episode_id"),
        regexp_replace(col("path"), _EPISODE_DIR_PATTERN, "").alias("episode_dir"),
    ).where(col("__kind").not_null() & col("episode_id").not_null())
    if selected_tasks is not None:
        catalog = catalog.where(col("task_slug").is_in(list(selected_tasks)))

    return (
        catalog.groupby(*_IDENTITY_COLUMNS)
        .agg(
            pick("episode", "path").alias("episode_path"),
            pick("episode", "size").alias("episode_size"),
            pick("annotation", "path").alias("annotation_path"),
            pick("annotation", "size").alias("annotation_size"),
        )
        .where(col("episode_path").not_null())
        .with_columns(
            {
                "annotated": col("annotation_path").not_null(),
                "episode_mcap": file(col("episode_path"), io_config=io_config),
                "annotation_mcap": file(col("annotation_path"), io_config=io_config),
            }
        )
        .select(*_CATALOG_COLUMNS)
    )


def _require_columns(episodes: DataFrame, *columns: str) -> None:
    _mcap.require_columns(episodes, *columns, source="abc.raw()")


# --------------------------------------------------------------------------- #
# MCAP summaries
# --------------------------------------------------------------------------- #


def _first_value(values: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = values.get(key)
        if value:
            return value
    return None


@daft.func(return_dtype=_METADATA_DTYPE, use_process=False, unnest=True)
def _read_abc_metadata(handle: daft.File) -> dict[str, object]:
    info, records = _mcap.read_summary(handle, dataset="ABC-130k", extra="abc")

    # ABC releases have used both an "episode-metadata" and a legacy "session-metadata" record.
    by_name = {record.name: record for record in records}
    record = by_name.get("episode-metadata") or by_name.get("session-metadata")
    values: dict[str, str] = {} if record is None else dict(record.metadata)

    duration: float | None = None
    raw_duration = _first_value(values, "duration", "duration_seconds", "duration-seconds", "episode-duration")
    if raw_duration is not None:
        try:
            duration = float(raw_duration)
        except ValueError:
            pass
    start, end = info["message_start_time"], info["message_end_time"]
    if duration is None and start is not None and end is not None:
        duration = (end - start) / 1_000_000_000

    channels = info["channels"]
    return {
        "session_id": _first_value(values, "session_id", "session-id", "session_uuid", "session-uuid"),
        "operator_id": _first_value(values, "operator_id", "operator-id", "operator_uuid", "operator-uuid"),
        "task_name": _first_value(values, "task_name", "task-name", "instruction"),
        "duration_seconds": duration,
        "message_count": info["message_count"],
        "message_start_time": start,
        "message_end_time": end,
        "chunk_count": info["chunk_count"],
        "topics": list(dict.fromkeys(topic for topic, _ in channels)),
        "video_topics": list(dict.fromkeys(topic for topic, schema in channels if schema == _VIDEO_SCHEMA)),
        "indexed": info["indexed"],
        "episode_metadata_json": None
        if record is None
        else json.dumps({"name": record.name, "metadata": values}, separators=(",", ":"), sort_keys=True),
    }


def metadata(episodes: DataFrame) -> DataFrame:
    """Range-read MCAP summaries for a bounded ABC episode catalog.

    Filter and limit the :func:`raw` catalog first: each surviving episode costs
    a few small header/footer/summary range reads (message payloads are not
    downloaded). An MCAP without a summary section falls back to a full
    sequential read and reports ``indexed=False``.

    Returns:
        The catalog columns plus ``session_id``, ``operator_id``, ``task_name``,
        ``duration_seconds``, ``message_count``, ``message_start_time``,
        ``message_end_time``, ``chunk_count``, ``topics``, ``video_topics``,
        ``indexed``, and ``episode_metadata_json``.
    """
    _require_mcap()
    _require_columns(episodes, "episode_mcap")
    return episodes.where(col("episode_mcap").not_null()).select(
        *episodes.schema().column_names(), cast("Any", _read_abc_metadata)(col("episode_mcap"))
    )


# --------------------------------------------------------------------------- #
# Messages and annotations
# --------------------------------------------------------------------------- #


def _with_identity(dataframe: DataFrame) -> DataFrame:
    source = col("source_path")
    return dataframe.with_columns(
        {
            "split": regexp_extract(source, _SPLIT_PATTERN, 1),
            "task_slug": regexp_extract(source, _TASK_PATTERN, 1),
            "episode_id": regexp_extract(source, _EPISODE_ID_PATTERN, 1),
            "episode_dir": regexp_replace(source, _EPISODE_DIR_PATTERN, ""),
            "file_kind": regexp_extract(source, _FILE_KIND_PATTERN, 1),
        }
    )


def _collect_paths(episodes: DataFrame, columns: Sequence[str]) -> list[str]:
    return _mcap.collect_paths(episodes, columns, source="abc.raw()")


def messages(
    episodes: DataFrame,
    *,
    topics: str | Sequence[str] | None = DEFAULT_MESSAGE_TOPICS,
    start_time: int | None = None,
    end_time: int | None = None,
    files: Literal["episode", "annotation", "both"] = "episode",
    batch_size: int = 1000,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Read timestamped ABC messages from a bounded episode catalog.

    The catalog's paths are collected eagerly (filter and limit it first), then
    each file becomes a :func:`daft.read_mcap` scan with the topic and time
    filters pushed into the reader. Streams run on independent clocks and are
    not aligned or resampled.

    Args:
        episodes: Bounded catalog from :func:`raw`.
        topics: Topic or topics to read; defaults to the eight arm/gripper state
            and action streams. ``None`` reads every topic, including video.
        start_time: Inclusive ``log_time`` bound (Unix nanoseconds).
        end_time: Exclusive ``log_time`` bound (Unix nanoseconds).
        files: Read ``episode.mcap``, ``annotation.mcap``, or both.
        batch_size: Messages per source batch.
        io_config: Storage configuration; ``hf://`` tokens are resolved as in :func:`raw`.

    Returns:
        One row per message: ``source_path``, ``topic``, ``log_time``,
        ``publish_time``, ``sequence`` (all integers as ``int64``), raw protobuf
        ``data`` as binary, and ``split``, ``task_slug``, ``episode_id``,
        ``episode_dir``, ``file_kind`` parsed from the path.
    """
    if files not in ("episode", "annotation", "both"):
        raise ValueError("files must be one of: 'episode', 'annotation', or 'both'")
    selected_topics = _normalize_names(topics, name="topics")
    columns = {"episode": ("episode_path",), "annotation": ("annotation_path",)}.get(
        files, ("episode_path", "annotation_path")
    )
    paths = _collect_paths(episodes, columns)
    if not paths:
        return _with_identity(_mcap.empty_messages())
    _require_mcap()
    io_config = _resolve_hf_io_config(io_config, paths)
    frames = _mcap.read_messages(
        paths,
        topics=selected_topics,
        start_time=start_time,
        end_time=end_time,
        batch_size=batch_size,
        io_config=io_config,
    )
    return _with_identity(frames)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("Malformed protobuf varint")


def _protobuf_fields(data: bytes) -> dict[int, object]:
    """Parse a protobuf message into ``{field_number: last value}`` without a schema.

    Varints are returned as ints and length-delimited fields as bytes; fixed-width
    fields are skipped. Enough for the small ABC annotation and Foxglove video messages.
    """
    fields: dict[int, object] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        number, wire_type = tag >> 3, tag & 0x07
        if number == 0:
            raise ValueError("Invalid protobuf field number 0")
        if wire_type == 0:
            fields[number], offset = _read_varint(data, offset)
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            if offset + length > len(data):
                raise ValueError("Truncated protobuf field")
            fields[number] = data[offset : offset + length]
            offset += length
        elif wire_type in (1, 5):
            offset += 8 if wire_type == 1 else 4
            if offset > len(data):
                raise ValueError("Truncated protobuf field")
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
    return fields


def _timestamp_ns(data: object) -> int | None:
    """Convert an encoded ``google.protobuf.Timestamp`` to nanoseconds."""
    if not isinstance(data, bytes):
        return None
    fields = _protobuf_fields(data)
    seconds = cast("int", fields.get(1, 0))
    nanos = cast("int", fields.get(2, 0))
    if seconds >= 1 << 63:
        seconds -= 1 << 64
    if not 0 <= nanos < 1_000_000_000:
        raise ValueError(f"Invalid timestamp nanos: {nanos}")
    return seconds * 1_000_000_000 + nanos


@daft.func(return_dtype=_ANNOTATION_DTYPE, use_process=False, unnest=True)
def _parse_annotation(data: bytes) -> dict[str, object]:
    try:
        fields = _protobuf_fields(data)
        label = fields.get(2)
        return {
            "timestamp_ns": _timestamp_ns(fields.get(1)),
            "label": label.decode("utf-8") if isinstance(label, bytes) else None,
        }
    except (UnicodeDecodeError, ValueError):
        return {"timestamp_ns": None, "label": None}


def annotations(
    episodes: DataFrame,
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    batch_size: int = 1000,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Read typed ``/subtask-annotation`` events from the selected episodes' ``annotation.mcap``.

    Each event marks the start of a free-form subtask that lasts until the next
    event (or the episode end). Episodes without an annotation file contribute
    no rows; malformed payloads yield null ``timestamp_ns`` and ``label``.

    Returns:
        Episode identity, ``file_kind``, ``source_path``, ``topic``, ``log_time``,
        ``publish_time``, ``sequence``, ``timestamp_ns``, and ``label``.
    """
    dataframe = messages(
        episodes,
        topics=ANNOTATION_TOPIC,
        start_time=start_time,
        end_time=end_time,
        files="annotation",
        batch_size=batch_size,
        io_config=io_config,
    )
    return dataframe.select(
        *_IDENTITY_COLUMNS,
        "file_kind",
        "source_path",
        "topic",
        "log_time",
        "publish_time",
        "sequence",
        cast("Any", _parse_annotation)(col("data")),
    )


# --------------------------------------------------------------------------- #
# Camera frames
# --------------------------------------------------------------------------- #


def _nal_types(data: bytes, codec: str) -> Iterator[int]:
    """Yield NAL unit types from an Annex-B H.264/H.265 access unit."""
    offset = data.find(b"\x00\x00\x01")
    while offset != -1:
        header = offset + 3
        if header >= len(data):
            return
        byte = data[header]
        yield (byte & 0x1F) if codec == "h264" else (byte >> 1) & 0x3F
        offset = data.find(b"\x00\x00\x01", header)


def _is_random_access(data: bytes, codec: str) -> bool | None:
    """Whether decoding can start at this access unit; ``None`` when the codec is not inspected."""
    if codec == "h264":
        return any(nal == 5 for nal in _nal_types(data, codec))  # IDR slice
    if codec == "hevc":
        return any(16 <= nal <= 23 for nal in _nal_types(data, codec))  # BLA / IDR / CRA
    return None


def _decode_topic(
    handle: daft.File,
    topic: str,
    start_time: int | None,
    end_time: int | None,
    width: int | None,
    height: int | None,
) -> Iterator[dict[str, object]]:
    import av

    reader_module = _require_mcap()
    with _open_mcap(handle) as stream:
        reader = reader_module.make_reader(stream, decoder_factories=[])
        context: Any = None
        pending: dict[int, dict[str, object]] = {}
        backlog: list[tuple[dict[str, object], bytes]] = []
        index = 0

        def decode(meta: dict[str, object] | None, payload: bytes | None) -> Iterator[dict[str, object]]:
            nonlocal index
            if payload is None:
                frames = context.decode(None)
            else:
                packet = av.Packet(payload)
                packet.pts = index
                pending[index] = cast("dict[str, object]", meta)
                index += 1
                frames = context.decode(packet)
            for frame in frames:
                info = pending.pop(frame.pts, None) if frame.pts is not None else None
                if info is None:
                    continue
                if start_time is not None and cast("int", info["log_time"]) < start_time:
                    continue
                image = frame.to_ndarray(format="rgb24", width=width, height=height)
                yield {
                    **info,
                    "is_key_frame": bool(frame.key_frame),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "data": image,
                }

        # Read from the topic's start: a window that begins mid-GOP needs the
        # preceding keyframe. Payloads are buffered (not decoded) from the latest
        # H.264/H.265 random-access point until the window opens.
        for _, _, message in reader.iter_messages(topics=[topic], end_time=end_time, log_time_order=True):
            fields = _protobuf_fields(message.data)
            fmt = fields.get(4, b"")
            fmt = fmt.decode("utf-8") if isinstance(fmt, bytes) else ""
            payload = fields.get(3, b"")
            if not isinstance(payload, bytes) or not payload:
                continue
            codec = _VIDEO_CODECS.get(fmt.lower())
            if codec is None:
                raise ValueError(f"Unsupported CompressedVideo format {fmt!r} on {topic}")
            if context is None:
                context = cast("Any", av.CodecContext.create(codec, "r"))
            frame_id = fields.get(2)
            meta: dict[str, object] = {
                "log_time": message.log_time,
                "publish_time": message.publish_time,
                "sequence": message.sequence,
                "timestamp_ns": _timestamp_ns(fields.get(1)),
                "frame_id": frame_id.decode("utf-8", "replace") if isinstance(frame_id, bytes) else None,
                "format": fmt,
            }
            before_window = start_time is not None and message.log_time < start_time
            random_access = _is_random_access(payload, codec)
            if before_window and random_access is not None:
                if random_access:
                    backlog = [(meta, payload)]
                elif backlog:
                    backlog.append((meta, payload))
                continue
            if random_access is False and not backlog and index == 0:
                # No preceding random-access point: this frame cannot be decoded.
                continue
            for buffered_meta, buffered_payload in backlog:
                yield from decode(buffered_meta, buffered_payload)
            backlog = []
            yield from decode(meta, payload)
        if context is not None:
            yield from decode(None, None)


@daft.func(return_dtype=_FRAME_DTYPE, use_process=False, unnest=True)
def _decode_frames(
    handle: daft.File,
    topic: str,
    start_time: int | None,
    end_time: int | None,
    width: int | None,
    height: int | None,
) -> Iterator[dict[str, object]]:
    yield from _decode_topic(handle, topic, start_time, end_time, width, height)


def camera_frames(
    episodes: DataFrame,
    cameras: str | Sequence[str] = ("top", "left_wrist", "right_wrist"),
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> DataFrame:
    """Decode ABC camera topics into one row per RGB frame.

    Each (episode, topic) pair is decoded by its own PyAV codec context, chosen
    from the ``format`` field inside each Foxglove ``CompressedVideo`` message
    (H.264 and H.265 in ABC). The topic is read from its start up to
    ``end_time``; for H.264/H.265 only the GOP containing ``start_time`` onward
    is decoded, and frames before ``start_time`` are dropped. Bound the catalog
    and the time window first.

    Args:
        episodes: Bounded catalog from :func:`raw` (needs ``episode_mcap``).
        cameras: Camera alias or aliases: ``top`` (all top streams present),
            ``top_mono``, ``top_left``, ``top_right``, ``left_wrist``, ``right_wrist``.
        start_time: Inclusive ``log_time`` bound (Unix nanoseconds).
        end_time: Exclusive ``log_time`` bound (Unix nanoseconds).
        width: Optional output width; must be given with ``height``.
        height: Optional output height; must be given with ``width``.

    Returns:
        Episode identity, ``camera``, ``topic``, ``log_time``, ``publish_time``,
        ``sequence``, ``timestamp_ns``, ``frame_id``, ``format``,
        ``is_key_frame``, ``width``, ``height``, and RGB image ``data``.
    """
    from daft.dependencies import av

    if not cast("Any", av).module_available():
        raise ImportError("Decoding ABC camera frames needs PyAV (daft[video]).")
    if (width is None) != (height is None):
        raise ValueError("width and height must be given together")
    selected = _normalize_names(cameras, name="cameras")
    assert selected is not None
    unknown = [camera for camera in selected if camera not in CAMERA_TOPICS]
    if unknown:
        raise ValueError(f"Unknown camera(s): {unknown}. Expected one or more of: {', '.join(CAMERA_TOPICS)}.")
    _require_columns(episodes, "episode_mcap", *_IDENTITY_COLUMNS)
    topics = list(dict.fromkeys(topic for camera in selected for topic in CAMERA_TOPICS[camera]))

    camera = when(col("topic") == topics[0], lit(_TOPIC_TO_CAMERA[topics[0]]))
    for topic in topics[1:]:
        camera = camera.when(col("topic") == topic, lit(_TOPIC_TO_CAMERA[topic]))

    decode = cast("Any", _decode_frames)
    return (
        episodes.where(col("episode_mcap").not_null())
        .select(*_IDENTITY_COLUMNS, "episode_mcap", lit(topics).alias("topic"))
        .explode("topic")
        .select(
            *_IDENTITY_COLUMNS,
            "topic",
            decode(col("episode_mcap"), col("topic"), lit(start_time), lit(end_time), lit(width), lit(height)),
        )
        # A topic absent from an episode yields no frames, which surfaces as one all-null row.
        .where(col("log_time").not_null())
        .with_column("camera", camera.otherwise(lit(None)))
        .select(
            *_IDENTITY_COLUMNS,
            "camera",
            "topic",
            "log_time",
            "publish_time",
            "sequence",
            "timestamp_ns",
            "frame_id",
            "format",
            "is_key_frame",
            "width",
            "height",
            "data",
        )
    )


__all__ = [
    "ANNOTATION_TOPIC",
    "CAMERA_TOPICS",
    "DEFAULT_MESSAGE_TOPICS",
    "HF_DATASET",
    "annotations",
    "camera_frames",
    "messages",
    "metadata",
    "raw",
]
