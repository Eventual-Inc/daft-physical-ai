"""Shared MCAP machinery for the MCAP-backed dataset readers (``abc``, ``hiw500``).

Private module. It holds the format-neutral pieces: Hugging Face auth and
listing helpers, a seekable ``daft.File`` adapter so the ``mcap`` package can
range-read summaries, summary extraction with a full-scan fallback, and
normalization of :func:`daft.read_mcap` output across Daft releases.
Message-encoding specifics (protobuf, ROS 2 CDR) stay in the dataset modules.
"""

from __future__ import annotations

import ast
import io
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import daft
from daft.datatype import DataType
from daft.expressions import col, lit

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.io import IOConfig

HF_PREFIX = "hf://datasets/"
LIBRARY_NAME = "daft-physical-ai"


def require_mcap(dataset: str, extra: str) -> Any:
    """Import ``mcap.reader`` or raise an ImportError naming the extra to install."""
    try:
        import mcap.reader
    except ImportError as error:
        raise ImportError(
            f"{dataset} needs the mcap reader. Install it with: pip install 'daft-physical-ai[{extra}]'"
        ) from error
    return mcap.reader


def normalize_names(value: str | Sequence[str] | None, *, name: str) -> tuple[str, ...] | None:
    """Turn a name or names into a de-duplicated tuple, rejecting empty input."""
    if value is None:
        return None
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} values must be non-empty strings")
    return tuple(dict.fromkeys(values))


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #


def resolve_hf_io_config(io_config: IOConfig | None, paths: Sequence[str]) -> IOConfig | None:
    """Fill in a Hugging Face token from ``HF_TOKEN`` or the Hub login for ``hf://`` paths."""
    if not any(path.startswith("hf://") for path in paths):
        return io_config
    if io_config is not None and (io_config.hf.anonymous or io_config.hf.token is not None):
        return io_config
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except ImportError:
            pass
    if token is None:
        return io_config
    from daft.io import HuggingFaceConfig, IOConfig

    if io_config is not None:
        return io_config.replace(hf=io_config.hf.replace(token=token))
    return IOConfig(hf=HuggingFaceConfig(token=token, use_xet=True))


def parse_hf_root(root: str, *, example: str) -> tuple[str, str | None]:
    """Split ``hf://datasets/<org>/<name>[@<revision>]`` into ``(repo_id, revision)``."""
    parts = root.removeprefix(HF_PREFIX).split("/")
    if not root.startswith(HF_PREFIX) or len(parts) != 2:
        raise ValueError(f"Hugging Face path must be a dataset root such as {example}")
    namespace, name_revision = parts
    name, _, revision = name_revision.partition("@")
    return f"{namespace}/{name}", revision or None


def hf_token(io_config: IOConfig | None) -> str | bool | None:
    """The token argument for ``huggingface_hub.HfApi`` implied by ``io_config``."""
    if io_config is None:
        return None
    return False if io_config.hf.anonymous else io_config.hf.token


def require_hf_hub(dataset: str, extra: str) -> Any:
    """Import ``huggingface_hub`` or raise an ImportError naming the extra to install."""
    try:
        import huggingface_hub
    except ImportError as error:
        raise ImportError(
            f"Listing {dataset} on Hugging Face needs huggingface_hub. "
            f"Install it with: pip install 'daft-physical-ai[{extra}]'"
        ) from error
    return huggingface_hub


def empty_listing() -> DataFrame:
    """An empty ``(path, size)`` object listing."""
    return daft.from_pydict({"path": [""], "size": [0]}).where(lit(False))


# --------------------------------------------------------------------------- #
# Seekable daft.File and MCAP summaries
# --------------------------------------------------------------------------- #


class SeekableFile(io.RawIOBase):
    """Expose a ``daft.File`` as a seekable stream, so ``mcap`` range-reads the footer and summary."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._handle.__enter__()

    def readinto(self, buffer: Any) -> int:
        data = self._handle.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        if not self.closed:
            self._handle.__exit__(None, None, None)
        super().close()


def open_mcap(handle: daft.File) -> io.BufferedReader:
    """Open a ``daft.File`` as a buffered, seekable binary stream."""
    return io.BufferedReader(SeekableFile(handle.open()), buffer_size=1 << 16)


def _scan_summary(stream: io.BufferedReader) -> tuple[dict[str, Any], list[Any]]:
    """Collect channels, statistics, and metadata in one pass for MCAPs without a summary section."""
    from mcap.records import Channel, Message, Metadata, Schema
    from mcap.stream_reader import StreamReader

    stream.seek(0)
    schemas: dict[int, str] = {}
    channels: dict[int, tuple[str, int]] = {}
    counts: dict[int, int] = {}
    metadata: list[Any] = []
    count, start, end = 0, None, None
    for record in StreamReader(stream).records:
        if isinstance(record, Schema):
            schemas[record.id] = record.name
        elif isinstance(record, Channel):
            channels[record.id] = (record.topic, record.schema_id)
        elif isinstance(record, Metadata):
            metadata.append(record)
        elif isinstance(record, Message):
            count += 1
            counts[record.channel_id] = counts.get(record.channel_id, 0) + 1
            start = record.log_time if start is None else min(start, record.log_time)
            end = record.log_time if end is None else max(end, record.log_time)
    info = {
        "channels": [(topic, schemas.get(schema_id, "")) for topic, schema_id in channels.values()],
        "channel_message_counts": {topic: counts.get(channel_id, 0) for channel_id, (topic, _) in channels.items()},
        "message_count": count,
        "message_start_time": start,
        "message_end_time": end,
        "chunk_count": None,
        "indexed": False,
    }
    return info, metadata


def read_summary(handle: daft.File, *, dataset: str, extra: str) -> tuple[dict[str, Any], list[Any]]:
    """Range-read an MCAP summary; fall back to a full scan when the file has none.

    Returns ``(info, metadata_records)``. ``info`` holds ``channels`` (a list of
    ``(topic, schema_name)``), ``channel_message_counts`` (``{topic: count}``),
    ``message_count``, ``message_start_time``, ``message_end_time``,
    ``chunk_count``, and ``indexed``.
    """
    reader_module = require_mcap(dataset, extra)
    with open_mcap(handle) as stream:
        reader = reader_module.make_reader(stream)
        summary = reader.get_summary()
        if summary is None:
            return _scan_summary(stream)
        stats = summary.statistics
        channel_counts = {} if stats is None else stats.channel_message_counts
        info = {
            "channels": [
                (
                    channel.topic,
                    summary.schemas[channel.schema_id].name if channel.schema_id in summary.schemas else "",
                )
                for channel in summary.channels.values()
            ],
            "channel_message_counts": {
                channel.topic: channel_counts.get(channel_id, 0) for channel_id, channel in summary.channels.items()
            },
            "message_count": None if stats is None else stats.message_count,
            "message_start_time": None if stats is None else stats.message_start_time,
            "message_end_time": None if stats is None else stats.message_end_time,
            "chunk_count": None if stats is None else stats.chunk_count,
            "indexed": bool(summary.chunk_indexes),
        }
        return info, list(reader.iter_metadata())


# --------------------------------------------------------------------------- #
# daft.read_mcap normalization
# --------------------------------------------------------------------------- #


def decode_payload(value: str | None) -> bytes | None:
    """Recover the payload bytes from the released reader's ``str(bytes)`` rendering."""
    if value is None:
        return None
    if value[:2] in ("b'", 'b"'):
        decoded = ast.literal_eval(value)
        if isinstance(decoded, bytes):
            return decoded
    return value.encode("utf-8", "surrogateescape")


_decode_payload_udf = daft.func(decode_payload, return_dtype=DataType.binary(), use_process=False)


def normalize_messages(dataframe: DataFrame, path: str) -> DataFrame:
    """Give every ``read_mcap`` release the same columns and types.

    Released Daft (<= v0.7.25) uses a Python reader with no ``source_path``,
    ``int64``/``int32`` times and sequence, and ``data`` rendered as
    ``str(bytes)``. Daft main's native reader adds ``source_path`` and uses
    ``uint64``/``uint32`` with binary ``data``. Both normalize to ``int64``
    times and sequence and binary ``data``.
    """
    schema = dataframe.schema()
    data = col("data")
    if schema["data"].dtype == DataType.string():
        data = _decode_payload_udf(data)
    elif schema["data"].dtype != DataType.binary():
        data = data.cast(DataType.binary())
    source = col("source_path") if "source_path" in schema.column_names() else lit(path)
    return dataframe.select(
        source.alias("source_path"),
        col("topic"),
        col("log_time").cast(DataType.int64()),
        col("publish_time").cast(DataType.int64()),
        col("sequence").cast(DataType.int64()),
        data.alias("data"),
    )


def empty_messages() -> DataFrame:
    """An empty frame with the normalized message columns."""
    import pyarrow as pa

    schema = pa.schema(
        [
            ("source_path", pa.string()),
            ("topic", pa.string()),
            ("log_time", pa.int64()),
            ("publish_time", pa.int64()),
            ("sequence", pa.int64()),
            ("data", pa.binary()),
        ]
    )
    return daft.from_arrow(schema.empty_table())


def read_messages(
    paths: Sequence[str],
    *,
    topics: tuple[str, ...] | None,
    start_time: int | None,
    end_time: int | None,
    batch_size: int,
    io_config: IOConfig | None,
) -> DataFrame:
    """One normalized :func:`daft.read_mcap` scan per path, concatenated.

    Topic and time filters are pushed into each reader. ``paths`` must be
    non-empty; callers return :func:`empty_messages` themselves.
    """
    frames = [
        normalize_messages(
            daft.read_mcap(
                path,
                io_config=io_config,
                start_time=start_time,
                end_time=end_time,
                topics=None if topics is None else list(topics),
                batch_size=batch_size,
            ),
            path,
        )
        for path in paths
    ]
    return frames[0] if len(frames) == 1 else daft.concat(frames)


def require_columns(episodes: DataFrame, *columns: str, source: str) -> None:
    """Raise a ValueError naming ``source`` when ``episodes`` lacks any of ``columns``."""
    missing = [name for name in columns if name not in episodes.schema().column_names()]
    if missing:
        raise ValueError(f"Expected an episode DataFrame from {source} with columns: {missing}")


def collect_paths(episodes: DataFrame, columns: Sequence[str], *, source: str) -> list[str]:
    """Eagerly collect the distinct non-null paths in ``columns`` of a bounded catalog."""
    require_columns(episodes, *columns, source=source)
    selected = episodes.select(*columns).to_pydict()
    paths = [value for name in columns for value in selected[name] if value is not None]
    return list(dict.fromkeys(paths))
