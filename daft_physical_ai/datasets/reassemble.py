"""Lazy access to the original `REASSEMBLE <https://tuwien-asl.github.io/REASSEMBLE_page/>`_ HDF5 release.

REASSEMBLE (RSS 2025, arXiv:2502.05086, CC-BY-4.0) records contact-rich assembly
and disassembly on the NIST Assembly Task Board with a Franka arm. TU Wien
publishes one HDF5 file per recording session (DOI 10.48436/0ewrv-8cb44)::

    <YYYY-MM-DD-HH-MM-SS>.h5
    |-- hama1, hama2, hand              MP4 (H.264) blobs, 640x480 @ 30 fps
    |-- capture_node-camera-image      MP4 render of the event camera, 346x260
    |-- hama1_audio, hama2_audio, hand_audio
    |                                  MP3 bitstreams, one byte per int64 element
    |-- events                         (N, 3) int64: x, y, polarity
    |-- robot_state/<stream>           (N, k) float64 at the sensor's native rate
    |-- timestamps/<stream>            (N,) float64 Unix seconds per stream
    `-- segments_info/<i>/{start,end,success,text,index}
        `-- low_level/<j>/{start,end,success,text}

The release ships as a single 58.9 GB ``data.zip`` whose members are
deflate-compressed, so the HDF5 files cannot be range-read in place. Point
:func:`raw` at extracted copies (local disk or any store Daft reads), or use
:func:`download` to pull only the recordings you need out of the archive with
HTTP range requests.

Each reader keeps output at a bounded granularity: :func:`segments` emits one
row per action segment, :func:`robot_state` and :func:`events` one row per
recording, :func:`audio` one row per microphone, and :func:`camera_frames` one
row per decoded frame. All times are the dataset's own clock, float64 Unix
seconds, so segment ``start``/``end`` values can be passed straight to the
``start_time``/``end_time`` windows.

:func:`lerobot_sidecars` catalogs the per-episode audio and event side files of
the Hugging Face LeRobot port (``robot-lev/reassemble``); :func:`audio` and
:func:`events` read them into the same schemas as the original release.
"""

from __future__ import annotations

import io
import json
import os
import struct
import urllib.request
import wave
import zipfile
import zlib
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import daft
from daft.datatype import DataType
from daft.expressions import col, lit
from daft.functions import file, hdf5_file, when

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.file.hdf5 import Hdf5File
    from daft.io import IOConfig

RECORD_URL = "https://researchdata.tuwien.ac.at/records/0ewrv-8cb44"
DATA_ZIP_URL = "https://researchdata.tuwien.ac.at/api/records/0ewrv-8cb44/files/data.zip/content"
HF_LEROBOT_PORT = "hf://datasets/robot-lev/reassemble"

# Public camera names -> HDF5 dataset names. ``event_cam`` matches the LeRobot port.
CAMERAS: dict[str, str] = {
    "hama1": "hama1",
    "hama2": "hama2",
    "hand": "hand",
    "event_cam": "capture_node-camera-image",
}
MICROPHONES: tuple[str, ...] = ("hama1", "hama2", "hand")

# The robot_state streams in the release (the README's "compenseted" spelling and
# 7-column measured_torque do not match the files: it is compensated_* and 3 columns).
ROBOT_STATE_FIELDS: tuple[str, ...] = (
    "compensated_base_force",
    "compensated_base_torque",
    "gripper_efforts",
    "gripper_positions",
    "gripper_velocities",
    "joint_efforts",
    "joint_positions",
    "joint_velocities",
    "measured_force",
    "measured_torque",
    "pose",
    "velocity",
)
DEFAULT_ROBOT_STATE_FIELDS: tuple[str, ...] = (
    "measured_force",
    "measured_torque",
    "joint_positions",
    "gripper_positions",
    "pose",
)

_IDENTITY_COLUMNS: tuple[str, ...] = ("episode_index", "recording", "split")
_TENSOR_F64 = DataType.tensor(DataType.float64())

_LOW_LEVEL_DTYPE = DataType.struct(
    {
        "skill_index": DataType.int64(),
        "text": DataType.string(),
        "start": DataType.float64(),
        "end": DataType.float64(),
        "success": DataType.bool(),
    }
)
_HIGH_SEGMENT_DTYPE = DataType.struct(
    {
        "segment_index": DataType.int64(),
        "text": DataType.string(),
        "start": DataType.float64(),
        "end": DataType.float64(),
        "success": DataType.bool(),
        "low_level": DataType.list(_LOW_LEVEL_DTYPE),
    }
)
_LOW_SEGMENT_DTYPE = DataType.struct(
    {
        "segment_index": DataType.int64(),
        "segment_text": DataType.string(),
        "skill_index": DataType.int64(),
        "text": DataType.string(),
        "start": DataType.float64(),
        "end": DataType.float64(),
        "success": DataType.bool(),
    }
)
_DECODED_AUDIO_DTYPE = DataType.struct(
    {
        "sample_rate": DataType.int64(),
        "channels": DataType.int64(),
        "num_samples": DataType.int64(),
        "samples": DataType.tensor(DataType.float32()),
    }
)
_EVENTS_DTYPE = DataType.struct(
    {
        "num_events": DataType.int64(),
        "events": DataType.tensor(DataType.int64()),
        "timestamps": _TENSOR_F64,
    }
)
_FRAME_DTYPE = DataType.struct(
    {
        "frame_index": DataType.int64(),
        "timestamp": DataType.float64(),
        "width": DataType.int64(),
        "height": DataType.int64(),
        "image": DataType.image("RGB"),
    }
)
_SIDECAR_DTYPE = DataType.struct(
    {"episode_index": DataType.int64(), "recording": DataType.string(), "split": DataType.string()}
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _names(value: str | Sequence[str] | None, *, name: str, allowed: Sequence[str]) -> tuple[str, ...]:
    values = tuple(allowed) if value is None else ((value,) if isinstance(value, str) else tuple(value))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}. Expected one or more of: {', '.join(allowed)}.")
    return tuple(dict.fromkeys(values))


def _identity(episodes: DataFrame) -> list[str]:
    names = episodes.schema().column_names()
    return [name for name in _IDENTITY_COLUMNS if name in names]


def _require(episodes: DataFrame, *columns: str, source: str = "reassemble.raw()") -> None:
    missing = [name for name in columns if name not in episodes.schema().column_names()]
    if missing:
        raise ValueError(f"Expected a DataFrame from {source} with columns: {missing}")


def _require_h5py() -> Any:
    from daft.dependencies import h5py  # type: ignore[attr-defined]

    if not h5py.module_available():  # ty: ignore[unresolved-attribute]
        raise ImportError("Reading REASSEMBLE HDF5 files requires daft[hdf5].")
    return h5py


def _require_av() -> Any:
    from daft.dependencies import av

    if not cast("Any", av).module_available():
        raise ImportError("Decoding REASSEMBLE audio and video requires PyAV (daft[video]).")
    return av


def _text(value: object) -> str:
    if hasattr(value, "item"):
        value = cast("Any", value).item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _window(timestamps: Any, start_time: float | None, end_time: float | None) -> tuple[int, int]:
    """Index range ``[lo, hi)`` of sorted ``timestamps`` inside ``[start_time, end_time)``."""
    import numpy as np

    lo = 0 if start_time is None else int(np.searchsorted(timestamps, start_time, side="left"))
    hi = len(timestamps) if end_time is None else int(np.searchsorted(timestamps, end_time, side="left"))
    return lo, max(lo, hi)


def _bisect(dataset: Any, value: float) -> int:
    """First index with ``dataset[index] >= value``, reading O(log n) elements of a sorted 1-D dataset."""
    lo, hi = 0, dataset.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if dataset[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def raw(
    path: str,
    io_config: IOConfig | None = None,
    *,
    recordings: str | Sequence[str] | None = None,
) -> DataFrame:
    """Catalog extracted REASSEMBLE recordings as a lazy, one-row-per-file DataFrame.

    ``path`` is a directory (searched recursively for ``*.h5``) or a glob. The
    output has ``recording`` (the file stem, e.g. ``2025-01-11-14-43-37``) and a
    lazy ``file`` HDF5 column. Nothing is opened until a reader needs it.

    Args:
        path: Local directory, object-store prefix, or glob of extracted ``.h5`` files.
        io_config: Optional Daft IO configuration for remote stores.
        recordings: Optional recording name(s) to keep; filtered from paths alone.
    """
    pattern = path if any(char in path for char in "*?[") else f"{path.rstrip('/')}/**/*.h5"
    episodes = daft.from_glob_path(pattern, io_config=io_config).select(
        "path",
        col("path").split("/")[-1].regexp_replace(r"\.h5$", "").alias("recording"),
    )
    if recordings is not None:
        values = [recordings] if isinstance(recordings, str) else list(recordings)
        episodes = episodes.where(col("recording").is_in(values))
    return episodes.select("recording", hdf5_file(col("path"), io_config=io_config).alias("file"))


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #


def _sorted_groups(group: Any) -> list[tuple[int, Any]]:
    return sorted(((int(name), group[name]) for name in group), key=lambda item: item[0])


def _segment_fields(group: Any) -> dict[str, object]:
    return {
        "text": _text(group["text"][()]),
        "start": float(group["start"][()]),
        "end": float(group["end"][()]),
        "success": bool(group["success"][()]),
    }


def _read_segments(handle: Hdf5File, level: str) -> Iterator[dict[str, object]]:
    h5py = _require_h5py()
    with handle.open() as stream, h5py.File(stream, "r") as h5:
        if "segments_info" not in h5:
            return
        for segment_index, group in _sorted_groups(h5["segments_info"]):
            segment = _segment_fields(group)
            low_group = group.get("low_level")
            low = [] if low_group is None else _sorted_groups(low_group)
            if level == "high":
                yield {
                    "segment_index": segment_index,
                    **segment,
                    "low_level": [{"skill_index": index, **_segment_fields(skill)} for index, skill in low],
                }
                continue
            for skill_index, skill in low:
                yield {
                    "segment_index": segment_index,
                    "segment_text": segment["text"],
                    "skill_index": skill_index,
                    **_segment_fields(skill),
                }


@daft.func(return_dtype=_HIGH_SEGMENT_DTYPE, use_process=False, unnest=True)
def _high_segments(handle: Hdf5File) -> Iterator[dict[str, object]]:
    yield from _read_segments(handle, "high")


@daft.func(return_dtype=_LOW_SEGMENT_DTYPE, use_process=False, unnest=True)
def _low_segments(handle: Hdf5File) -> Iterator[dict[str, object]]:
    yield from _read_segments(handle, "low")


def segments(episodes: DataFrame, level: Literal["high", "low"] = "high") -> DataFrame:
    """Read the hierarchical action-segment annotations.

    ``level="high"`` returns one row per high-level action (``"Pick square peg
    3."``, ``"No action."``) with ``segment_index``, ``text``, ``start``,
    ``end``, ``success``, and a ``low_level`` list of its skills.
    ``level="low"`` returns one row per low-level skill (``"Grasp"``,
    ``"Lift"``) with its parent ``segment_index`` and ``segment_text``.
    ``start`` and ``end`` are Unix seconds. Only ``segments_info`` is read.
    """
    _require(episodes, "file")
    if level not in ("high", "low"):
        raise ValueError("level must be 'high' or 'low'")
    udf = cast("Any", _high_segments if level == "high" else _low_segments)
    columns = (
        ("segment_index", "text", "start", "end", "success", "low_level")
        if level == "high"
        else ("segment_index", "segment_text", "skill_index", "text", "start", "end", "success")
    )
    identity = _identity(episodes)
    return (
        episodes.where(col("file").not_null())
        .select(*identity, udf(col("file")))
        # A recording without segments yields one all-null row.
        .where(col("segment_index").not_null())
        .with_column("duration", col("end") - col("start"))
        .select(*identity, *columns[:5], "duration", *columns[5:])
    )


# --------------------------------------------------------------------------- #
# Robot state
# --------------------------------------------------------------------------- #


def robot_state(
    episodes: DataFrame,
    fields: Sequence[str] = DEFAULT_ROBOT_STATE_FIELDS,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> DataFrame:
    """Read proprioception and force/torque streams at their native rates.

    Output stays one row per recording. Each requested field becomes two
    columns: ``<field>``, an ``(N, k)`` float64 tensor, and
    ``<field>_timestamps``, its ``(N,)`` Unix-second timestamps. Streams run at
    different rates (about 1 kHz for measured F/T and joints, about 0.5 kHz for
    pose and base F/T), so ``N`` differs per field. Optional ``start_time`` /
    ``end_time`` (Unix seconds, end exclusive) bound every stream; only that
    slice is read. A field absent from a file yields empty tensors.
    """
    h5py = _require_h5py()
    _require(episodes, "file")
    fields = tuple(fields)
    if not fields:
        raise ValueError("fields must contain at least one robot_state stream")
    unknown = [field for field in fields if field not in ROBOT_STATE_FIELDS]
    if unknown:
        raise ValueError(f"Unknown robot_state field(s): {unknown}. Expected any of: {', '.join(ROBOT_STATE_FIELDS)}.")

    dtype: dict[str, DataType] = {}
    for field in fields:
        dtype[field] = _TENSOR_F64
        dtype[f"{field}_timestamps"] = _TENSOR_F64

    @daft.func(return_dtype=DataType.struct(dtype), use_process=False, unnest=True)
    def read_robot_state(handle: Hdf5File) -> dict[str, object]:
        import numpy as np

        # Daft cannot build a tensor column whose values are all null, so an absent stream is empty.
        out: dict[str, object] = {}
        for field in fields:
            out[field], out[f"{field}_timestamps"] = np.zeros((0, 0)), np.zeros((0,))
        with handle.open() as stream, h5py.File(stream, "r") as h5:
            for field in fields:
                values_path, times_path = f"robot_state/{field}", f"timestamps/{field}"
                if values_path not in h5 or times_path not in h5:
                    continue
                times = h5[times_path][()].reshape(-1)
                lo, hi = _window(times, start_time, end_time)
                out[field] = h5[values_path][lo:hi]
                out[f"{field}_timestamps"] = times[lo:hi]
        return out

    identity = _identity(episodes)
    return episodes.where(col("file").not_null()).select(*identity, cast("Any", read_robot_state)(col("file")))


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def _mp3_from_wav(data: bytes) -> tuple[bytes | None, Any]:
    """Return ``(mp3_bytes, None)`` for the LeRobot port's WAVs, else ``(None, pcm_array)``.

    The port writes each MP3 *byte* as one 32-bit PCM sample under a 16 kHz
    header, so the WAV plays as noise; the bytes round-trip exactly. A WAV
    holding real PCM (values outside 0..255) is returned as samples instead.
    """
    import numpy as np

    with wave.open(io.BytesIO(data)) as wav:
        width, channels, rate = wav.getsampwidth(), wav.getnchannels(), wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    samples = np.frombuffer(frames, dtype=dtypes[width])
    if width == 4 and channels == 1 and samples.size and samples.min() >= 0 and samples.max() <= 255:
        return samples.astype(np.uint8).tobytes(), None
    scale = {1: 128.0, 2: 32768.0, 4: 2147483648.0}[width]
    offset = 128.0 if width == 1 else 0.0
    pcm = ((samples.astype(np.float32) - offset) / scale).reshape(-1, channels).T
    return None, (rate, np.ascontiguousarray(pcm))


def _decode_mp3(data: bytes) -> dict[str, object]:
    import numpy as np

    av = _require_av()
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        chunks = [frame.to_ndarray().reshape(frame.layout.nb_channels, -1) for frame in container.decode(stream)]
        rate = int(stream.rate)
    if not chunks:
        return {"sample_rate": rate, "channels": 0, "num_samples": 0, "samples": np.zeros((0, 0), np.float32)}
    samples = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    return {
        "sample_rate": rate,
        "channels": int(samples.shape[0]),
        "num_samples": int(samples.shape[1]),
        "samples": samples,
    }


def _read_mp3(handle: daft.File, microphone: str, from_port: bool) -> bytes | None:
    import numpy as np

    if from_port:
        with handle.open() as stream:
            mp3, _ = _mp3_from_wav(stream.read())
        return mp3
    h5py = _require_h5py()
    with handle.open() as stream, h5py.File(stream, "r") as h5:
        name = f"{microphone}_audio"
        if name not in h5:
            return None
        return h5[name][()].astype(np.uint8).tobytes()


def audio(
    episodes: DataFrame,
    microphones: str | Sequence[str] | None = None,
    *,
    decode: bool = True,
) -> DataFrame:
    """Read the three microphone tracks, one row per (recording, microphone).

    The release stores MP3 bitstreams (``hama1``/``hama2``: 16 kHz stereo;
    ``hand``: 48 kHz mono). With ``decode=True`` the output has
    ``sample_rate``, ``channels``, ``num_samples``, and ``samples``, a
    ``(channels, num_samples)`` float32 tensor; with ``decode=False`` it has
    the raw ``mp3`` bytes. The release has no per-sample audio timestamps.

    Works on :func:`raw` catalogs and on :func:`lerobot_sidecars` catalogs; for
    the latter the MP3 bytes are recovered from the port's mislabelled WAVs.
    """
    selected = _names(microphones, name="microphones", allowed=MICROPHONES)
    names = episodes.schema().column_names()
    from_port = "file" not in names
    if from_port:
        _require(
            episodes,
            *(f"{microphone}_audio" for microphone in selected),
            source="reassemble.raw() or reassemble.lerobot_sidecars()",
        )
    if decode:
        _require_av()
    identity = _identity(episodes)

    @daft.func(return_dtype=DataType.binary(), use_process=False)
    def read_mp3(handle: daft.File | None, microphone: str) -> bytes | None:
        return None if handle is None else _read_mp3(handle, microphone, from_port)

    # Generators: yielding nothing for a missing track gives a null row, which Daft
    # handles even when every row of a batch is missing (a returned None does not).
    @daft.func(return_dtype=_DECODED_AUDIO_DTYPE, use_process=False, unnest=True)
    def read_decoded(handle: daft.File | None, microphone: str) -> Iterator[dict[str, object]]:
        if handle is None:
            return
        if from_port:
            with handle.open() as stream:
                mp3, pcm = _mp3_from_wav(stream.read())
            if mp3 is None:
                rate, samples = pcm
                yield {
                    "sample_rate": int(rate),
                    "channels": int(samples.shape[0]),
                    "num_samples": int(samples.shape[1]),
                    "samples": samples,
                }
                return
        else:
            mp3 = _read_mp3(handle, microphone, from_port=False)
        if mp3 is not None:
            yield _decode_mp3(mp3)

    if from_port:
        per_microphone = [
            episodes.select(*identity, lit(microphone).alias("microphone"), col(f"{microphone}_audio").alias("handle"))
            for microphone in selected
        ]
        source = per_microphone[0]
        for other in per_microphone[1:]:
            source = source.concat(other)
    else:
        source = (
            episodes.where(col("file").not_null())
            .select(*identity, "file", lit(list(selected)).alias("microphone"))
            .explode("microphone")
            .select(*identity, "microphone", col("file").alias("handle"))
        )

    if not decode:
        mp3 = cast("Any", read_mp3)(col("handle"), col("microphone"))
        return source.select(*identity, "microphone", mp3.alias("mp3")).where(col("mp3").not_null())
    decoded = cast("Any", read_decoded)(col("handle"), col("microphone"))
    return (
        source.select(*identity, "microphone", decoded)
        .where(col("sample_rate").not_null())
        .select(*identity, "microphone", "sample_rate", "channels", "num_samples", "samples")
    )


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def _events_from_h5(handle: Hdf5File, start_time: float | None, end_time: float | None) -> dict[str, object] | None:
    h5py = _require_h5py()
    with handle.open() as stream, h5py.File(stream, "r") as h5:
        if "events" not in h5 or "timestamps/events" not in h5:
            return None
        times = h5["timestamps/events"]
        lo = 0 if start_time is None else _bisect(times, start_time)
        hi = times.shape[0] if end_time is None else _bisect(times, end_time)
        hi = max(lo, hi)
        return {"num_events": hi - lo, "events": h5["events"][lo:hi], "timestamps": times[lo:hi]}


def _events_from_npz(handle: daft.File, start_time: float | None, end_time: float | None) -> dict[str, object]:
    import numpy as np

    with handle.open() as stream, np.load(io.BytesIO(stream.read())) as npz:
        events, times = npz["events"], npz["timestamps"]
    lo, hi = _window(times, start_time, end_time)
    return {"num_events": hi - lo, "events": events[lo:hi], "timestamps": times[lo:hi]}


def events(
    episodes: DataFrame,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> DataFrame:
    """Read the sparse DAVIS346 event stream, one row per recording.

    Output has ``num_events``, ``events`` (an ``(N, 3)`` int64 tensor of ``x``,
    ``y``, ``polarity``), and ``timestamps`` (``(N,)`` Unix seconds). The
    stream runs at roughly 100k events per second, so bound it with
    ``start_time`` / ``end_time`` (end exclusive). On HDF5 the window is found
    by bisecting the timestamps, so only the windowed slice is read.

    Works on :func:`raw` catalogs and on :func:`lerobot_sidecars` catalogs
    (the port's ``events/episode_*.npz`` files hold the same arrays; an NPZ is
    read whole before windowing).
    """
    names = episodes.schema().column_names()
    from_port = "file" not in names
    column = "events_file" if from_port else "file"
    if from_port:
        _require(episodes, "events_file", source="reassemble.raw() or reassemble.lerobot_sidecars()")
    else:
        _require_h5py()

    @daft.func(return_dtype=_EVENTS_DTYPE, use_process=False, unnest=True)
    def read_events(handle: daft.File) -> Iterator[dict[str, object]]:
        row = (
            _events_from_npz(handle, start_time, end_time)
            if from_port
            else _events_from_h5(cast("Any", handle), start_time, end_time)
        )
        if row is not None:
            yield row

    identity = _identity(episodes)
    return (
        episodes.where(col(column).not_null())
        .select(*identity, cast("Any", read_events)(col(column)))
        .where(col("num_events").not_null())
    )


# --------------------------------------------------------------------------- #
# Camera frames
# --------------------------------------------------------------------------- #


def _decode_camera(
    handle: Hdf5File,
    dataset: str,
    start_time: float | None,
    end_time: float | None,
    width: int | None,
    height: int | None,
) -> Iterator[dict[str, object]]:
    h5py = _require_h5py()
    av = _require_av()
    with handle.open() as stream, h5py.File(stream, "r") as h5:
        if dataset not in h5:
            return
        blob = h5[dataset][()].tobytes()
        times_path = f"timestamps/{dataset}"
        times = h5[times_path][()].reshape(-1) if times_path in h5 else None
    first, last = (0, None) if times is None else _window(times, start_time, end_time)
    if last is not None and first >= last:
        return
    with av.open(io.BytesIO(blob)) as container:
        video = container.streams.video[0]
        rate, time_base = video.average_rate, video.time_base
        if first > 0 and rate and time_base:
            container.seek(int(first / rate / time_base), stream=video, backward=True, any_frame=False)
        for position, frame in enumerate(container.decode(video)):
            if frame.pts is not None and rate and time_base:
                index = round(float(frame.pts * time_base * rate))
            else:
                index = position
            if index < first:
                continue
            if last is not None and index >= last:
                break
            image = (
                frame.reformat(width=width, height=height, format="rgb24")
                if width is not None
                else frame.reformat(format="rgb24")
            )
            array = image.to_ndarray()
            yield {
                "frame_index": index,
                "timestamp": None if times is None or index >= len(times) else float(times[index]),
                "width": int(array.shape[1]),
                "height": int(array.shape[0]),
                "image": array,
            }


@daft.func(return_dtype=_FRAME_DTYPE, use_process=False, unnest=True)
def _camera_frames_udf(
    handle: Hdf5File,
    dataset: str,
    start_time: float | None,
    end_time: float | None,
    width: int | None,
    height: int | None,
) -> Iterator[dict[str, object]]:
    yield from _decode_camera(handle, dataset, start_time, end_time, width, height)


def camera_frames(
    episodes: DataFrame,
    cameras: str | Sequence[str] = "hand",
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> DataFrame:
    """Decode the embedded MP4 blobs into one row per RGB frame.

    ``cameras`` picks from ``hama1``, ``hama2``, ``hand`` (640x480 @ 30 fps)
    and ``event_cam`` (the event-camera render, 346x260, about 6 fps). Each
    frame carries its ``frame_index`` and its ``timestamp`` from
    ``timestamps/<camera>`` (Unix seconds). ``start_time`` / ``end_time``
    (end exclusive) select frames by timestamp; decoding seeks to the keyframe
    before the window. A camera missing from a recording yields no rows.
    """
    _require_av()
    _require_h5py()
    _require(episodes, "file")
    if (width is None) != (height is None):
        raise ValueError("width and height must be given together")
    selected = _names(cameras, name="cameras", allowed=tuple(CAMERAS))
    identity = _identity(episodes)

    dataset = when(col("camera") == selected[0], lit(CAMERAS[selected[0]]))
    for camera in selected[1:]:
        dataset = dataset.when(col("camera") == camera, lit(CAMERAS[camera]))

    decode = cast("Any", _camera_frames_udf)
    return (
        episodes.where(col("file").not_null())
        .select(*identity, "file", lit(list(selected)).alias("camera"))
        .explode("camera")
        .with_column("dataset", dataset.otherwise(lit(None)))
        .select(
            *identity,
            "camera",
            decode(col("file"), col("dataset"), lit(start_time), lit(end_time), lit(width), lit(height)),
        )
        # A camera absent from a recording yields one all-null row.
        .where(col("frame_index").not_null())
        .select(*identity, "camera", "frame_index", "timestamp", "width", "height", "image")
    )


# --------------------------------------------------------------------------- #
# LeRobot port side files
# --------------------------------------------------------------------------- #


def lerobot_sidecars(dataset_uri: str = HF_LEROBOT_PORT, io_config: IOConfig | None = None) -> DataFrame:
    """Catalog the LeRobot port's audio and event side files, one row per episode.

    ``daft.datasets.lerobot`` reads the port's frames and videos but not the
    side folders. This reads ``meta/splits.json`` (episode -> original
    recording and author split) and returns ``episode_index``, ``recording``,
    ``split``, lazy ``hama1_audio`` / ``hama2_audio`` / ``hand_audio`` WAV file
    columns, and a lazy ``events_file`` NPZ column. Pass the result to
    :func:`audio` or :func:`events`, or join it to :func:`raw` on
    ``recording``.
    """
    from ._mcap import resolve_hf_io_config

    root = dataset_uri.rstrip("/")
    io_config = resolve_hf_io_config(io_config, [root])

    @daft.func(return_dtype=DataType.list(_SIDECAR_DTYPE), use_process=False)
    def read_splits(handle: daft.File) -> list[dict[str, object]]:
        with handle.open() as stream:
            splits = json.loads(stream.read())
        return [
            {"episode_index": int(index), "recording": entry.get("recording"), "split": entry.get("split")}
            for index, entry in sorted(splits.items(), key=lambda item: int(item[0]))
        ]

    episodes = (
        daft.from_pydict({"url": [f"{root}/meta/splits.json"]})
        .select(cast("Any", read_splits)(file(col("url"), io_config=io_config)).alias("episode"))
        .explode("episode")
        .select(col("episode").unnest())
    )
    padded = col("episode_index").cast(DataType.string()).lpad(6, "0")
    audio_root = lit(f"{root}/audio/episode_").concat(padded)
    return episodes.select(
        *_IDENTITY_COLUMNS,
        *(
            file(audio_root.concat(lit(f"/{microphone}.wav")), io_config=io_config).alias(f"{microphone}_audio")
            for microphone in MICROPHONES
        ),
        file(lit(f"{root}/events/episode_").concat(padded).concat(lit(".npz")), io_config=io_config).alias(
            "events_file"
        ),
    )


# --------------------------------------------------------------------------- #
# Selective download from the TU Wien archive
# --------------------------------------------------------------------------- #


def _http_range(url: str, start: int, end: int) -> Any:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(request, timeout=120)


def _http_size(url: str) -> int:
    with _http_range(url, 0, 0) as response:
        content_range = response.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise OSError(f"{url} does not support HTTP range requests")
    return int(content_range.rsplit("/", 1)[1])


class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, one HTTP range request per read."""

    def __init__(self, url: str) -> None:
        self._url, self._size, self._position = url, _http_size(url), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        self._position = base + offset
        return self._position

    def readinto(self, buffer: Any) -> int:
        count = min(len(buffer), self._size - self._position)
        if count <= 0:
            return 0
        with _http_range(self._url, self._position, self._position + count - 1) as response:
            data = response.read()
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)


def _zip_members(url: str) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(io.BufferedReader(_HttpRangeFile(url), buffer_size=1 << 16)) as archive:
        return {
            os.path.basename(info.filename).removesuffix(".h5"): info
            for info in archive.infolist()
            if info.filename.endswith(".h5")
        }


def remote_catalog(url: str = DATA_ZIP_URL) -> DataFrame:
    """List the recordings inside the TU Wien ``data.zip`` without downloading it.

    Reads only the archive's central directory (a few HTTP range requests).
    Returns ``recording``, ``size_bytes`` (extracted HDF5 size), and
    ``compressed_bytes`` (what :func:`download` transfers), sorted by name.
    """
    members = sorted(_zip_members(url).items())
    return daft.from_pydict(
        {
            "recording": [name for name, _ in members],
            "size_bytes": [info.file_size for _, info in members],
            "compressed_bytes": [info.compress_size for _, info in members],
        }
    )


def download(
    recordings: str | Sequence[str],
    dest: str,
    *,
    url: str = DATA_ZIP_URL,
    overwrite: bool = False,
) -> list[str]:
    """Extract selected recordings from the TU Wien ``data.zip`` with HTTP range reads.

    Each member is streamed (one range request over its compressed bytes),
    inflated, and CRC-checked into ``dest/<recording>.h5``, so you transfer
    only those recordings instead of the 58.9 GB archive. Existing files are
    kept unless ``overwrite`` is set. Returns the local paths.
    """
    names = [recordings] if isinstance(recordings, str) else list(recordings)
    if not names:
        raise ValueError("recordings must contain at least one recording name")
    members = _zip_members(url)
    missing = [name for name in names if name not in members]
    if missing:
        raise ValueError(f"Recording(s) not in the archive: {missing}")
    os.makedirs(dest, exist_ok=True)
    paths = []
    for name in names:
        info, target = members[name], os.path.join(dest, f"{name}.h5")
        paths.append(target)
        if os.path.exists(target) and not overwrite:
            continue
        with _http_range(url, info.header_offset, info.header_offset + 29) as response:
            header = response.read()
        if header[:4] != b"PK\x03\x04":
            raise OSError(f"Bad local header for {info.filename}")
        name_length, extra_length = struct.unpack("<HH", header[26:30])
        start = info.header_offset + 30 + name_length + extra_length
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise OSError(f"Unsupported compression method {info.compress_type} for {info.filename}")
        inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
        partial, crc = f"{target}.part", 0
        with _http_range(url, start, start + info.compress_size - 1) as response, open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                data = inflater.decompress(chunk) if inflater is not None else chunk
                crc = zlib.crc32(data, crc)
                out.write(data)
            if inflater is not None:
                tail = inflater.flush()
                crc = zlib.crc32(tail, crc)
                out.write(tail)
        if crc != info.CRC or os.path.getsize(partial) != info.file_size:
            os.remove(partial)
            raise OSError(f"Checksum mismatch extracting {info.filename}")
        os.replace(partial, target)
    return paths


__all__ = [
    "CAMERAS",
    "DATA_ZIP_URL",
    "DEFAULT_ROBOT_STATE_FIELDS",
    "HF_LEROBOT_PORT",
    "MICROPHONES",
    "RECORD_URL",
    "ROBOT_STATE_FIELDS",
    "audio",
    "camera_frames",
    "download",
    "events",
    "lerobot_sidecars",
    "raw",
    "remote_catalog",
    "robot_state",
    "segments",
]
