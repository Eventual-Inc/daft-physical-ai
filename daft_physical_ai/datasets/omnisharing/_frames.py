"""Frame-level signal expansion and camera-clock alignment."""

from __future__ import annotations

from bisect import bisect_left
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from daft.datatype import DataType
from daft.expressions import col
from daft.functions import unnest

from ._cameras import _normalize_camera_selectors, _stream_name
from ._common import (
    SIDES,
    get_node,
    h5path,
    read_episodes,
    require_episode_column,
)
from ._signals import TRAJECTORY_FIELDS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from daft.dataframe import DataFrame

FRAME_FIELDS: tuple[str, ...] = (
    *TRAJECTORY_FIELDS,
    *(f"observation/{side}/tactile" for side in SIDES),
)


def nearest_timestamp(clock_us: Sequence[int], timestamp_us: int) -> tuple[int, int]:
    """Return nearest clock index and signed ``clock - target`` residual.

    Ties resolve to the earlier/lower index for deterministic alignment.
    """
    if not clock_us:
        raise ValueError("camera clock must contain at least one timestamp")
    if any(right < left for left, right in pairwise(clock_us)):
        raise ValueError("camera clock timestamps must be monotonically non-decreasing")
    position = bisect_left(clock_us, timestamp_us)
    if position == 0:
        index = 0
    elif position == len(clock_us):
        index = len(clock_us) - 1
    else:
        before = position - 1
        after = position
        index = before if timestamp_us - clock_us[before] <= clock_us[after] - timestamp_us else after
    return index, int(clock_us[index]) - int(timestamp_us)


def _normalize_fields(fields: str | Sequence[str]) -> tuple[str, ...]:
    selected = (fields,) if isinstance(fields, str) else tuple(fields)
    if not selected:
        raise ValueError(
            "frames() requires an explicit field whitelist; tactile expansion can be thousands of floats per row"
        )
    unknown = [field for field in selected if field not in FRAME_FIELDS]
    if unknown:
        raise ValueError(f"Unknown frame field(s): {unknown}. Valid fields are: {', '.join(FRAME_FIELDS)}")
    return tuple(dict.fromkeys(selected))


def _normalize_include_columns(episodes: DataFrame, columns: str | Sequence[str]) -> tuple[str, ...]:
    selected = (columns,) if isinstance(columns, str) else tuple(columns)
    selected = tuple(dict.fromkeys(selected))
    missing = [name for name in selected if name not in episodes.column_names]
    if missing:
        raise ValueError(f"Unknown include_columns: {missing}")
    if "episode" in selected:
        raise ValueError("include_columns cannot broadcast the HDF5 'episode' handle")
    return selected


def frames(
    episodes: DataFrame,
    fields: str | Sequence[str],
    *,
    align_cameras: Sequence[str | tuple[str, str]] = (),
    include_columns: str | Sequence[str] = (),
) -> DataFrame:
    """Expand selected signals to frame rows and align explicit camera clocks."""
    require_episode_column(episodes)
    selected_fields = _normalize_fields(fields)
    selected_columns = _normalize_include_columns(episodes, include_columns)
    camera_targets = _normalize_camera_selectors(align_cameras) if align_cameras else ()

    row_fields: dict[str, DataType] = {
        "frame_index": DataType.int64(),
        "timestamp_us": DataType.int64(),
    }
    for field in selected_fields:
        row_fields[field] = DataType.tensor(DataType.float32())
    for camera, stream in camera_targets:
        name = _stream_name(camera, stream)
        row_fields[f"{name}/frame_index"] = DataType.int64()
        row_fields[f"{name}/timestamp_residual_us"] = DataType.int64()

    def expand_frames(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        timestamp_path = h5path("observation", "aligned_timestamp")
        timestamp_node = get_node(h5, timestamp_path)
        if timestamp_node is None:
            raise KeyError(f"Required OmniSharing frame clock is missing: {timestamp_path}")
        if np.dtype(timestamp_node.dtype) != np.dtype("int64") or len(timestamp_node.shape) != 1:
            raise ValueError(f"{timestamp_path} must be a rank-1 int64 clock")
        timestamps = np.asarray(timestamp_node[()])
        frame_count = int(timestamps.shape[0])

        arrays: dict[str, Any] = {}
        for field in selected_fields:
            dataset_path = h5path(field, "data")
            node = get_node(h5, dataset_path)
            if node is None:
                raise KeyError(f"Required OmniSharing frame field is missing: {dataset_path}")
            if np.dtype(node.dtype) != np.dtype("float32") or len(node.shape) != 2:
                raise ValueError(f"{dataset_path} must be a rank-2 float32 signal")
            if int(node.shape[0]) != frame_count:
                raise ValueError(f"{dataset_path} has {node.shape[0]} frames but aligned_timestamp has {frame_count}")
            arrays[field] = node[()]

        clocks: dict[str, list[int]] = {}
        for camera, stream in camera_targets:
            camera_group = get_node(h5, h5path("observation", "image", camera))
            if camera_group is None:
                continue
            clock_node = get_node(camera_group, f"{stream}_timestamp" if stream else "timestamp")
            if clock_node is None:
                clock_node = get_node(camera_group, "timestamp")
            if clock_node is None:
                continue
            if np.dtype(clock_node.dtype) != np.dtype("int64") or len(clock_node.shape) != 1:
                raise ValueError(f"{clock_node.name} must be a rank-1 int64 camera clock")
            clocks[_stream_name(camera, stream)] = [int(value) for value in clock_node[()]]

        rows: list[dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps):
            row: dict[str, Any] = {"frame_index": index, "timestamp_us": int(timestamp)}
            for field, values in arrays.items():
                # OmniSharing action arrays describe the following state.
                # Advance exactly once and repeat the final action.
                source_index = min(index + 1, frame_count - 1) if field.startswith("action/") else index
                row[field] = values[source_index]
            for camera, stream in camera_targets:
                name = _stream_name(camera, stream)
                clock = clocks.get(name)
                if not clock:
                    row[f"{name}/frame_index"] = None
                    row[f"{name}/timestamp_residual_us"] = None
                    continue
                camera_index, residual = nearest_timestamp(clock, int(timestamp))
                row[f"{name}/frame_index"] = camera_index
                row[f"{name}/timestamp_residual_us"] = residual
            rows.append(row)
        return {"_frame": rows}

    frame_fields = {"_frame": DataType.list(DataType.struct(row_fields))}
    expanded = read_episodes(episodes, frame_fields, expand_frames, columns=selected_columns).explode("_frame")
    return expanded.select(*selected_columns, unnest(col("_frame")))


__all__ = ["FRAME_FIELDS", "frames", "nearest_timestamp"]
