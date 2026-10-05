"""Typed episode metadata and signal materializers."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from daft.datatype import DataType

from ._common import (
    BRANCHES,
    HANDPOSE_ORDER,
    SIDES,
    attrs,
    get_node,
    h5path,
    read_episodes,
    require_episode_column,
    to_python,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from daft.dataframe import DataFrame

_OBJECT_GROUP_RE = re.compile(r"obj\d+")

_METADATA_FIELDS: dict[str, DataType] = {
    "generated_time": DataType.string(),
    "data_id": DataType.string(),
    "vendor": DataType.string(),
    "instruction": DataType.string(),
    "frame_count": DataType.int64(),
    "audio_sample_rate": DataType.int64(),
    "audio_sample_count": DataType.int64(),
    "camera_names": DataType.list(DataType.string()),
    "object_names": DataType.list(DataType.string()),
    "task_labels_json": DataType.string(),
}

TRAJECTORY_FIELDS: tuple[str, ...] = tuple(
    f"{branch}/{side}/{signal}" for branch in BRANCHES for side in SIDES for signal in ("joints", "handpose")
)
DEFAULT_TRAJECTORY_FIELDS: tuple[str, ...] = TRAJECTORY_FIELDS

_TACTILE_LAYOUT_DTYPE = DataType.struct(
    {"name": DataType.string(), "offset": DataType.int64(), "width": DataType.int64()}
)
_TACTILE_SENSOR_DTYPE = DataType.struct(
    {
        "name": DataType.string(),
        "offset": DataType.int64(),
        "width": DataType.int64(),
        "values": DataType.tensor(DataType.float32()),
    }
)
_AUDIO_FIELDS: dict[str, DataType] = {
    "audio/waveform": DataType.tensor(DataType.float64()),
    "audio/sample_rate": DataType.int64(),
    "audio/sample_count": DataType.int64(),
}
OBJECT_POSE_WIDTH = 17
_OBJECT_DTYPE = DataType.struct(
    {
        "index": DataType.int64(),
        "name": DataType.string(),
        "id": DataType.int64(),
        "layout": DataType.string(),
        "pose": DataType.tensor(DataType.float32()),
    }
)


def _normalize_selector(values: str | Sequence[str], *, name: str, valid: Sequence[str]) -> tuple[str, ...]:
    selected = (values,) if isinstance(values, str) else tuple(values)
    if not selected:
        raise ValueError(f"{name} must contain at least one value")
    unknown = [value for value in selected if value not in valid]
    if unknown:
        raise ValueError(f"Unknown {name} value(s): {unknown}. Valid values are: {', '.join(valid)}")
    return tuple(dict.fromkeys(selected))


def _object_group_names(observation: Any) -> list[str]:
    if observation is None:
        return []
    names = (str(name) for name in observation if _OBJECT_GROUP_RE.fullmatch(str(name)))
    return sorted(names, key=lambda name: int(name[3:]))


def _joint_names_field(field: str) -> str:
    """Column name for a joints field's names, e.g. ``observation/lefthand/joint_names``."""
    return f"{field.removesuffix('/joints')}/joint_names"


def episode_metadata(episodes: DataFrame) -> DataFrame:
    """Append fixed, typed episode metadata; absent optional values stay empty/null."""
    require_episode_column(episodes)

    def read_metadata(h5: Any) -> dict[str, Any]:
        root = get_node(h5, h5path())
        meta = get_node(h5, h5path("meta"))
        audio_node = get_node(h5, h5path("observation", "audio"))
        aligned = get_node(h5, h5path("observation", "aligned_timestamp"))
        image = get_node(h5, h5path("observation", "image"))
        observation = get_node(h5, h5path("observation"))

        root_attrs = attrs(root) if root is not None else {}
        meta_attrs = attrs(meta) if meta is not None else {}
        vendor = meta_attrs.pop("vendor", None)
        audio_attrs = attrs(audio_node) if audio_node is not None else {}

        return {
            "generated_time": root_attrs.get("generated_time"),
            "data_id": root_attrs.get("data_id"),
            "vendor": vendor,
            "instruction": audio_attrs.get("txt"),
            "frame_count": int(aligned.shape[0]) if aligned is not None and aligned.shape else None,
            "audio_sample_rate": (
                int(audio_attrs["samplerate"]) if audio_attrs.get("samplerate") is not None else None
            ),
            "audio_sample_count": (int(audio_node.shape[0]) if audio_node is not None and audio_node.shape else None),
            "camera_names": sorted(str(name) for name in image) if image is not None else [],
            "object_names": _object_group_names(observation),
            "task_labels_json": json.dumps(meta_attrs, ensure_ascii=False, sort_keys=True),
        }

    return read_episodes(episodes, _METADATA_FIELDS, read_metadata)


def _pose_order(value: Any) -> tuple[str, ...]:
    text = str(to_python(value) or "").strip().strip("[]")
    return tuple(part.strip().strip("'\"") for part in text.split(",") if part.strip())


def trajectory(
    episodes: DataFrame,
    fields: str | Sequence[str] = DEFAULT_TRAJECTORY_FIELDS,
) -> DataFrame:
    """Append only the requested slash-addressed Float32 trajectory tensors.

    Each ``.../joints`` field also gets a ``.../joint_names`` list column, read
    from the dataset's ``joint_names`` attribute. Joint order and width differ by
    stage (29 glove joints in DF-2, 17 in DF-2R), so index joints by name.
    """
    require_episode_column(episodes)
    selected = _normalize_selector(fields, name="trajectory field", valid=TRAJECTORY_FIELDS)
    return_fields: dict[str, DataType] = {}
    for field in selected:
        return_fields[field] = DataType.tensor(DataType.float32())
        if field.endswith("/joints"):
            return_fields[_joint_names_field(field)] = DataType.list(DataType.string())

    def read_trajectory(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        result: dict[str, Any] = {}
        frame_count: int | None = None
        for field in selected:
            dataset_path = h5path(field, "data")
            node = get_node(h5, dataset_path)
            if node is None:
                raise KeyError(f"Required OmniSharing trajectory dataset is missing: {dataset_path}")
            if np.dtype(node.dtype) != np.dtype("float32"):
                raise TypeError(f"{dataset_path} must have dtype float32, found {node.dtype}")
            if len(node.shape) != 2 or not node.shape[1]:
                raise ValueError(f"{dataset_path} must be a non-empty rank-2 trajectory, found {node.shape}")
            if field.endswith("/handpose"):
                if int(node.shape[1]) != len(HANDPOSE_ORDER):
                    raise ValueError(f"{dataset_path} must be width {len(HANDPOSE_ORDER)}, found {node.shape[1]}")
                pose_group = h5[h5path(*field.split("/"))]
                actual_order = _pose_order(attrs(pose_group).get("order"))
                if actual_order != HANDPOSE_ORDER:
                    raise ValueError(
                        f"{pose_group.name} hand-pose order must be {HANDPOSE_ORDER}, found {actual_order}"
                    )
            else:
                names = [str(name) for name in attrs(node).get("joint_names") or []]
                if names and len(names) != int(node.shape[1]):
                    raise ValueError(
                        f"{dataset_path} declares {len(names)} joint_names for {node.shape[1]} joint columns"
                    )
                result[_joint_names_field(field)] = names
            length = int(node.shape[0])
            if frame_count is None:
                frame_count = length
            elif length != frame_count:
                raise ValueError(
                    f"Requested trajectory fields have inconsistent frame counts: expected {frame_count}, "
                    f"but {dataset_path} has {length}"
                )
            result[field] = node[()]
        return result

    return read_episodes(episodes, return_fields, read_trajectory)


def tactile(
    episodes: DataFrame,
    sides: str | Sequence[str] = SIDES,
    *,
    split_by_sensor: bool = False,
) -> DataFrame:
    """Read tactile vectors with a stable nested sensor layout."""
    require_episode_column(episodes)
    selected = _normalize_selector(sides, name="tactile side", valid=SIDES)
    return_fields: dict[str, DataType] = {}
    for side in selected:
        prefix = f"observation/{side}/tactile"
        if split_by_sensor:
            return_fields[f"{prefix}_sensors"] = DataType.list(_TACTILE_SENSOR_DTYPE)
        else:
            return_fields[prefix] = DataType.tensor(DataType.float32())
            return_fields[f"{prefix}_layout"] = DataType.list(_TACTILE_LAYOUT_DTYPE)

    def read_tactile(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        result: dict[str, Any] = {}
        for side in selected:
            prefix = f"observation/{side}/tactile"
            dataset_path = h5path(prefix, "data")
            node = get_node(h5, dataset_path)
            if node is None:
                raise KeyError(f"Required OmniSharing tactile dataset is missing: {dataset_path}")
            if np.dtype(node.dtype) != np.dtype("float32"):
                raise TypeError(f"{dataset_path} must have dtype float32, found {node.dtype}")
            if len(node.shape) != 2:
                raise ValueError(f"{dataset_path} must be rank 2, found {node.shape}")

            node_attrs = attrs(node)
            names = [str(value) for value in node_attrs.get("sensor_names", [])]
            widths = [int(value) for value in node_attrs.get("sensor_lengths", [])]
            if not names or len(names) != len(widths):
                raise ValueError(f"{dataset_path} sensor_names and sensor_lengths must be non-empty and equally sized")
            if len(set(names)) != len(names):
                raise ValueError(f"{dataset_path} sensor_names must be unique, found {names}")
            if any(width <= 0 for width in widths):
                raise ValueError(f"{dataset_path} sensor widths must all be positive, found {widths}")
            stored_width = int(node.shape[1])
            if sum(widths) != stored_width:
                raise ValueError(
                    f"{dataset_path} declared sensor widths total {sum(widths)}, but data is {stored_width} wide"
                )

            layout: list[dict[str, Any]] = []
            offset = 0
            if split_by_sensor:
                values = node[()]
                for name, width in zip(names, widths):
                    layout.append(
                        {
                            "name": name,
                            "offset": offset,
                            "width": width,
                            "values": values[:, offset : offset + width],
                        }
                    )
                    offset += width
                result[f"{prefix}_sensors"] = layout
            else:
                for name, width in zip(names, widths):
                    layout.append({"name": name, "offset": offset, "width": width})
                    offset += width
                result[prefix] = node[()]
                result[f"{prefix}_layout"] = layout
        return result

    return read_episodes(episodes, return_fields, read_tactile)


def audio(
    episodes: DataFrame,
    *,
    mono: bool = False,
    max_seconds: float | None = None,
) -> DataFrame:
    """Read optional Float64 PCM audio, slicing bounded previews at read time."""
    require_episode_column(episodes)
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError(f"max_seconds must be positive, got {max_seconds}")

    def read_audio(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        empty = {
            "audio/waveform": np.empty((0,), dtype="float64"),
            "audio/sample_rate": None,
            "audio/sample_count": None,
        }
        dataset_path = h5path("observation", "audio")
        node = get_node(h5, dataset_path)
        if node is None:
            return empty
        if np.dtype(node.dtype) != np.dtype("float64"):
            raise TypeError(f"{dataset_path} must have dtype float64, found {node.dtype}")
        node_attrs = attrs(node)
        sample_rate_value = node_attrs.get("samplerate")
        sample_rate = int(sample_rate_value) if sample_rate_value is not None else None
        stop = None
        if max_seconds is not None and sample_rate is not None:
            stop = max(1, int(max_seconds * sample_rate))
        waveform = np.asarray(node[:stop] if stop is not None else node[()], dtype="float64")
        if mono and waveform.ndim > 1:
            waveform = waveform.mean(axis=tuple(range(1, waveform.ndim)), dtype="float64")
        return {
            "audio/waveform": waveform,
            "audio/sample_rate": sample_rate,
            "audio/sample_count": int(waveform.shape[0]) if waveform.ndim else 1,
        }

    return read_episodes(episodes, _AUDIO_FIELDS, read_audio)


def objects(episodes: DataFrame) -> DataFrame:
    """Read variable object tracks into one stable list-of-structs column."""
    require_episode_column(episodes)
    return_fields = {"n_objects": DataType.int64(), "objects": DataType.list(_OBJECT_DTYPE)}

    def read_objects(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        rows: list[dict[str, Any]] = []
        for group_name in _object_group_names(get_node(h5, h5path("observation"))):
            dataset_path = h5path("observation", group_name, "data")
            node = get_node(h5, dataset_path)
            if node is None:
                raise KeyError(f"Present object group is missing its required dataset: {dataset_path}")
            if np.dtype(node.dtype) != np.dtype("float32"):
                raise TypeError(f"{dataset_path} must have dtype float32, found {node.dtype}")
            if len(node.shape) != 2 or int(node.shape[1]) != OBJECT_POSE_WIDTH:
                raise ValueError(f"{dataset_path} must have shape (frames, {OBJECT_POSE_WIDTH}), found {node.shape}")
            node_attrs = attrs(node)
            object_id = node_attrs.get("obj_id")
            rows.append(
                {
                    "index": int(group_name[3:]),
                    "name": node_attrs.get("obj_name"),
                    "id": int(object_id) if object_id is not None else None,
                    "layout": node_attrs.get("order") or node_attrs.get("detail"),
                    "pose": node[()],
                }
            )
        return {"n_objects": len(rows), "objects": rows}

    return read_episodes(episodes, return_fields, read_objects)


__all__ = [
    "DEFAULT_TRAJECTORY_FIELDS",
    "OBJECT_POSE_WIDTH",
    "TRAJECTORY_FIELDS",
    "audio",
    "episode_metadata",
    "objects",
    "tactile",
    "trajectory",
]
