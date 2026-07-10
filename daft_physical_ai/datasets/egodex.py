"""Lazy access to a locally extracted `EgoDex <https://github.com/apple/ml-egodex>`_ release.

EgoDex is distributed as large ZIP archives under CC-BY-NC-ND. This module
intentionally does not download or extract those archives: point :func:`raw`
at a copy that you have downloaded and extracted yourself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import daft
from daft.datatype import DataType
from daft.expressions import col, lit
from daft.functions import (  # type: ignore[attr-defined]
    file_exists,
    hdf5_file,
    regexp_replace,
    video_file,
    video_frames,
    when,
)

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.file.hdf5 import Hdf5File
    from daft.io import IOConfig


# The 68 upper-body and hand joints tracked with a per-frame confidence score.
JOINTS: tuple[str, ...] = (
    "hip",
    "leftArm",
    "leftForearm",
    "leftHand",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerKnuckle",
    "leftIndexFingerMetacarpal",
    "leftIndexFingerTip",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerKnuckle",
    "leftLittleFingerMetacarpal",
    "leftLittleFingerTip",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerMetacarpal",
    "leftMiddleFingerTip",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerKnuckle",
    "leftRingFingerMetacarpal",
    "leftRingFingerTip",
    "leftShoulder",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbKnuckle",
    "leftThumbTip",
    "neck1",
    "neck2",
    "neck3",
    "neck4",
    "rightArm",
    "rightForearm",
    "rightHand",
    "rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip",
    "rightIndexFingerKnuckle",
    "rightIndexFingerMetacarpal",
    "rightIndexFingerTip",
    "rightLittleFingerIntermediateBase",
    "rightLittleFingerIntermediateTip",
    "rightLittleFingerKnuckle",
    "rightLittleFingerMetacarpal",
    "rightLittleFingerTip",
    "rightMiddleFingerIntermediateBase",
    "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerKnuckle",
    "rightMiddleFingerMetacarpal",
    "rightMiddleFingerTip",
    "rightRingFingerIntermediateBase",
    "rightRingFingerIntermediateTip",
    "rightRingFingerKnuckle",
    "rightRingFingerMetacarpal",
    "rightRingFingerTip",
    "rightShoulder",
    "rightThumbIntermediateBase",
    "rightThumbIntermediateTip",
    "rightThumbKnuckle",
    "rightThumbTip",
    "spine1",
    "spine2",
    "spine3",
    "spine4",
    "spine5",
    "spine6",
    "spine7",
)

TRANSFORM_JOINTS: tuple[str, ...] = ("camera", *JOINTS)
TRAJECTORY_FIELDS: tuple[str, ...] = (
    "camera/intrinsic",
    *(f"confidences/{joint}" for joint in JOINTS),
    *(f"transforms/{joint}" for joint in TRANSFORM_JOINTS),
)
_TRAJECTORY_DTYPES = {field: DataType.tensor(DataType.float32()) for field in TRAJECTORY_FIELDS}

DEFAULT_TRAJECTORY_FIELDS: tuple[str, ...] = (
    "camera/intrinsic",
    "transforms/camera",
    "transforms/leftHand",
    "transforms/rightHand",
)

_METADATA_FIELD_DTYPES: dict[str, DataType] = {
    "task": DataType.string(),
    "llm_description": DataType.string(),
    "llm_description2": DataType.string(),
    "which_llm_description": DataType.string(),
    "llm_type": DataType.string(),
    "llm_verbs": DataType.list(DataType.string()),
    "llm_objects": DataType.list(DataType.string()),
    "environment": DataType.string(),
    "object": DataType.string(),
    "session_name": DataType.string(),
    "annotated": DataType.bool(),
    "annotator_version": DataType.string(),
    "extra": DataType.string(),
    "description": DataType.string(),
    "description2": DataType.string(),
    "type": DataType.string(),
}
_METADATA_DTYPE = DataType.struct(_METADATA_FIELD_DTYPES)
_METADATA_FIELDS = tuple(_METADATA_FIELD_DTYPES)
_LIST_METADATA_FIELDS = frozenset(("llm_verbs", "llm_objects"))


def _attr_value(value: object) -> object:
    """Coerce h5py attribute values (NumPy scalars/arrays, bytes) to Python."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "ndim"):
        array_value = cast("Any", value)
        if array_value.ndim == 0:
            return _attr_value(array_value.item())
        return [_attr_value(item) for item in array_value.tolist()]
    if hasattr(value, "item"):
        return _attr_value(cast("Any", value).item())
    return value


def _as_str_list(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


def _as_int_list(value: int | Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [value] if isinstance(value, int) else list(value)


def raw(
    path: str,
    io_config: IOConfig | None = None,
    *,
    tasks: str | Sequence[str] | None = None,
    episode_ids: int | Sequence[int] | None = None,
) -> DataFrame:
    """Catalog an extracted EgoDex release as a lazy, episode-level DataFrame.

    ``path`` may be the extracted root, a split directory, or one task
    directory. Output has one row per HDF5 file: ``task``, ``episode_id``, a
    typed ``metadata`` struct, lazy ``trajectory`` HDF5 and ``video`` MP4 file
    columns. The sibling video is null when it is absent.

    The data is not redistributed by this package. Download and extract it
    directly from Apple first; see https://github.com/apple/ml-egodex.
    """
    task_values = _as_str_list(tasks)
    episode_id_values = _as_int_list(episode_ids)

    @daft.func(return_dtype=_METADATA_DTYPE, use_process=False)
    def read_egodex_metadata(file: Hdf5File) -> dict[str, object]:
        attrs = file.attrs()
        values = {name: _attr_value(attrs.get(name)) for name in _METADATA_FIELDS}
        # Daft cannot form a list-typed struct field when every value is null.
        for name in _LIST_METADATA_FIELDS:
            if values[name] is None:
                values[name] = []
        return values

    episodes = daft.from_glob_path(f"{path.rstrip('/')}/**/*.hdf5", io_config=io_config).select(
        "path",
        col("path").split("/")[-2].alias("task"),
        col("path").split("/")[-1].split(".")[0].cast(DataType.int64()).alias("episode_id"),
    )
    if task_values is not None:
        episodes = episodes.where(col("task").is_in(task_values))
    if episode_id_values is not None:
        episodes = episodes.where(col("episode_id").is_in(episode_id_values))

    return (
        episodes.select(
            "task",
            "episode_id",
            hdf5_file(col("path"), io_config=io_config).alias("trajectory"),
            video_file(regexp_replace(col("path"), r"\.hdf5$", ".mp4"), io_config=io_config).alias("video"),
        )
        .with_column("video", when(file_exists(col("video")), col("video")).otherwise(lit(None)))
        .with_column("metadata", cast("Any", read_egodex_metadata)(col("trajectory")))
        .select("task", "episode_id", "metadata", "trajectory", "video")
    )


def trajectory(episodes: DataFrame, fields: Sequence[str] = DEFAULT_TRAJECTORY_FIELDS) -> DataFrame:
    """Read selected EgoDex HDF5 pose datasets as tensor columns.

    Filter and limit ``episodes`` before calling this function. Each requested
    field is read lazily from the annotation file; output stays one row per
    episode. ``camera/intrinsic`` is ``(3, 3)``, transforms are ``(N, 4, 4)``,
    and confidences are ``(N,)`` float32 tensors. Confidence datasets are
    optional in EgoDex, so they are not read by default; request them explicitly
    with ``fields`` when the selected files contain them.
    """
    from daft.dependencies import h5py  # type: ignore[attr-defined]

    if not h5py.module_available():  # ty:ignore[unresolved-attribute]
        raise ImportError("EgoDex trajectories require daft[hdf5].")
    if "trajectory" not in episodes.schema().column_names():
        raise ValueError("Expected an episode DataFrame with a `trajectory` column.")

    fields = tuple(fields)
    if not fields:
        raise ValueError("fields must contain at least one HDF5 dataset path")
    unknown = [field for field in fields if field not in _TRAJECTORY_DTYPES]
    if unknown:
        raise ValueError(f"Unknown trajectory field(s): {unknown}")

    @daft.func(
        return_dtype=DataType.struct({field: _TRAJECTORY_DTYPES[field] for field in fields}),
        use_process=False,
        unnest=True,
    )
    def read_egodex_trajectory(file: Hdf5File) -> dict[str, object]:
        with file.to_tempfile() as temporary_file, h5py.File(temporary_file.name, "r") as h5:
            return {field: h5[field][()] for field in fields}

    return episodes.where(col("trajectory").not_null()).select(
        "task", "episode_id", "metadata", cast("Any", read_egodex_trajectory)(col("trajectory")), "video"
    )


def camera_frames(
    episodes: DataFrame,
    *,
    start_time: float = 0,
    end_time: float | None = None,
    width: int | None = None,
    height: int | None = None,
    is_key_frame: bool | None = None,
    sample_interval_seconds: float | None = None,
) -> DataFrame:
    """Append a lazy ``video_frames`` list column to EgoDex episodes."""
    from daft.dependencies import av

    if not cast("Any", av).module_available():
        raise ImportError("EgoDex video decoding requires daft[video].")
    if "video" not in episodes.schema().column_names():
        raise ValueError("Expected an episode DataFrame with an EgoDex `video` column.")

    return episodes.with_column(
        "video_frames",
        video_frames(
            col("video"),
            start_time=start_time,
            end_time=end_time,
            width=width,
            height=height,
            is_key_frame=is_key_frame,
            sample_interval_seconds=sample_interval_seconds,
        ),
    )


__all__ = [
    "DEFAULT_TRAJECTORY_FIELDS",
    "JOINTS",
    "TRAJECTORY_FIELDS",
    "TRANSFORM_JOINTS",
    "camera_frames",
    "raw",
    "trajectory",
]
