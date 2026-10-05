"""Lazy access to `HIW-500 <https://huggingface.co/datasets/BitRobot/HIW-500>`_ humanoid episodes stored as MCAP.

HIW-500 ("Humanoids In-the-Wild", BitRobot Foundation, CC-BY-4.0) is whole-body
teleoperation of a Unitree G1 in real homes. The raw release is ROS 2 (Humble)
bags in MCAP, one directory per episode::

    <Task-Name>/episode_<YYYY-MM-DD_HH-MM-SS>/episode_<NNNN>/
        episode_<NNNN>.mcap
        info.json                                  # task, timing, scene, subtask labels
        calibration/params/camera_<serial>.json    # RealSense wrist cameras
        calibration/params/head_camera_params.yaml # head stereo pair

Every MCAP message is ROS 2 CDR. :func:`messages` returns the raw CDR bytes for
any topic; :func:`joint_states`, :func:`wbc_states`, and :func:`camera_frames`
decode the joint-state, whole-body-controller, and JPEG camera topics. The
episode-level json/yaml files are parsed by :func:`info`, :func:`subtasks`, and
:func:`calibration`.

A LeRobot v3.0 conversion (``BitRobot/HIW-500-LeRobot``) is read natively by
``daft.datasets.lerobot.read``; :func:`lerobot_subtask` adds the active subtask
label per frame.

Requires the ``hiw500`` extra (``pip install 'daft-physical-ai[hiw500]'``).
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import daft
from daft.datatype import DataType
from daft.expressions import col, lit
from daft.functions import (
    decode_image,
    download,
    file,
    image_height,
    image_width,
    list_filter,
    list_sort,
    regexp_extract,
    resize,
    to_datetime,
    when,
)

from daft_physical_ai.datasets import _mcap

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.expressions import Expression
    from daft.io import IOConfig

HF_DATASET = "hf://datasets/BitRobot/HIW-500"
LEROBOT_REPO = "BitRobot/HIW-500-LeRobot"

LOWSTATE_TOPIC = "/stamped/lowstate"
WBC_TOPIC = "/wbc_lerobot"
ANNOTATION_TOPIC = "/annotation"

# Every non-camera topic in the release. The default keeps a bare messages()
# call away from the JPEG streams, which hold most of each file's bytes.
DEFAULT_MESSAGE_TOPICS: tuple[str, ...] = (
    "/stamped/lowstate",
    "/stamped/lowcmd",
    "/stamped/secondary_imu",
    "/stamped/dex1/left/state",
    "/stamped/dex1/right/state",
    "/stamped/dex1/left/cmd",
    "/stamped/dex1/right/cmd",
    "/lf/odommodestate",
    "/wbc_lerobot",
    "/annotation",
)

CAMERA_TOPICS: dict[str, str] = {
    "head": "/camera/head/image/compressed",
    "left_wrist": "/camera/left_wrist/image/compressed",
    "right_wrist": "/camera/right_wrist/image/compressed",
    "left_wrist_ir1": "/camera/left_wrist/ir1/compressed",
    "left_wrist_ir2": "/camera/left_wrist/ir2/compressed",
    "right_wrist_ir1": "/camera/right_wrist/ir1/compressed",
    "right_wrist_ir2": "/camera/right_wrist/ir2/compressed",
}

# The 29 body joints, in Unitree G1 motor order: the first 29 of LowState's 35
# motor slots, and the order of the LeRobot release's ``observation.state``.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

_LOWSTATE_MOTOR_SLOTS = 35
_NUM_JOINTS = len(JOINT_NAMES)

_CATALOG_COLUMNS: tuple[str, ...] = (
    "task",
    "session",
    "session_time",
    "episode_number",
    "episode_id",
    "episode_dir",
    "mcap_path",
    "mcap_size",
    "info_path",
    "calibration_paths",
    "episode_mcap",
)

_IDENTITY_COLUMNS: tuple[str, ...] = ("task", "session", "episode_number", "episode_id")

# <root>/<Task>/episode_<session>/episode_<NNNN>/<file>
_LAYOUT_PATTERN = r"/([^/]+)/episode_([^/]+)/episode_(\d+)/"
_EPISODE_DIR_PATTERN = r"^(.*/episode_[^/]+/episode_\d+)/"
_EPISODE_ID_PATTERN = r"([^/]+/episode_[^/]+/episode_\d+)/"
_REST_PATTERN = r"/episode_[^/]+/episode_\d+/(.+)$"
_SESSION_TIME_PATTERN = r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
_SESSION_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

_LISTING_DTYPE = DataType.struct({"path": DataType.string(), "size": DataType.int64()})


def _floats(size: int, dtype: DataType | None = None) -> DataType:
    return DataType.fixed_size_list(dtype or DataType.float32(), size)


_JOINT_DTYPE = DataType.struct(
    {
        "stamp_ns": DataType.int64(),
        "tick": DataType.int64(),
        "mode_machine": DataType.int64(),
        "q": _floats(_NUM_JOINTS),
        "dq": _floats(_NUM_JOINTS),
        "tau_est": _floats(_NUM_JOINTS),
        "imu_quaternion": _floats(4),
        "imu_gyroscope": _floats(3),
        "imu_accelerometer": _floats(3),
        "imu_rpy": _floats(3),
    }
)

_WBC_DTYPE = DataType.struct(
    {
        "pivot": _floats(7, DataType.float64()),
        "ee_state": _floats(12, DataType.float64()),
        "ee_action": _floats(12, DataType.float64()),
        "left_trigger": DataType.float64(),
        "left_squeeze": DataType.float64(),
        "right_trigger": DataType.float64(),
        "right_squeeze": DataType.float64(),
    }
)

_IMAGE_DTYPE = DataType.struct(
    {
        "stamp_ns": DataType.int64(),
        "frame_id": DataType.string(),
        "format": DataType.string(),
        "encoded": DataType.binary(),
    }
)

_INFO_DTYPE = DataType.struct(
    {
        "episode_name": DataType.string(),
        "task_name": DataType.string(),
        "scene": DataType.int64(),
        "start_timestamp_ns": DataType.int64(),
        "end_timestamp_ns": DataType.int64(),
        "duration_ns": DataType.int64(),
        "duration_seconds": DataType.float64(),
        "subtask_count": DataType.int64(),
        "info_json": DataType.string(),
    }
)

_SUBTASK_DTYPE = DataType.struct(
    {
        "subtask_index": DataType.int64(),
        "label": DataType.string(),
        "start_timestamp_ns": DataType.int64(),
        "end_timestamp_ns": DataType.int64(),
        "start_offset_seconds": DataType.float64(),
        "end_offset_seconds": DataType.float64(),
    }
)

_CALIBRATION_DTYPE = DataType.struct(
    {
        "camera": DataType.string(),
        "stream": DataType.string(),
        "serial_number": DataType.string(),
        "width": DataType.int64(),
        "height": DataType.int64(),
        "fx": DataType.float64(),
        "fy": DataType.float64(),
        "cx": DataType.float64(),
        "cy": DataType.float64(),
        "distortion_model": DataType.string(),
        "distortion_coeffs": DataType.list(DataType.float64()),
        "extrinsics_reference": DataType.string(),
        "rotation": DataType.list(DataType.float64()),
        "translation": DataType.list(DataType.float64()),
        "params_json": DataType.string(),
    }
)

_METADATA_DTYPE = DataType.struct(
    {
        "ros_distro": DataType.string(),
        "message_count": DataType.int64(),
        "message_start_time": DataType.int64(),
        "message_end_time": DataType.int64(),
        "chunk_count": DataType.int64(),
        "indexed": DataType.bool(),
        "topics": DataType.list(DataType.string()),
        "channels": DataType.list(
            DataType.struct(
                {"topic": DataType.string(), "schema_name": DataType.string(), "message_count": DataType.int64()}
            )
        ),
    }
)


def _require_mcap() -> Any:
    return _mcap.require_mcap("HIW-500", "hiw500")


def _require_columns(episodes: DataFrame, *columns: str) -> None:
    _mcap.require_columns(episodes, *columns, source="hiw500.raw()")


def _passthrough(episodes: DataFrame, added: DataType) -> list[str]:
    """Input columns to keep next to ``added``'s fields; same-named inputs are replaced."""
    return [name for name in episodes.schema().column_names() if name not in added.fields]


def _resolve_io_config(io_config: IOConfig | None, paths: Sequence[str] = (HF_DATASET,)) -> IOConfig | None:
    # HIW-500 is not gated; a token only raises the Hub rate limits.
    return _mcap.resolve_hf_io_config(io_config, paths)


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def _hf_tasks(repo_id: str, revision: str | None, token: str | bool | None) -> list[str]:
    """Task folders at the repository root (one non-recursive listing call)."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    api = HfApi(token=token, library_name=_mcap.LIBRARY_NAME)
    entries = api.list_repo_tree(repo_id, recursive=False, revision=revision, repo_type="dataset")
    return sorted(
        entry.path
        for entry in entries
        if isinstance(entry, RepoFolder) and entry.path != "assets" and not entry.path.startswith(".")
    )


def _list_hf_task(
    root: str, repo_id: str, revision: str | None, token: str | bool | None, task: str
) -> list[dict[str, object]]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi(token=token, library_name=_mcap.LIBRARY_NAME)
    try:
        return [
            {"path": f"{root}/{entry.path}", "size": entry.size}
            for entry in api.list_repo_tree(
                repo_id, path_in_repo=task, recursive=True, revision=revision, repo_type="dataset"
            )
            if isinstance(entry, RepoFile)
        ]
    except EntryNotFoundError:
        return []


def _hf_listing(root: str, *, tasks: tuple[str, ...] | None, io_config: IOConfig | None) -> DataFrame:
    """List HIW-500 objects lazily, one recursive-tree walk per task.

    Only the task folder names are listed eagerly (a single call, when ``tasks``
    is not given). Each task's recursive walk runs when the plan executes, the
    walks run concurrently, and unselected tasks are never walked.
    """
    _mcap.require_hf_hub("HIW-500", "hiw500")
    repo_id, revision = _mcap.parse_hf_root(root, example=HF_DATASET)
    token = _mcap.hf_token(io_config)
    selected = list(tasks) if tasks is not None else _hf_tasks(repo_id, revision, token)
    if not selected:
        return _mcap.empty_listing()

    @daft.func(return_dtype=DataType.list(_LISTING_DTYPE), use_process=False)
    async def list_task(task: str) -> list[dict[str, object]]:
        return await asyncio.to_thread(_list_hf_task, root, repo_id, revision, token, task)

    return (
        daft.from_pydict({"task": selected})
        .select(cast("Any", list_task)(col("task")).alias("entry"))
        .explode("entry")
        .where(col("entry").not_null())
        .select(col("entry")["path"].alias("path"), col("entry")["size"].alias("size"))
    )


def _glob_listing(root: str, *, tasks: tuple[str, ...] | None, io_config: IOConfig | None) -> DataFrame:
    patterns = [f"{root}/{task}/episode_*/episode_*/**" for task in (tasks or ("*",))]
    return daft.from_glob_path(patterns, io_config=io_config).select("path", "size")


def raw(
    path: str = HF_DATASET,
    *,
    tasks: str | Sequence[str] | None = None,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Catalog HIW-500 episodes as a lazy, episode-level DataFrame.

    Lists objects and parses paths only; nothing is downloaded. ``path``
    defaults to the Hugging Face dataset (append ``@<revision>`` to pin a
    commit) and may also be a local mirror or another object store. On Hugging
    Face, only the task folder names are listed when ``raw()`` is called; each
    task's recursive file listing runs when the plan executes, one task per
    partition. Other stores use Daft's lazy glob.

    Args:
        path: Dataset root.
        tasks: Optional task folder name or names, e.g. ``"Hang-Hanger"``. They
            narrow the listing, so unselected tasks are never walked.
        io_config: Storage configuration. For ``hf://`` paths without a token,
            ``HF_TOKEN`` or the ``hf auth login`` credential is used (the
            dataset is public; a token only raises rate limits).

    Returns:
        One row per episode: ``task`` (folder name), ``session`` (the
        ``YYYY-MM-DD_HH-MM-SS`` recording-session stamp), ``session_time``
        (that stamp as a naive local-time timestamp), ``episode_number``,
        ``episode_id`` (``<task>/episode_<session>/episode_<NNNN>``),
        ``episode_dir``, ``mcap_path``, ``mcap_size``, ``info_path``,
        ``calibration_paths`` (sorted camera json/yaml paths), and a lazy
        ``daft.File`` reference ``episode_mcap``.
    """
    selected_tasks = _mcap.normalize_names(tasks, name="tasks")
    if selected_tasks is not None and any("/" in task or "*" in task for task in selected_tasks):
        raise ValueError("tasks must be exact task folder names, without '/' or glob wildcards")

    root = path.rstrip("/")
    io_config = _mcap.resolve_hf_io_config(io_config, (root,))
    listing = _hf_listing if root.startswith("hf://") else _glob_listing
    objects = listing(root, tasks=selected_tasks, io_config=io_config)

    path_col = col("path")
    rest = regexp_extract(path_col, _REST_PATTERN, 1)
    kind = (
        when(rest == "info.json", lit("info"))
        .when(regexp_extract(rest, r"^(calibration/params/[^/]+\.(?:json|yaml))$", 1).not_null(), lit("calibration"))
        .when(regexp_extract(rest, r"^(episode_\d+\.mcap)$", 1).not_null(), lit("mcap"))
        .otherwise(lit(None))
    )
    catalog = objects.select(
        "path",
        "size",
        kind.alias("__kind"),
        regexp_extract(path_col, _LAYOUT_PATTERN, 1).alias("task"),
        regexp_extract(path_col, _LAYOUT_PATTERN, 2).alias("session"),
        regexp_extract(path_col, _LAYOUT_PATTERN, 3).alias("__number"),
        regexp_extract(path_col, _EPISODE_DIR_PATTERN, 1).alias("episode_dir"),
        regexp_extract(path_col, _EPISODE_ID_PATTERN, 1).alias("episode_id"),
    ).where(col("__kind").not_null() & col("task").not_null())
    if selected_tasks is not None:
        catalog = catalog.where(col("task").is_in(list(selected_tasks)))

    def pick(kind_name: str, column: str) -> Expression:
        return when(col("__kind") == kind_name, col(column)).otherwise(lit(None)).any_value(ignore_nulls=True)

    calibration = when(col("__kind") == "calibration", col("path")).otherwise(lit(None)).list_agg()
    return (
        catalog.groupby("task", "session", "__number", "episode_dir", "episode_id")
        .agg(
            pick("mcap", "path").alias("mcap_path"),
            pick("mcap", "size").alias("mcap_size"),
            pick("info", "path").alias("info_path"),
            calibration.alias("calibration_paths"),
        )
        .where(col("mcap_path").not_null())
        .with_columns(
            {
                "session_time": to_datetime(
                    regexp_extract(col("session"), _SESSION_TIME_PATTERN, 1), _SESSION_TIME_FORMAT
                ),
                "episode_number": col("__number").cast(DataType.int64()),
                "calibration_paths": list_sort(list_filter(col("calibration_paths"), daft.element().not_null())),
                "episode_mcap": file(col("mcap_path"), io_config=io_config),
            }
        )
        .select(*_CATALOG_COLUMNS)
    )


# --------------------------------------------------------------------------- #
# info.json: episode info and subtask annotations
# --------------------------------------------------------------------------- #


def _int_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


@daft.func(return_dtype=_INFO_DTYPE, use_process=False, unnest=True)
def _parse_info(data: bytes | None) -> dict[str, object] | None:
    if data is None:
        return None
    info = json.loads(data)
    subtask_list = info.get("subtasks")
    duration = info.get("duration_sec")
    return {
        "episode_name": info.get("episode_name"),
        "task_name": info.get("task"),
        "scene": _int_or_none(info.get("scene")),
        "start_timestamp_ns": _int_or_none(info.get("start_timestamp_ns")),
        "end_timestamp_ns": _int_or_none(info.get("end_timestamp_ns")),
        "duration_ns": _int_or_none(info.get("duration_ns")),
        "duration_seconds": float(duration) if isinstance(duration, (int, float)) else None,
        "subtask_count": len(subtask_list) if isinstance(subtask_list, list) else None,
        "info_json": json.dumps(info, separators=(",", ":"), sort_keys=True),
    }


@daft.func(return_dtype=_SUBTASK_DTYPE, use_process=False, unnest=True)
def _parse_subtasks(data: bytes | None) -> Iterator[dict[str, object]]:
    if data is None:
        return
    info = json.loads(data)
    episode_start = _int_or_none(info.get("start_timestamp_ns"))
    episode_end = _int_or_none(info.get("end_timestamp_ns"))
    entries = [entry for entry in info.get("subtasks") or [] if isinstance(entry, dict)]
    starts = [_int_or_none(entry.get("timestamp_ns")) for entry in entries]
    for index, (entry, start) in enumerate(zip(entries, starts)):
        end = starts[index + 1] if index + 1 < len(starts) else episode_end

        def offset(value: int | None) -> float | None:
            if value is None or episode_start is None:
                return None
            return (value - episode_start) / 1e9

        yield {
            "subtask_index": index,
            "label": entry.get("task"),
            "start_timestamp_ns": start,
            "end_timestamp_ns": end,
            "start_offset_seconds": offset(start),
            "end_offset_seconds": offset(end),
        }


def _downloaded_info(episodes: DataFrame, io_config: IOConfig | None) -> DataFrame:
    _require_columns(episodes, "info_path", *_IDENTITY_COLUMNS)
    io_config = _resolve_io_config(io_config)
    return episodes.where(col("info_path").not_null()).with_column(
        "__info", download(col("info_path"), io_config=io_config)
    )


def info(episodes: DataFrame, *, io_config: IOConfig | None = None) -> DataFrame:
    """Parse each episode's ``info.json`` (one small download per episode).

    Returns:
        The input columns plus ``episode_name`` (the session directory name),
        ``task_name`` (natural-language task, e.g. ``"hang hanger"``),
        ``scene`` (home/scene id), ``start_timestamp_ns``,
        ``end_timestamp_ns``, ``duration_ns``, ``duration_seconds``,
        ``subtask_count``, and the full file as ``info_json``. Timestamps are
        Unix nanoseconds on the same clock as MCAP ``log_time``.
    """
    downloaded = _downloaded_info(episodes, io_config)
    return downloaded.select(*_passthrough(episodes, _INFO_DTYPE), cast("Any", _parse_info)(col("__info")))


def subtasks(episodes: DataFrame, *, io_config: IOConfig | None = None) -> DataFrame:
    """One row per subtask annotation in each episode's ``info.json``.

    Each label starts at its ``timestamp_ns`` and lasts until the next label,
    or the episode end for the last one. Episodes with no subtasks contribute
    no rows.

    Returns:
        Episode identity, ``subtask_index``, ``label``, ``start_timestamp_ns``,
        ``end_timestamp_ns``, and ``start_offset_seconds`` /
        ``end_offset_seconds`` relative to the episode start (comparable to the
        LeRobot release's per-episode ``timestamp``).
    """
    downloaded = _downloaded_info(episodes, io_config)
    return downloaded.select(*_IDENTITY_COLUMNS, cast("Any", _parse_subtasks)(col("__info"))).where(
        col("subtask_index").not_null()
    )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def _float_list(values: Any) -> list[float] | None:
    if not isinstance(values, list):
        return None
    flat: list[float] = []
    for value in values:
        if isinstance(value, list):
            flat.extend(float(item) for item in value)
        else:
            flat.append(float(value))
    return flat


def _realsense_rows(params: dict[str, Any]) -> Iterator[dict[str, object]]:
    position = params.get("position")
    camera = f"{position}_wrist" if position in ("left", "right") else position
    serial = params.get("serial_number")
    for stream in ("color", "ir1", "ir2"):
        entry = params.get(stream)
        if not isinstance(entry, dict):
            continue
        intrinsics = entry.get("intrinsics") or {}
        extrinsics = entry.get("extrinsics_to_color")
        yield {
            "camera": camera,
            "stream": stream,
            "serial_number": None if serial is None else str(serial),
            "width": _int_or_none(intrinsics.get("width")),
            "height": _int_or_none(intrinsics.get("height")),
            "fx": intrinsics.get("fx"),
            "fy": intrinsics.get("fy"),
            "cx": intrinsics.get("ppx"),
            "cy": intrinsics.get("ppy"),
            "distortion_model": intrinsics.get("model"),
            "distortion_coeffs": _float_list(intrinsics.get("coeffs")),
            "extrinsics_reference": None if extrinsics is None else "color",
            "rotation": None if extrinsics is None else _float_list(extrinsics.get("rotation")),
            "translation": None if extrinsics is None else _float_list(extrinsics.get("translation")),
        }


def _stereo_rows(params: dict[str, Any]) -> Iterator[dict[str, object]]:
    size = params.get("image_size") or [None, None]
    for side in ("left", "right"):
        matrix = params.get(f"camera_matrix_{side}")
        if not isinstance(matrix, list) or len(matrix) != 3:
            continue
        is_right = side == "right"
        yield {
            "camera": "head",
            "stream": side,
            "serial_number": None,
            "width": _int_or_none(size[0]),
            "height": _int_or_none(size[1]),
            "fx": float(matrix[0][0]),
            "fy": float(matrix[1][1]),
            "cx": float(matrix[0][2]),
            "cy": float(matrix[1][2]),
            "distortion_model": "plumb_bob",
            "distortion_coeffs": _float_list(params.get(f"dist_coeffs_{side}")),
            "extrinsics_reference": "left" if is_right else None,
            "rotation": _float_list(params.get("R")) if is_right else None,
            "translation": _float_list(params.get("T")) if is_right else None,
        }


@daft.func(return_dtype=_CALIBRATION_DTYPE, use_process=False, unnest=True)
def _parse_calibration(source_path: str, data: bytes | None) -> Iterator[dict[str, object]]:
    if data is None:
        return
    if source_path.endswith((".yaml", ".yml")):
        import yaml

        params = yaml.safe_load(data)
        rows = _stereo_rows(params) if isinstance(params, dict) else iter(())
    else:
        params = json.loads(data)
        rows = _realsense_rows(params) if isinstance(params, dict) else iter(())
    params_json = json.dumps(params, separators=(",", ":"), sort_keys=True)
    for row in rows:
        yield {**row, "params_json": params_json}


def calibration(episodes: DataFrame, *, io_config: IOConfig | None = None) -> DataFrame:
    """Per-camera, per-stream intrinsics and extrinsics from each episode's calibration files.

    Reads the RealSense wrist-camera json files (``color``, ``ir1``, ``ir2``
    streams) and the head stereo yaml (``left``, ``right``). Values are passed
    through as stored: wrist extrinsics map each IR stream to ``color``
    (RealSense ``rs2_extrinsics``: column-major rotation, metres), and the head
    ``right`` row carries the yaml's stereo ``R`` and ``T`` relative to
    ``left``. The full source file is kept in ``params_json`` (the yaml also has
    rectification ``R1``/``R2``/``P1``/``P2``/``Q``).

    Returns:
        Episode identity, ``source_path``, ``camera`` (``left_wrist``,
        ``right_wrist``, ``head``), ``stream``, ``serial_number``, ``width``,
        ``height``, ``fx``, ``fy``, ``cx``, ``cy``, ``distortion_model``,
        ``distortion_coeffs``, ``extrinsics_reference``, ``rotation`` (9
        values), ``translation`` (3 values), and ``params_json``.
    """
    _require_columns(episodes, "calibration_paths", *_IDENTITY_COLUMNS)
    io_config = _resolve_io_config(io_config)
    return (
        episodes.select(*_IDENTITY_COLUMNS, col("calibration_paths").alias("source_path"))
        .explode("source_path")
        .where(col("source_path").not_null())
        .with_column("__params", download(col("source_path"), io_config=io_config))
        .select(
            *_IDENTITY_COLUMNS,
            "source_path",
            cast("Any", _parse_calibration)(col("source_path"), col("__params")),
        )
        .where(col("stream").not_null())
    )


# --------------------------------------------------------------------------- #
# MCAP summaries
# --------------------------------------------------------------------------- #


@daft.func(return_dtype=_METADATA_DTYPE, use_process=False, unnest=True)
def _read_hiw_metadata(handle: daft.File) -> dict[str, object]:
    summary, records = _mcap.read_summary(handle, dataset="HIW-500", extra="hiw500")
    ros_distro = next((dict(record.metadata).get("ROS_DISTRO") for record in records if record.name == "rosbag2"), None)
    start, end = summary["message_start_time"], summary["message_end_time"]
    counts = summary["channel_message_counts"]
    channels = sorted(dict.fromkeys(summary["channels"]))
    return {
        "ros_distro": ros_distro,
        "message_count": summary["message_count"],
        "message_start_time": start,
        "message_end_time": end,
        "chunk_count": summary["chunk_count"],
        "indexed": summary["indexed"],
        "topics": [topic for topic, _ in channels],
        "channels": [
            {"topic": topic, "schema_name": schema, "message_count": counts.get(topic)} for topic, schema in channels
        ],
    }


def metadata(episodes: DataFrame) -> DataFrame:
    """Range-read MCAP summaries for a bounded HIW-500 catalog.

    Each episode costs a few small header/footer/summary range reads; message
    payloads are not downloaded. An MCAP without a summary section falls back
    to a full sequential read and reports ``indexed=False``.

    Returns:
        The input columns plus ``ros_distro``, ``message_count``,
        ``message_start_time``, ``message_end_time``, ``chunk_count``,
        ``indexed``, ``topics``, and ``channels`` (a list of
        ``{topic, schema_name, message_count}``). It composes with
        :func:`info`; :func:`info` carries the episode duration.
    """
    _require_mcap()
    _require_columns(episodes, "episode_mcap")
    return episodes.where(col("episode_mcap").not_null()).select(
        *_passthrough(episodes, _METADATA_DTYPE), cast("Any", _read_hiw_metadata)(col("episode_mcap"))
    )


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


def _with_identity(dataframe: DataFrame) -> DataFrame:
    source = col("source_path")
    return dataframe.with_columns(
        {
            "task": regexp_extract(source, _LAYOUT_PATTERN, 1),
            "session": regexp_extract(source, _LAYOUT_PATTERN, 2),
            "episode_number": regexp_extract(source, _LAYOUT_PATTERN, 3).cast(DataType.int64()),
            "episode_id": regexp_extract(source, _EPISODE_ID_PATTERN, 1),
        }
    )


def messages(
    episodes: DataFrame,
    *,
    topics: str | Sequence[str] | None = DEFAULT_MESSAGE_TOPICS,
    start_time: int | None = None,
    end_time: int | None = None,
    batch_size: int = 1000,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Read timestamped ROS 2 messages (raw CDR bytes) from a bounded HIW-500 catalog.

    The catalog's MCAP paths are collected eagerly (filter and limit it first),
    then each file becomes a :func:`daft.read_mcap` scan with the topic and time
    filters pushed into the reader. Topics run on independent clocks and are
    not aligned or resampled.

    Args:
        episodes: Bounded catalog from :func:`raw`.
        topics: Topic or topics to read; defaults to every non-camera topic
            (:data:`DEFAULT_MESSAGE_TOPICS`). ``None`` reads every topic,
            including the JPEG camera streams.
        start_time: Inclusive ``log_time`` bound (Unix nanoseconds).
        end_time: Exclusive ``log_time`` bound (Unix nanoseconds).
        batch_size: Messages per source batch.
        io_config: Storage configuration; ``hf://`` tokens are resolved as in :func:`raw`.

    Returns:
        One row per message: ``source_path``, ``topic``, ``log_time``,
        ``publish_time``, ``sequence`` (all integers as ``int64``), the raw
        CDR payload ``data`` as binary, and ``task``, ``session``,
        ``episode_number``, ``episode_id`` parsed from the path.
    """
    selected_topics = _mcap.normalize_names(topics, name="topics")
    paths = _mcap.collect_paths(episodes, ("mcap_path",), source="hiw500.raw()")
    if not paths:
        return _with_identity(_mcap.empty_messages())
    _require_mcap()
    io_config = _resolve_io_config(io_config, paths)
    frames = _mcap.read_messages(
        paths,
        topics=selected_topics,
        start_time=start_time,
        end_time=end_time,
        batch_size=batch_size,
        io_config=io_config,
    )
    return _with_identity(frames)


# --------------------------------------------------------------------------- #
# ROS 2 CDR decoding
# --------------------------------------------------------------------------- #

_CDR_SIZES = {"B": 1, "b": 1, "h": 2, "H": 2, "i": 4, "I": 4, "f": 4, "d": 8}


class _CdrReader:
    """Plain (XCDR1) CDR reader for the fixed ROS 2 layouts decoded here.

    Primitives are aligned to their size relative to the end of the 4-byte
    encapsulation header, as ROS 2's Fast CDR serializer writes them.
    """

    def __init__(self, data: bytes) -> None:
        if len(data) < 4 or data[0] != 0 or data[1] not in (0, 1):
            raise ValueError("Not a plain CDR payload")
        self._data = data
        self._endian = "<" if data[1] == 1 else ">"
        self._offset = 0

    def values(self, code: str, count: int = 1) -> tuple[Any, ...]:
        size = _CDR_SIZES[code]
        self._offset += -self._offset % size
        result = struct.unpack_from(f"{self._endian}{count}{code}", self._data, 4 + self._offset)
        self._offset += size * count
        return result

    def value(self, code: str) -> Any:
        return self.values(code)[0]

    def octets(self) -> bytes:
        length = self.value("I")
        start = 4 + self._offset
        if start + length > len(self._data):
            raise ValueError("Truncated CDR sequence")
        self._offset += length
        return self._data[start : start + length]

    def string(self) -> str:
        return self.octets().rstrip(b"\x00").decode("utf-8", "replace")

    def header(self) -> tuple[int, str]:
        """``std_msgs/Header``: stamp in nanoseconds and frame id."""
        seconds = self.value("i")
        nanoseconds = self.value("I")
        return seconds * 1_000_000_000 + nanoseconds, self.string()


def _decode_lowstate(data: bytes) -> dict[str, object]:
    """Decode ``homies/msg/LowStateStamped`` (a header plus ``unitree_hg/LowState``)."""
    reader = _CdrReader(data)
    stamp_ns, _ = reader.header()
    reader.values("I", 2)  # version
    _, mode_machine = reader.values("B", 2)  # mode_pr, mode_machine
    tick = reader.value("I")
    quaternion = reader.values("f", 4)
    gyroscope = reader.values("f", 3)
    accelerometer = reader.values("f", 3)
    rpy = reader.values("f", 3)
    reader.value("h")  # IMU temperature
    q: list[float] = []
    dq: list[float] = []
    tau_est: list[float] = []
    for _ in range(_LOWSTATE_MOTOR_SLOTS):
        reader.value("B")  # mode
        position, velocity, _, torque = reader.values("f", 4)  # q, dq, ddq, tau_est
        reader.values("h", 2)  # temperature
        reader.value("f")  # vol
        reader.values("I", 2)  # sensor
        reader.value("I")  # motorstate
        reader.values("I", 4)  # reserve
        q.append(position)
        dq.append(velocity)
        tau_est.append(torque)
    reader.values("B", 40)  # wireless_remote
    reader.values("I", 4)  # reserve
    reader.value("I")  # crc
    return {
        "stamp_ns": stamp_ns,
        "tick": tick,
        "mode_machine": mode_machine,
        "q": q[:_NUM_JOINTS],
        "dq": dq[:_NUM_JOINTS],
        "tau_est": tau_est[:_NUM_JOINTS],
        "imu_quaternion": list(quaternion),
        "imu_gyroscope": list(gyroscope),
        "imu_accelerometer": list(accelerometer),
        "imu_rpy": list(rpy),
    }


@daft.func(return_dtype=_JOINT_DTYPE, use_process=False, unnest=True)
def _parse_lowstate(data: bytes) -> dict[str, object] | None:
    try:
        return _decode_lowstate(data)
    except (struct.error, ValueError):
        return None


def _decode_string(data: bytes) -> str:
    """Decode ``std_msgs/msg/String``."""
    return _CdrReader(data).string()


def _fixed(values: Any, size: int) -> list[float] | None:
    if not isinstance(values, list) or len(values) != size:
        return None
    return [float(value) for value in values]


@daft.func(return_dtype=_WBC_DTYPE, use_process=False, unnest=True)
def _parse_wbc(data: bytes) -> dict[str, object] | None:
    try:
        payload = json.loads(_decode_string(data))
    except (struct.error, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    grippers = payload.get("gripper_controls")
    grippers = grippers if isinstance(grippers, dict) else {}

    def gripper(name: str) -> float | None:
        value = grippers.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "pivot": _fixed(payload.get("pivot"), 7),
        "ee_state": _fixed(payload.get("ee_state"), 12),
        "ee_action": _fixed(payload.get("ee_action"), 12),
        "left_trigger": gripper("left_trigger"),
        "left_squeeze": gripper("left_squeeze"),
        "right_trigger": gripper("right_trigger"),
        "right_squeeze": gripper("right_squeeze"),
    }


@daft.func(return_dtype=DataType.string(), use_process=False)
def _parse_string(data: bytes) -> str | None:
    try:
        return _decode_string(data)
    except (struct.error, ValueError):
        return None


@daft.func(return_dtype=_IMAGE_DTYPE, use_process=False, unnest=True)
def _parse_compressed_image(data: bytes) -> dict[str, object] | None:
    """Decode ``sensor_msgs/msg/CompressedImage`` down to its encoded image bytes."""
    try:
        reader = _CdrReader(data)
        stamp_ns, frame_id = reader.header()
        fmt = reader.string()
        encoded = reader.octets()
    except (struct.error, ValueError):
        return None
    return {"stamp_ns": stamp_ns, "frame_id": frame_id, "format": fmt, "encoded": encoded}


_MESSAGE_COLUMNS: tuple[str, ...] = (*_IDENTITY_COLUMNS, "topic", "log_time", "publish_time", "sequence")


def joint_states(
    episodes: DataFrame,
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    batch_size: int = 1000,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Decode ``/stamped/lowstate`` into typed 29-DoF joint states and the pelvis IMU.

    The topic is the G1 low-level state at about 100 Hz. Joints follow
    :data:`JOINT_NAMES` (the first 29 of the message's 35 motor slots).
    Malformed payloads yield nulls.

    Returns:
        Episode identity, ``topic``, ``log_time``, ``publish_time``,
        ``sequence``, ``stamp_ns`` (message header), ``tick``,
        ``mode_machine``, ``q`` / ``dq`` / ``tau_est`` (29 x float32: rad,
        rad/s, N*m), and ``imu_quaternion`` (w, x, y, z), ``imu_gyroscope``,
        ``imu_accelerometer``, ``imu_rpy``.
    """
    rows = messages(
        episodes,
        topics=LOWSTATE_TOPIC,
        start_time=start_time,
        end_time=end_time,
        batch_size=batch_size,
        io_config=io_config,
    )
    return rows.select(*_MESSAGE_COLUMNS, cast("Any", _parse_lowstate)(col("data")))


def wbc_states(
    episodes: DataFrame,
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    batch_size: int = 1000,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Decode ``/wbc_lerobot``: whole-body-controller state and teleop action.

    Each message is a ``std_msgs/String`` holding JSON (about 50 Hz): ``pivot``
    (vx, vy, vyaw, roll, pitch, yaw, height), end-effector pose ``ee_state``
    and target ``ee_action`` (left then right: x, y, z, roll, pitch, yaw), and
    the four gripper controls. The LeRobot release's ``observation.state.wbc``
    is ``pivot + ee_state + grippers`` and its ``action`` is
    ``pivot + ee_action + grippers``.

    Returns:
        Episode identity, ``topic``, ``log_time``, ``publish_time``,
        ``sequence``, ``pivot`` (7), ``ee_state`` (12), ``ee_action`` (12),
        ``left_trigger``, ``left_squeeze``, ``right_trigger``,
        ``right_squeeze``.
    """
    rows = messages(
        episodes, topics=WBC_TOPIC, start_time=start_time, end_time=end_time, batch_size=batch_size, io_config=io_config
    )
    return rows.select(*_MESSAGE_COLUMNS, cast("Any", _parse_wbc)(col("data")))


def annotations(
    episodes: DataFrame,
    *,
    batch_size: int = 1000,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Decode the ``/annotation`` topic (``std_msgs/String``) recorded in the MCAP.

    The topic carries the episode's task text at the episode start, then one
    message per subtask label at that subtask's start time: the same labels and
    timestamps as ``info.json``. :func:`subtasks` reads them from ``info.json``
    with end times added, without opening the MCAP.

    Returns:
        Episode identity, ``topic``, ``log_time``, ``publish_time``,
        ``sequence``, and ``text``.
    """
    rows = messages(episodes, topics=ANNOTATION_TOPIC, batch_size=batch_size, io_config=io_config)
    return rows.select(*_MESSAGE_COLUMNS, cast("Any", _parse_string)(col("data")).alias("text"))


def camera_frames(
    episodes: DataFrame,
    cameras: str | Sequence[str] = ("head", "left_wrist", "right_wrist"),
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    width: int | None = None,
    height: int | None = None,
    mode: str = "RGB",
    batch_size: int = 100,
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Decode HIW-500 camera topics into one row per image.

    Every camera topic is ``sensor_msgs/CompressedImage`` with one JPEG per
    message, so frames decode independently (no keyframe warm-up) with Daft's
    native image decoder. The head image is the stereo pair side by side
    (1280x480); wrist RGB and IR streams are 640x480 (IR is single-channel).
    Bound the catalog and the time window first.

    Args:
        episodes: Bounded catalog from :func:`raw`.
        cameras: Camera alias or aliases from :data:`CAMERA_TOPICS`.
        start_time: Inclusive ``log_time`` bound (Unix nanoseconds).
        end_time: Exclusive ``log_time`` bound (Unix nanoseconds).
        width: Optional output width; must be given with ``height``.
        height: Optional output height; must be given with ``width``.
        mode: Image mode for decoding, e.g. ``"RGB"`` or ``"L"``.
        batch_size: Messages per source batch.
        io_config: Storage configuration; ``hf://`` tokens are resolved as in :func:`raw`.

    Returns:
        Episode identity, ``camera``, ``topic``, ``log_time``,
        ``publish_time``, ``sequence``, ``stamp_ns`` (capture time from the
        message header), ``frame_id``, ``format``, ``width``, ``height``, and
        the decoded image ``data``.
    """
    if (width is None) != (height is None):
        raise ValueError("width and height must be given together")
    selected = _mcap.normalize_names(cameras, name="cameras")
    assert selected is not None
    unknown = [camera for camera in selected if camera not in CAMERA_TOPICS]
    if unknown:
        raise ValueError(f"Unknown camera(s): {unknown}. Expected one or more of: {', '.join(CAMERA_TOPICS)}.")
    topics = [CAMERA_TOPICS[camera] for camera in selected]

    camera = when(col("topic") == topics[0], lit(selected[0]))
    for alias, topic in zip(selected[1:], topics[1:]):
        camera = camera.when(col("topic") == topic, lit(alias))

    rows = messages(
        episodes, topics=topics, start_time=start_time, end_time=end_time, batch_size=batch_size, io_config=io_config
    )
    image = decode_image(col("encoded"), mode=mode)
    if width is not None and height is not None:
        image = resize(image, width, height)
    return (
        rows.select(*_MESSAGE_COLUMNS, cast("Any", _parse_compressed_image)(col("data")))
        .where(col("encoded").not_null())
        .with_columns({"camera": camera.otherwise(lit(None)), "data": image})
        .select(
            *_IDENTITY_COLUMNS,
            "camera",
            "topic",
            "log_time",
            "publish_time",
            "sequence",
            "stamp_ns",
            "frame_id",
            "format",
            image_width(col("data")).cast(DataType.int64()).alias("width"),
            image_height(col("data")).cast(DataType.int64()).alias("height"),
            "data",
        )
    )


# --------------------------------------------------------------------------- #
# LeRobot release
# --------------------------------------------------------------------------- #


@daft.func(return_dtype=DataType.string(), use_process=False)
def _active_subtask(entries: list[dict[str, Any]] | None, timestamp: float | None) -> str | None:
    if not entries or timestamp is None:
        return None
    label = None
    best = None
    for entry in entries:
        start = entry.get("timestamp")
        if entry.get("style") not in (None, "subtask") or start is None:
            continue
        # Frame timestamps are float32; allow for rounding at a label boundary.
        if start <= timestamp + 1e-4 and (best is None or start >= best):
            label, best = entry.get("content"), start
    return label


def lerobot_subtask(
    language_persistent: Expression | None = None,
    timestamp: Expression | None = None,
) -> Expression:
    """The subtask label active at each frame of the LeRobot release.

    ``BitRobot/HIW-500-LeRobot`` stores each episode's full subtask list in
    ``language_persistent`` on every frame (``style == "subtask"``, with
    ``timestamp`` in seconds from the episode start; the first entry, at 0, is
    the episode task). This picks the latest entry at or before the frame's
    ``timestamp``. ``language_events`` is empty in the release.

    Example::

        from daft.datasets import lerobot

        frames = lerobot.read(hiw500.LEROBOT_REPO)
        frames = frames.with_column("subtask", hiw500.lerobot_subtask())

    Args:
        language_persistent: The ``language_persistent`` column (default).
        timestamp: The per-frame ``timestamp`` column (default).
    """
    entries = col("language_persistent") if language_persistent is None else language_persistent
    seconds = col("timestamp") if timestamp is None else timestamp
    return cast("Any", _active_subtask)(entries, seconds.cast(DataType.float64()))


__all__ = [
    "ANNOTATION_TOPIC",
    "CAMERA_TOPICS",
    "DEFAULT_MESSAGE_TOPICS",
    "HF_DATASET",
    "JOINT_NAMES",
    "LEROBOT_REPO",
    "LOWSTATE_TOPIC",
    "WBC_TOPIC",
    "annotations",
    "calibration",
    "camera_frames",
    "info",
    "joint_states",
    "lerobot_subtask",
    "messages",
    "metadata",
    "raw",
    "subtasks",
    "wbc_states",
]
