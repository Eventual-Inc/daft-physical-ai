"""Camera inventory, encoded payload, depth, and calibration access."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from daft.datatype import DataType
from daft.expressions import col
from daft.functions import unnest

from ._common import (
    attrs,
    get_node,
    h5path,
    read_episodes,
    require_episode_column,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from daft.dataframe import DataFrame

RGBD_STREAMS: tuple[str, ...] = ("color", "left", "right")
_DECODABLE_CODECS: frozenset[str] = frozenset({"h26x-annexb", "matroska", "mp4"})
_AV_FORMAT_HINT: dict[str, str] = {"h26x-annexb": "hevc", "matroska": "matroska", "mp4": "mp4"}
_CODEC_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"\x1a\x45\xdf\xa3", "matroska"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"RIFF", "riff"),
    (b"\x00\x00\x00\x01", "h26x-annexb"),
    (b"\x00\x00\x01", "h26x-annexb"),
)


def _sniff_codec(head: bytes) -> str:
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"
    for magic, codec in _CODEC_MAGICS:
        if head.startswith(magic):
            return codec
    return "unknown"


def _is_checked_camera(camera: str, checked: str) -> bool:
    """Match ``checked_cam_name`` (stored as e.g. ``"Camera4"``) to one camera by exact id.

    A substring test would also flag ``RGB_Camera10``-``RGB_Camera12`` for ``"Camera1"``.
    """
    return bool(checked) and checked in (camera, camera.removeprefix("RGB_"))


def _stream_name(camera: str, stream: str) -> str:
    return f"{camera}/{stream}" if stream else camera


def _payload_path(camera: str, stream: str) -> str:
    return h5path("observation", "image", camera, stream, "data")


def _normalize_camera_selectors(
    selectors: str | Sequence[str | tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    is_single_rgbd_pair = (
        isinstance(selectors, (tuple, list))
        and len(selectors) == 2
        and isinstance(selectors[0], str)
        and selectors[0].upper().startswith("RGBD")
        and isinstance(selectors[1], str)
        and selectors[1] in RGBD_STREAMS
    )
    if isinstance(selectors, str):
        entries: Sequence[str | tuple[str, str]] = (selectors,)
    elif is_single_rgbd_pair:
        entries = ((str(selectors[0]), str(selectors[1])),)
    else:
        entries = tuple(selectors)
    if not entries:
        raise ValueError("cameras must contain at least one camera stream")
    targets: list[tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):
            if entry.upper().startswith("RGBD"):
                raise ValueError(f"RGBD selector {entry!r} must include a stream: (camera, stream)")
            camera, stream = entry, ""
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            camera, stream = str(entry[0]), str(entry[1])
            if not camera.upper().startswith("RGBD"):
                raise ValueError(f"Stream selectors are only valid for RGBD cameras, got {entry!r}")
            if stream not in RGBD_STREAMS:
                raise ValueError(f"Unknown RGBD stream {stream!r}. Expected one of: {', '.join(RGBD_STREAMS)}")
        else:
            raise ValueError(f"Invalid camera selector {entry!r}")
        targets.append((camera, stream))
    return tuple(dict.fromkeys(targets))


_CAMERA_DTYPE = DataType.struct(
    {
        "camera": DataType.string(),
        "kind": DataType.string(),
        "stream": DataType.string(),
        "h5path": DataType.string(),
        "codec": DataType.string(),
        "payload_bytes": DataType.int64(),
        "clock_h5path": DataType.string(),
        "clock_length": DataType.int64(),
        "width": DataType.int64(),
        "height": DataType.int64(),
        "intrinsics": DataType.tensor(DataType.float32()),
        "extrinsics": DataType.tensor(DataType.float64()),
        "distortion": DataType.string(),
        "relative_to": DataType.string(),
        "calibration_date": DataType.string(),
        "is_checked_camera": DataType.bool(),
    }
)
_VIDEO_FRAME_DTYPE = DataType.struct(
    {
        "frame_index": DataType.int64(),
        "frame_time": DataType.float64(),
        "frame_time_base": DataType.string(),
        "frame_pts": DataType.int64(),
        "frame_dts": DataType.int64(),
        "frame_duration": DataType.int64(),
        "is_key_frame": DataType.bool(),
        "data": DataType.image(),
    }
)


def cameras(episodes: DataFrame) -> DataFrame:
    """Return a long-form, fixed-schema camera stream inventory."""
    require_episode_column(episodes)

    def read_cameras(h5: Any) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        image_path = h5path("observation", "image")
        image = get_node(h5, image_path)
        if image is None:
            return {"camera_stream": rows}
        checked = str(attrs(image).get("checked_cam_name") or "")
        for camera in sorted(str(name) for name in image):
            camera_group = image[camera]
            is_rgbd = camera.upper().startswith("RGBD")
            streams = RGBD_STREAMS if is_rgbd else ("",)
            for stream in streams:
                holder = get_node(camera_group, stream) if stream else camera_group
                if holder is None:
                    continue
                payload = get_node(holder, "data")
                if payload is None:
                    continue
                intrinsics = get_node(holder, "intrinsics")
                intrinsics_attrs = attrs(intrinsics) if intrinsics is not None else {}
                extrinsics = (
                    get_node(holder, "extrinsics")
                    or get_node(camera_group, "extrinsics")
                    or (get_node(camera_group, "color/extrinsics") if is_rgbd else None)
                )
                extrinsics_attrs = attrs(extrinsics) if extrinsics is not None else {}
                stream_clock_name = f"{stream}_timestamp" if stream else "timestamp"
                stream_clock = get_node(camera_group, stream_clock_name)
                clock_name = stream_clock_name
                if stream_clock is None:
                    stream_clock = get_node(camera_group, "timestamp")
                    clock_name = "timestamp"
                head = bytes(payload[: min(16, int(payload.shape[0]))])
                holder_path = h5path("observation", "image", camera, stream)
                rows.append(
                    {
                        "camera": camera,
                        "kind": "rgbd" if is_rgbd else "rgb",
                        "stream": stream,
                        "h5path": holder_path,
                        "codec": _sniff_codec(head),
                        "payload_bytes": int(payload.shape[0]),
                        "clock_h5path": (
                            h5path("observation", "image", camera, clock_name) if stream_clock is not None else None
                        ),
                        "clock_length": (
                            int(stream_clock.shape[0]) if stream_clock is not None and stream_clock.shape else None
                        ),
                        "width": (
                            int(intrinsics_attrs["width"]) if intrinsics_attrs.get("width") is not None else None
                        ),
                        "height": (
                            int(intrinsics_attrs["height"]) if intrinsics_attrs.get("height") is not None else None
                        ),
                        "intrinsics": intrinsics[()] if intrinsics is not None else None,
                        "extrinsics": extrinsics[()] if extrinsics is not None else None,
                        "distortion": intrinsics_attrs.get("distortion"),
                        "relative_to": extrinsics_attrs.get("relative_to_who"),
                        "calibration_date": extrinsics_attrs.get("calib_date"),
                        "is_checked_camera": _is_checked_camera(camera, checked),
                    }
                )
        return {"camera_stream": rows}

    fields = {"camera_stream": DataType.list(_CAMERA_DTYPE)}
    rows = read_episodes(episodes, fields, read_cameras, columns=["episode_key"]).explode("camera_stream")
    return rows.select("episode_key", unnest(col("camera_stream")))


def camera_payloads(
    episodes: DataFrame,
    cameras: str | Sequence[str | tuple[str, str]],
) -> DataFrame:
    """Append selected encoded payloads, reading all of them in one HDF5 session."""
    require_episode_column(episodes)
    targets = _normalize_camera_selectors(cameras)
    return_fields: dict[str, DataType] = {}
    for camera, stream in targets:
        name = _stream_name(camera, stream)
        return_fields[f"{name}/payload"] = DataType.binary()
        return_fields[f"{name}/codec"] = DataType.string()

    def read_payloads(h5: Any) -> dict[str, Any]:
        result: dict[str, Any] = dict.fromkeys(return_fields)
        for camera, stream in targets:
            node = get_node(h5, _payload_path(camera, stream))
            if node is None:
                continue
            name = _stream_name(camera, stream)
            payload = node[()].tobytes()
            result[f"{name}/payload"] = payload
            result[f"{name}/codec"] = _sniff_codec(payload[:16])
        return result

    return read_episodes(episodes, return_fields, read_payloads)


def _decode_payload(
    payload: bytes,
    codec: str,
    clock_us: Any | None,
    *,
    start_time: float,
    end_time: float | None,
    width: int | None,
    height: int | None,
    is_key_frame: bool | None,
    sample_interval_seconds: float | None,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    """Decode one embedded stream into Daft's public video-frame struct."""
    import io

    from daft.dependencies import av, np

    rows: list[dict[str, Any]] = []
    next_sample_time = start_time
    try:
        with av.open(io.BytesIO(payload), "r", format=_AV_FORMAT_HINT[codec]) as container:
            if not container.streams.video:
                raise ValueError("encoded payload has no video stream")
            video = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(video)):
                frame_time = float(frame.time) if frame.time is not None else None
                frame_time_base = str(frame.time_base) if frame.time_base is not None else None
                if frame_time is None and clock_us is not None and frame_index < len(clock_us):
                    frame_time = (int(clock_us[frame_index]) - int(clock_us[0])) / 1_000_000.0
                    frame_time_base = "1/1000000"
                if frame_time is None:
                    raise ValueError(
                        "decoded frame has no elementary-stream timestamp and the HDF5 camera clock is unavailable"
                    )
                if frame_time < start_time:
                    continue
                if end_time is not None and frame_time >= end_time:
                    break
                key_frame = bool(frame.key_frame)
                if is_key_frame is not None and key_frame is not is_key_frame:
                    continue
                if sample_interval_seconds is not None:
                    if frame_time + 1e-12 < next_sample_time:
                        continue
                    while next_sample_time <= frame_time + 1e-12:
                        next_sample_time += sample_interval_seconds
                if width is not None and height is not None:
                    data = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                else:
                    data = frame.to_ndarray(format="rgb24")
                rows.append(
                    {
                        "frame_index": frame_index,
                        "frame_time": frame_time,
                        "frame_time_base": frame_time_base,
                        "frame_pts": int(frame.pts) if frame.pts is not None else None,
                        "frame_dts": int(frame.dts) if frame.dts is not None else None,
                        "frame_duration": int(frame.duration) if frame.duration is not None else None,
                        "is_key_frame": key_frame,
                        "data": np.asarray(data),
                    }
                )
                if max_frames is not None and len(rows) >= max_frames:
                    break
    except (av.FFmpegError, EOFError) as error:
        raise ValueError(f"failed to decode {codec} camera payload: {error}") from error
    return rows


def camera_frames(
    episodes: DataFrame,
    cameras: str | Sequence[str | tuple[str, str]],
    *,
    start_time: float = 0,
    end_time: float | None = None,
    width: int | None = None,
    height: int | None = None,
    is_key_frame: bool | None = None,
    sample_interval_seconds: float | None = None,
    max_frames: int | None = 1,
) -> DataFrame:
    """Decode selected embedded streams with DROID/video_frames-compatible options.

    Each payload holds a whole camera stream (1920x1200 HEVC for RGB cameras), so
    ``max_frames`` caps the frames decoded per stream after the time and
    key-frame filters. It defaults to 1; pass a larger value, or ``None`` to
    decode every selected frame.
    """
    require_episode_column(episodes)
    from daft.dependencies import av

    if not av.module_available():  # ty: ignore[unresolved-attribute]
        raise ImportError("Decoding OmniSharing streams requires the 'daft[video]' extra")
    targets = _normalize_camera_selectors(cameras)
    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}")
    if end_time is not None and end_time <= start_time:
        raise ValueError(f"end_time must be greater than start_time, got {end_time}")
    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    if width is not None and (width <= 0 or height is None or height <= 0):
        raise ValueError("width and height must be positive")
    if is_key_frame is not None and not isinstance(is_key_frame, bool):
        raise ValueError("is_key_frame must be bool or None")
    if sample_interval_seconds is not None and sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if max_frames is not None and (isinstance(max_frames, bool) or max_frames < 1):
        raise ValueError(f"max_frames must be a positive integer or None, got {max_frames}")

    return_fields = {
        f"{_stream_name(camera, stream)}/frames": DataType.list(_VIDEO_FRAME_DTYPE) for camera, stream in targets
    }

    def decode_streams(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        result: dict[str, Any] = {}
        for camera, stream in targets:
            name = _stream_name(camera, stream)
            output_name = f"{name}/frames"
            node = get_node(h5, _payload_path(camera, stream))
            if node is None:
                result[output_name] = []
                continue
            payload = node[()].tobytes()
            codec = _sniff_codec(payload[:16])
            if codec not in _DECODABLE_CODECS:
                raise ValueError(f"{node.name} has unsupported embedded video codec {codec!r}")
            camera_group = get_node(h5, h5path("observation", "image", camera))
            clock = None
            if camera_group is not None:
                clock_node = get_node(camera_group, f"{stream}_timestamp" if stream else "timestamp")
                if clock_node is None:
                    clock_node = get_node(camera_group, "timestamp")
                if clock_node is not None:
                    clock = np.asarray(clock_node[()])
            result[output_name] = _decode_payload(
                payload,
                codec,
                clock,
                start_time=start_time,
                end_time=end_time,
                width=width,
                height=height,
                is_key_frame=is_key_frame,
                sample_interval_seconds=sample_interval_seconds,
                max_frames=max_frames,
            )
        return result

    return read_episodes(episodes, return_fields, decode_streams)


def _normalize_camera_names(values: str | Sequence[str]) -> tuple[str, ...]:
    names = (values,) if isinstance(values, str) else tuple(values)
    if not names:
        raise ValueError("cameras must contain at least one RGBD camera")
    if any(not isinstance(name, str) or not name.upper().startswith("RGBD") for name in names):
        raise ValueError(f"Depth and stereo calibration require RGBD camera names, got {names}")
    return tuple(dict.fromkeys(names))


def depth_frames(
    episodes: DataFrame,
    cameras: str | Sequence[str],
    *,
    frame_indices: int | Sequence[int] = 0,
    strict: bool = False,
) -> DataFrame:
    """Read only requested aligned-depth indices into stable per-camera columns.

    Episode lengths vary, so by default each episode reads the requested indices
    it has and reports them in ``<camera>/depth_frame_indices``. Pass
    ``strict=True`` to fail on an episode that is too short instead.
    """
    require_episode_column(episodes)
    names = _normalize_camera_names(cameras)
    wanted = (frame_indices,) if isinstance(frame_indices, int) else tuple(frame_indices)
    if not wanted:
        raise ValueError("frame_indices must contain at least one index")
    if any(index < 0 for index in wanted):
        raise IndexError(f"frame_indices must be non-negative, got {wanted}")
    return_fields: dict[str, DataType] = {}
    for camera in names:
        return_fields[f"{camera}/depth"] = DataType.tensor(DataType.uint16())
        return_fields[f"{camera}/depth_frame_indices"] = DataType.list(DataType.int64())

    def read_depth(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        result: dict[str, Any] = {}
        for camera in names:
            data_key = f"{camera}/depth"
            index_key = f"{camera}/depth_frame_indices"
            node = get_node(h5, h5path("observation", "image", camera, "aligned_depth"))
            if node is None:
                result[data_key] = np.empty((0,), dtype="uint16")
                result[index_key] = []
                continue
            if np.dtype(node.dtype) != np.dtype("uint16") or len(node.shape) != 3:
                raise ValueError(f"{node.name} must be a rank-3 uint16 depth dataset, found {node.shape}/{node.dtype}")
            available = int(node.shape[0])
            out_of_range = [index for index in wanted if index >= available]
            if out_of_range and strict:
                raise IndexError(f"{node.name} has {available} frames; requested out-of-range indices {out_of_range}")
            usable = [index for index in wanted if index < available]
            if not usable:
                result[data_key] = np.empty((0,), dtype="uint16")
                result[index_key] = []
                continue
            result[data_key] = np.stack([np.asarray(node[index]) for index in usable])
            result[index_key] = usable
        return result

    return read_episodes(episodes, return_fields, read_depth)


def stereo_extrinsics(episodes: DataFrame, cameras: str | Sequence[str]) -> DataFrame:
    """Append strict per-camera left-eye-to-colour calibration."""
    require_episode_column(episodes)
    names = _normalize_camera_names(cameras)
    return_fields: dict[str, DataType] = {}
    for camera in names:
        return_fields[f"{camera}/left_to_color"] = DataType.tensor(DataType.float64())
        return_fields[f"{camera}/calibration_date"] = DataType.string()

    def read_stereo(h5: Any) -> dict[str, Any]:
        from daft.dependencies import np

        result: dict[str, Any] = {}
        for camera in names:
            matrix_key = f"{camera}/left_to_color"
            date_key = f"{camera}/calibration_date"
            result[matrix_key] = np.empty((0,), dtype="float64")
            result[date_key] = None
            node = get_node(h5, h5path("observation", "image", camera, "inner_extrinsic"))
            if node is None:
                continue
            raw_value = node[()]
            blob = raw_value[0] if getattr(raw_value, "shape", ()) else raw_value
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8", "strict")
            try:
                payload = json.loads(blob)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{node.name} contains malformed calibration JSON") from error
            if not isinstance(payload, dict) or "left_to_color" not in payload:
                raise ValueError(f"{node.name} must contain a left_to_color calibration matrix")
            try:
                matrix = np.asarray(payload["left_to_color"], dtype="float64")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{node.name} left_to_color must be numeric") from error
            if matrix.shape != (4, 4):
                raise ValueError(f"{node.name} left_to_color must have shape (4, 4), found {matrix.shape}")
            result[matrix_key] = matrix
            result[date_key] = str(payload["calib_date"]) if payload.get("calib_date") is not None else None
        return result

    return read_episodes(episodes, return_fields, read_stereo)


__all__ = [
    "RGBD_STREAMS",
    "camera_frames",
    "camera_payloads",
    "cameras",
    "depth_frames",
    "stereo_extrinsics",
]
