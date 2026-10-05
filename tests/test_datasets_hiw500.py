from __future__ import annotations

import io
import json
import os
import struct
from pathlib import Path

import daft
import pytest
from daft import DataType, col

pytest.importorskip("mcap")
yaml = pytest.importorskip("yaml")

from mcap.writer import CompressionType, Writer

from daft_physical_ai.datasets import hiw500

FIRST_SESSION = "2026-02-28_11-28-25"
FIRST_ID = f"Hang-Hanger/episode_{FIRST_SESSION}/episode_0001"
SECOND_ID = "Sweep-Floor/episode_2026-03-01_09-00-00/episode_0002"

START_NS = 1_772_249_452_000_000_000
STEP_NS = 10_000_000  # 100 Hz
FRAME_COUNT = 4


# --------------------------------------------------------------------------- #
# Synthetic ROS 2 CDR + MCAP fixtures
# --------------------------------------------------------------------------- #


class _CdrWriter:
    """Little-endian plain CDR with primitives aligned relative to the encapsulation header."""

    def __init__(self) -> None:
        self.buffer = bytearray(b"\x00\x01\x00\x00")

    def put(self, code: str, *values: object) -> None:
        size = struct.calcsize(code)
        self.buffer += b"\x00" * (-(len(self.buffer) - 4) % size)
        self.buffer += struct.pack(f"<{len(values)}{code}", *values)

    def string(self, value: str) -> None:
        encoded = value.encode() + b"\x00"
        self.put("I", len(encoded))
        self.buffer += encoded

    def octets(self, value: bytes) -> None:
        self.put("I", len(value))
        self.buffer += value

    def header(self, stamp_ns: int, frame_id: str = "") -> None:
        seconds, nanoseconds = divmod(stamp_ns, 1_000_000_000)
        self.put("i", seconds)
        self.put("I", nanoseconds)
        self.string(frame_id)


def _joint_q(step: int) -> list[float]:
    return [0.01 * joint + step for joint in range(35)]


def _lowstate(stamp_ns: int, step: int, frame_id: str = "") -> bytes:
    writer = _CdrWriter()
    writer.header(stamp_ns, frame_id)
    writer.put("I", 0, 0)  # version
    writer.put("B", 0, 5)  # mode_pr, mode_machine
    writer.put("I", 1000 + step)  # tick
    writer.put("f", 1.0, 0.0, 0.0, 0.0)  # quaternion
    writer.put("f", 0.5, 0.25, 0.125)  # gyroscope
    writer.put("f", 0.0, 0.0, 9.75)  # accelerometer
    writer.put("f", 0.0, 0.0, 1.5)  # rpy
    writer.put("h", 40)  # temperature
    for index, position in enumerate(_joint_q(step)):
        writer.put("B", 1)
        writer.put("f", position, -position, 0.0, 2.0 * index)
        writer.put("h", 30, 31)
        writer.put("f", 50.0)
        writer.put("I", 0, 0)
        writer.put("I", 0)
        writer.put("I", 0, 0, 0, 0)
    writer.put("B", *([0] * 40))
    writer.put("I", 0, 0, 0, 0)
    writer.put("I", 0)
    return bytes(writer.buffer)


def _string_message(text: str) -> bytes:
    writer = _CdrWriter()
    writer.string(text)
    return bytes(writer.buffer)


def _wbc(step: int) -> bytes:
    payload = {
        "pivot": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.74],
        "ee_state": [float(step)] * 12,
        "ee_action": [float(step) + 0.5] * 12,
        "gripper_controls": {"left_trigger": 10.0, "left_squeeze": 0.0, "right_trigger": 5.0, "right_squeeze": 1.0},
    }
    return _string_message(json.dumps(payload))


def _jpeg(level: int, size: tuple[int, int], mode: str) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    Image.new(mode, size, level if mode == "L" else (level, level, level)).save(output, format="JPEG", quality=95)
    return output.getvalue()


def _compressed_image(stamp_ns: int, encoded: bytes) -> bytes:
    writer = _CdrWriter()
    writer.header(stamp_ns)
    writer.string("jpeg")
    writer.octets(encoded)
    return bytes(writer.buffer)


def _write_mcap(path: Path, *, start_ns: int, task_text: str, malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        writer = Writer(output, chunk_size=4096, compression=CompressionType.NONE)
        writer.start(profile="ros2", library="test")

        def channel(topic: str, schema_name: str) -> int:
            schema = writer.register_schema(schema_name, "ros2msg", b"# synthetic")
            return writer.register_channel(topic, "cdr", schema)

        lowstate = channel(hiw500.LOWSTATE_TOPIC, "homies/msg/LowStateStamped")
        wbc = channel(hiw500.WBC_TOPIC, "std_msgs/msg/String")
        annotation = channel(hiw500.ANNOTATION_TOPIC, "std_msgs/msg/String")
        head = channel(hiw500.CAMERA_TOPICS["head"], "sensor_msgs/msg/CompressedImage")
        ir = channel(hiw500.CAMERA_TOPICS["left_wrist_ir1"], "sensor_msgs/msg/CompressedImage")

        def add(channel_id: int, log_time: int, data: bytes) -> None:
            writer.add_message(channel_id, log_time=log_time, publish_time=log_time, sequence=0, data=data)

        add(annotation, start_ns, _string_message(task_text))
        for step in range(FRAME_COUNT):
            stamp = start_ns + step * STEP_NS
            # A non-empty frame id shifts every later field's alignment.
            add(lowstate, stamp + 1, _lowstate(stamp, step, frame_id="x" * step))
            add(wbc, stamp + 2, _wbc(step))
            add(head, stamp + 3, _compressed_image(stamp, _jpeg(40 * step, (32, 16), "RGB")))
            add(ir, stamp + 4, _compressed_image(stamp, _jpeg(40 * step, (16, 16), "L")))
        if malformed:
            add(lowstate, start_ns + FRAME_COUNT * STEP_NS + 1, b"\x00\x01\x00\x00\x01")
        writer.add_metadata("rosbag2", {"ROS_DISTRO": "humble"})
        writer.finish()


REALSENSE = {
    "color": {
        "intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 433.0,
            "fy": 432.0,
            "ppx": 317.0,
            "ppy": 238.0,
            "model": "distortion.inverse_brown_conrady",
            "coeffs": [0.1, 0.2, 0.0, 0.0, 0.3],
        }
    },
    "ir1": {
        "intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 389.0,
            "fy": 389.0,
            "ppx": 317.5,
            "ppy": 237.5,
            "model": "distortion.brown_conrady",
            "coeffs": [0.0] * 5,
        },
        "extrinsics_to_color": {"rotation": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], "translation": [0.0, 0.0, 0.0]},
    },
    "serial_number": "409122272374",
    "position": "right",
}

HEAD = {
    "camera_matrix_left": [[323.5, 0.0, 300.5], [0.0, 322.5, 246.5], [0.0, 0.0, 1.0]],
    "camera_matrix_right": [[324.5, 0.0, 309.0], [0.0, 323.5, 239.0], [0.0, 0.0, 1.0]],
    "dist_coeffs_left": [0.03, -0.06, 0.0, 0.0, 0.01],
    "dist_coeffs_right": [0.02, -0.05, 0.0, 0.0, 0.01],
    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "T": [-60.0, 0.5, 1.0],
    "baseline": 60.0,
    "image_size": [640, 480],
}


def _info(start_ns: int, task: str, subtasks: list[tuple[str, int]], duration_ns: int) -> dict[str, object]:
    return {
        "episode_name": f"episode_{FIRST_SESSION}",
        "task": task,
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": start_ns + duration_ns,
        "duration_ns": duration_ns,
        "duration_sec": duration_ns / 1e9,
        "subtasks": [{"task": label, "timestamp_ns": start_ns + offset} for label, offset in subtasks],
        "scene": 3,
    }


@pytest.fixture
def hiw_root(tmp_path: Path) -> Path:
    first = tmp_path / FIRST_ID
    _write_mcap(first / "episode_0001.mcap", start_ns=START_NS, task_text="hang hanger")
    (first / "info.json").write_text(
        json.dumps(_info(START_NS, "hang hanger", [("pick hanger", 0), ("hang hanger", 15_000_000)], 40_000_000))
    )
    params = first / "calibration" / "params"
    params.mkdir(parents=True)
    (params / "camera_409122272374.json").write_text(json.dumps(REALSENSE))
    (params / "head_camera_params.yaml").write_text(yaml.safe_dump(HEAD))

    second = tmp_path / SECOND_ID
    _write_mcap(second / "episode_0002.mcap", start_ns=START_NS + 10**9, task_text="sweep floor", malformed=True)
    (second / "info.json").write_text(json.dumps(_info(START_NS + 10**9, "sweep floor", [], 40_000_000)))
    # Not part of any episode: ignored by the catalog.
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "readme.png").write_bytes(b"png")
    return tmp_path


def _first(dataframe: daft.DataFrame) -> daft.DataFrame:
    return dataframe.where(col("episode_id") == FIRST_ID)


# --------------------------------------------------------------------------- #
# raw()
# --------------------------------------------------------------------------- #


def test_raw_builds_episode_catalog(hiw_root: Path) -> None:
    episodes = hiw500.raw(str(hiw_root))
    assert episodes.column_names == list(hiw500._CATALOG_COLUMNS)
    assert episodes.schema()["episode_mcap"].dtype == DataType.file()

    result = episodes.exclude("episode_mcap").sort("episode_id").to_pydict()
    assert result["episode_id"] == [FIRST_ID, SECOND_ID]
    assert result["task"] == ["Hang-Hanger", "Sweep-Floor"]
    assert result["session"] == [FIRST_SESSION, "2026-03-01_09-00-00"]
    assert result["session_time"][0].isoformat() == "2026-02-28T11:28:25"
    assert result["episode_number"] == [1, 2]
    assert result["mcap_size"][0] == (hiw_root / FIRST_ID / "episode_0001.mcap").stat().st_size
    assert result["mcap_path"][0].endswith(f"{FIRST_ID}/episode_0001.mcap")
    assert result["info_path"][1].endswith(f"{SECOND_ID}/info.json")
    assert result["episode_dir"][0].endswith(FIRST_ID)
    first_calibration = [path.rsplit("/", 1)[-1] for path in result["calibration_paths"][0]]
    assert first_calibration == ["camera_409122272374.json", "head_camera_params.yaml"]
    assert result["calibration_paths"][1] == []


def test_raw_filters_tasks(hiw_root: Path) -> None:
    result = hiw500.raw(str(hiw_root), tasks="Sweep-Floor").select("episode_id").to_pydict()
    assert result == {"episode_id": [SECOND_ID]}


def test_raw_rejects_bad_tasks(hiw_root: Path) -> None:
    with pytest.raises(ValueError, match="exact task folder names"):
        hiw500.raw(str(hiw_root), tasks="Hang-*")
    with pytest.raises(ValueError, match="at least one"):
        hiw500.raw(str(hiw_root), tasks=[])


class _FakeRepoFile:
    def __init__(self, path: str, size: int) -> None:
        self.path = path
        self.size = size


class _FakeRepoFolder:
    def __init__(self, path: str) -> None:
        self.path = path


def test_raw_hf_listing_walks_each_task_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_api = pytest.importorskip("huggingface_hub.hf_api")
    from huggingface_hub.utils import EntryNotFoundError

    monkeypatch.setattr(hf_api, "RepoFile", _FakeRepoFile)
    monkeypatch.setattr(hf_api, "RepoFolder", _FakeRepoFolder)
    episode = "Hang-Hanger/episode_2026-02-28_11-28-25/episode_0001"
    tree = {
        None: [_FakeRepoFolder("Hang-Hanger"), _FakeRepoFolder("assets"), _FakeRepoFile("README.md", 1)],
        "Hang-Hanger": [
            _FakeRepoFolder(episode),
            _FakeRepoFile(f"{episode}/episode_0001.mcap", 100),
            _FakeRepoFile(f"{episode}/info.json", 2),
            _FakeRepoFile(f"{episode}/calibration/params/camera_1.json", 3),
            _FakeRepoFile(f"{episode}/calibration/params/head_camera-params.yaml", 4),
        ],
    }
    calls: list[tuple[str | None, bool, str | None]] = []

    class FakeApi:
        def __init__(self, token=None, library_name=None) -> None:
            self.token = token

        def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, revision=None, repo_type=None):
            assert repo_id == "BitRobot/HIW-500" and repo_type == "dataset"
            calls.append((path_in_repo, recursive, revision))
            if path_in_repo not in tree:
                raise EntryNotFoundError("missing")
            return iter(tree[path_in_repo])

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    episodes = hiw500.raw("hf://datasets/BitRobot/HIW-500@abc123")
    # Only the task folders are listed eagerly.
    assert calls == [(None, False, "abc123")]
    result = episodes.exclude("episode_mcap").to_pydict()
    assert calls == [(None, False, "abc123"), ("Hang-Hanger", True, "abc123")]
    assert result["episode_id"] == [episode]
    assert result["mcap_size"] == [100]
    assert result["mcap_path"] == [f"hf://datasets/BitRobot/HIW-500@abc123/{episode}/episode_0001.mcap"]
    assert [path.rsplit("/", 1)[-1] for path in result["calibration_paths"][0]] == [
        "camera_1.json",
        "head_camera-params.yaml",
    ]

    calls.clear()
    empty = hiw500.raw("hf://datasets/BitRobot/HIW-500", tasks="Missing-Task").to_pydict()
    assert calls == [("Missing-Task", True, None)]
    assert all(values == [] for values in empty.values())


# --------------------------------------------------------------------------- #
# info() / subtasks() / calibration() / metadata()
# --------------------------------------------------------------------------- #


def test_info_parses_info_json(hiw_root: Path) -> None:
    result = hiw500.info(hiw500.raw(str(hiw_root))).sort("episode_id").to_pydict()
    assert result["task_name"] == ["hang hanger", "sweep floor"]
    assert result["scene"] == [3, 3]
    assert result["start_timestamp_ns"] == [START_NS, START_NS + 10**9]
    assert result["duration_seconds"] == [0.04, 0.04]
    assert result["subtask_count"] == [2, 0]
    assert json.loads(result["info_json"][0])["subtasks"][1]["task"] == "hang hanger"
    assert result["episode_id"] == [FIRST_ID, SECOND_ID]


def test_subtasks_get_end_times(hiw_root: Path) -> None:
    result = hiw500.subtasks(hiw500.raw(str(hiw_root))).sort("subtask_index").to_pydict()
    assert result["episode_id"] == [FIRST_ID, FIRST_ID]
    assert result["label"] == ["pick hanger", "hang hanger"]
    assert result["start_timestamp_ns"] == [START_NS, START_NS + 15_000_000]
    assert result["end_timestamp_ns"] == [START_NS + 15_000_000, START_NS + 40_000_000]
    assert result["start_offset_seconds"] == [0.0, 0.015]
    assert result["end_offset_seconds"] == [0.015, 0.04]


def test_calibration_long_format(hiw_root: Path) -> None:
    result = hiw500.calibration(hiw500.raw(str(hiw_root))).sort(["camera", "stream"]).to_pydict()
    assert list(zip(result["camera"], result["stream"])) == [
        ("head", "left"),
        ("head", "right"),
        ("right_wrist", "color"),
        ("right_wrist", "ir1"),
    ]
    assert result["episode_id"] == [FIRST_ID] * 4
    assert result["fx"] == [323.5, 324.5, 433.0, 389.0]
    assert result["cx"] == [300.5, 309.0, 317.0, 317.5]
    assert result["width"] == [640] * 4
    assert result["serial_number"] == [None, None, "409122272374", "409122272374"]
    assert result["distortion_model"] == ["plumb_bob", "plumb_bob", "distortion.inverse_brown_conrady"] + [
        "distortion.brown_conrady"
    ]
    assert result["extrinsics_reference"] == [None, "left", None, "color"]
    assert result["rotation"][1] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert result["translation"][1] == [-60.0, 0.5, 1.0]
    assert result["rotation"][2] is None
    assert json.loads(result["params_json"][0])["baseline"] == 60.0


def test_metadata_reads_summary(hiw_root: Path) -> None:
    result = hiw500.metadata(_first(hiw500.raw(str(hiw_root)))).to_pydict()
    assert result["ros_distro"] == ["humble"]
    assert result["message_count"] == [1 + 4 * FRAME_COUNT]
    assert result["message_start_time"] == [START_NS]
    assert result["indexed"] == [True]
    channels = {entry["topic"]: entry for entry in result["channels"][0]}
    assert channels[hiw500.LOWSTATE_TOPIC] == {
        "topic": hiw500.LOWSTATE_TOPIC,
        "schema_name": "homies/msg/LowStateStamped",
        "message_count": FRAME_COUNT,
    }
    assert result["topics"][0] == sorted(channels)


def test_info_and_metadata_compose(hiw_root: Path) -> None:
    episodes = _first(hiw500.raw(str(hiw_root)))
    chained = hiw500.metadata(hiw500.info(episodes))
    names = chained.column_names
    assert len(names) == len(set(names))
    result = chained.select("duration_seconds", "message_count").to_pydict()
    assert result == {"duration_seconds": [0.04], "message_count": [1 + 4 * FRAME_COUNT]}
    # Re-applying replaces rather than duplicates.
    assert hiw500.info(hiw500.info(episodes)).column_names == hiw500.info(episodes).column_names


def test_episode_functions_require_catalog() -> None:
    with pytest.raises(ValueError, match="info_path"):
        hiw500.info(daft.from_pydict({"x": [1]}))
    with pytest.raises(ValueError, match="mcap_path"):
        hiw500.messages(daft.from_pydict({"x": [1]}))


# --------------------------------------------------------------------------- #
# messages() and decoders
# --------------------------------------------------------------------------- #


def test_messages_default_topics_skip_cameras(hiw_root: Path) -> None:
    episodes = _first(hiw500.raw(str(hiw_root)))
    rows = hiw500.messages(episodes).to_pydict()
    assert set(rows["topic"]) == {hiw500.LOWSTATE_TOPIC, hiw500.WBC_TOPIC, hiw500.ANNOTATION_TOPIC}
    assert set(rows["episode_id"]) == {FIRST_ID}
    assert set(rows["task"]) == {"Hang-Hanger"}
    assert set(rows["episode_number"]) == {1}
    assert all(isinstance(payload, bytes) and payload[:2] == b"\x00\x01" for payload in rows["data"])


def test_messages_time_window(hiw_root: Path) -> None:
    episodes = _first(hiw500.raw(str(hiw_root)))
    rows = hiw500.messages(
        episodes, topics=hiw500.LOWSTATE_TOPIC, start_time=START_NS + STEP_NS, end_time=START_NS + 3 * STEP_NS
    ).to_pydict()
    assert sorted(rows["log_time"]) == [START_NS + STEP_NS + 1, START_NS + 2 * STEP_NS + 1]


def test_messages_handles_empty_catalog(hiw_root: Path) -> None:
    episodes = hiw500.raw(str(hiw_root)).where(col("episode_id") == "missing")
    result = hiw500.messages(episodes).to_pydict()
    assert all(values == [] for values in result.values())
    assert {"source_path", "topic", "data", "episode_id"} <= set(result)


def test_joint_states_decode_lowstate(hiw_root: Path) -> None:
    states = hiw500.joint_states(_first(hiw500.raw(str(hiw_root))))
    assert states.schema()["q"].dtype == DataType.fixed_size_list(DataType.float32(), 29)

    result = states.sort("log_time").to_pydict()
    assert result["stamp_ns"] == [START_NS + step * STEP_NS for step in range(FRAME_COUNT)]
    assert result["tick"] == [1000 + step for step in range(FRAME_COUNT)]
    assert result["mode_machine"] == [5] * FRAME_COUNT
    for step in range(FRAME_COUNT):
        assert result["q"][step] == pytest.approx(_joint_q(step)[:29], abs=1e-5)
        assert result["dq"][step] == pytest.approx([-value for value in _joint_q(step)[:29]], abs=1e-5)
    assert result["tau_est"][0] == pytest.approx([2.0 * joint for joint in range(29)])
    assert result["imu_quaternion"][0] == [1.0, 0.0, 0.0, 0.0]
    assert result["imu_gyroscope"][0] == [0.5, 0.25, 0.125]
    assert result["imu_rpy"][0] == [0.0, 0.0, 1.5]


def test_joint_states_null_for_malformed_payload(hiw_root: Path) -> None:
    episodes = hiw500.raw(str(hiw_root)).where(col("episode_id") == SECOND_ID)
    result = hiw500.joint_states(episodes).sort("log_time").to_pydict()
    assert len(result["log_time"]) == FRAME_COUNT + 1
    assert result["q"][-1] is None and result["stamp_ns"][-1] is None
    assert result["q"][0] is not None


def test_wbc_states_decode_json(hiw_root: Path) -> None:
    result = hiw500.wbc_states(_first(hiw500.raw(str(hiw_root)))).sort("log_time").to_pydict()
    assert result["pivot"][0] == [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.74]
    assert result["ee_state"][2] == [2.0] * 12
    assert result["ee_action"][2] == [2.5] * 12
    assert result["left_trigger"] == [10.0] * FRAME_COUNT
    assert result["right_squeeze"] == [1.0] * FRAME_COUNT


def test_annotations_decode_task_text(hiw_root: Path) -> None:
    result = hiw500.annotations(hiw500.raw(str(hiw_root))).sort("episode_id").to_pydict()
    assert result["text"] == ["hang hanger", "sweep floor"]
    assert result["log_time"] == [START_NS, START_NS + 10**9]


def _gray_levels(images: list) -> list[int]:
    return [round(float(image.mean()) / 40) for image in images]


def test_camera_frames_decode_jpeg(hiw_root: Path) -> None:
    frames = hiw500.camera_frames(_first(hiw500.raw(str(hiw_root))), cameras=["head", "left_wrist_ir1"])
    assert frames.schema()["data"].dtype == DataType.image("RGB")

    result = frames.sort(["camera", "log_time"]).to_pydict()
    assert result["camera"] == ["head"] * FRAME_COUNT + ["left_wrist_ir1"] * FRAME_COUNT
    assert result["topic"][0] == hiw500.CAMERA_TOPICS["head"]
    assert result["format"] == ["jpeg"] * (2 * FRAME_COUNT)
    assert result["stamp_ns"][:FRAME_COUNT] == [START_NS + step * STEP_NS for step in range(FRAME_COUNT)]
    assert result["width"] == [32] * FRAME_COUNT + [16] * FRAME_COUNT
    assert result["height"] == [16] * (2 * FRAME_COUNT)
    assert result["data"][FRAME_COUNT].shape == (16, 16, 3)
    assert _gray_levels(result["data"]) == list(range(FRAME_COUNT)) * 2


def test_camera_frames_window_resize_and_mode(hiw_root: Path) -> None:
    result = (
        hiw500.camera_frames(
            _first(hiw500.raw(str(hiw_root))),
            cameras="left_wrist_ir1",
            start_time=START_NS + 2 * STEP_NS,
            width=8,
            height=4,
            mode="L",
        )
        .sort("log_time")
        .to_pydict()
    )
    assert result["width"] == [8, 8] and result["height"] == [4, 4]
    assert result["data"][0].shape[:2] == (4, 8)
    assert _gray_levels(result["data"]) == [2, 3]


def test_camera_frames_validates_arguments(hiw_root: Path) -> None:
    episodes = hiw500.raw(str(hiw_root))
    with pytest.raises(ValueError, match="Unknown camera"):
        hiw500.camera_frames(episodes, cameras="chest")
    with pytest.raises(ValueError, match="together"):
        hiw500.camera_frames(episodes, width=32)


def test_cdr_reader_rejects_non_cdr() -> None:
    with pytest.raises(ValueError, match="CDR"):
        hiw500._CdrReader(b"\x01\x00\x00\x00")


# --------------------------------------------------------------------------- #
# LeRobot helper
# --------------------------------------------------------------------------- #


def test_lerobot_subtask_picks_active_label() -> None:
    entries = [
        {"role": "assistant", "content": "clean up the room", "style": "subtask", "timestamp": 0.0},
        {"role": "assistant", "content": "move to bed", "style": "subtask", "timestamp": 5.225},
        {"role": "assistant", "content": "pick trash", "style": "subtask", "timestamp": 31.412},
    ]
    frames = daft.from_pydict(
        {"timestamp": [0.0, 5.2, 5.225, 40.0, 1.0], "language_persistent": [entries] * 4 + [[]]}
    ).with_column("timestamp", col("timestamp").cast(DataType.float32()))
    result = frames.with_column("subtask", hiw500.lerobot_subtask()).to_pydict()
    assert result["subtask"] == ["clean up the room", "clean up the room", "move to bed", "pick trash", None]


# --------------------------------------------------------------------------- #
# Real HIW-500 (public): pinned to one repo revision
# --------------------------------------------------------------------------- #

PINNED_ROOT = "hf://datasets/BitRobot/HIW-500@c35830b9fcccf444c0318560466d0e35c7285675"
PINNED_TASK = "Clean-Up-The-Room"
PINNED_EPISODE = "Clean-Up-The-Room/episode_2026-02-25_10-16-03/episode_0001"
PINNED_START = 1_771_985_771_368_999_936
# LeRobot episode 0 is this episode; its t=0 is about 36 ms after info.json's start.
LEROBOT_OFFSET_NS = 36_000_000

requires_hf_token = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"), reason="set HF_TOKEN to run real-data tests (avoids Hub rate limits)"
)


def _pinned_episode() -> daft.DataFrame:
    return hiw500.raw(PINNED_ROOT, tasks=PINNED_TASK).where(col("episode_id") == PINNED_EPISODE).collect()


@pytest.mark.integration
@requires_hf_token
def test_hiw500_huggingface_smoke() -> None:
    episodes = _pinned_episode()
    catalog = episodes.select("mcap_size", "calibration_paths").to_pydict()
    assert catalog["mcap_size"] == [3_068_348_781]
    assert len(catalog["calibration_paths"][0]) == 2

    summary = hiw500.metadata(episodes).select("message_count", "chunk_count", "ros_distro", "topics").to_pydict()
    assert summary["message_count"] == [563_885]
    assert summary["chunk_count"] == [3_784]
    assert summary["ros_distro"] == ["humble"]
    assert hiw500.LOWSTATE_TOPIC in summary["topics"][0]

    details = hiw500.info(episodes).select("start_timestamp_ns", "subtask_count", "scene").to_pydict()
    assert details == {"start_timestamp_ns": [PINNED_START], "subtask_count": [21], "scene": [1]}
    labels = hiw500.subtasks(episodes).sort("subtask_index").limit(2).to_pydict()
    assert labels["label"] == ["move to bed", "pick trash"]
    assert labels["start_offset_seconds"][0] == pytest.approx(5.261)

    calibration = hiw500.calibration(episodes).select("camera", "stream", "width").to_pydict()
    assert sorted(set(calibration["camera"])) == ["left_wrist", "right_wrist"]
    assert len(calibration["stream"]) == 6

    start = PINNED_START + LEROBOT_OFFSET_NS
    states = hiw500.joint_states(episodes, start_time=start, end_time=start + 1_000_000_000).sort("log_time")
    rows = states.to_pydict()
    assert len(rows["log_time"]) == 100
    # Matches the LeRobot release's observation.state at episode 0, frame 0.
    expected = [-0.363, 0.024, -0.05, 0.645, -0.279, -0.01, -0.365, -0.016]
    assert rows["q"][0][:8] == pytest.approx(expected, abs=1e-3)

    wbc = hiw500.wbc_states(episodes, start_time=start, end_time=start + 1_000_000_000).to_pydict()
    assert 40 <= len(wbc["log_time"]) <= 60
    assert all(pivot is not None for pivot in wbc["pivot"])


@pytest.mark.integration
@requires_hf_token
def test_hiw500_huggingface_camera_smoke() -> None:
    start = PINNED_START + 10_000_000_000
    result = (
        hiw500.camera_frames(
            _pinned_episode(), cameras=["head", "left_wrist"], start_time=start, end_time=start + 200_000_000
        )
        .select("camera", "format", "width", "height")
        .to_pydict()
    )
    sizes = set(zip(result["camera"], result["width"], result["height"]))
    assert sizes == {("head", 1280, 480), ("left_wrist", 640, 480)}
    assert set(result["format"]) == {"jpeg"}
