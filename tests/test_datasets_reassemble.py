from __future__ import annotations

import io
import json
import os
import wave
import zipfile
from pathlib import Path

import daft
import pytest
from daft import DataType, MediaType, col

from daft_physical_ai.datasets import reassemble
from daft_physical_ai.datasets.reassemble import (
    CAMERAS,
    DEFAULT_ROBOT_STATE_FIELDS,
    MICROPHONES,
    ROBOT_STATE_FIELDS,
    audio,
    camera_frames,
    download,
    events,
    lerobot_sidecars,
    raw,
    remote_catalog,
    robot_state,
    segments,
)

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
av = pytest.importorskip("av")

T0 = 1_736_603_018.0  # Unix seconds, the clock every REASSEMBLE stream uses
FULL = "2025-01-11-14-43-37"
PARTIAL = "2025-01-10-16-17-40"
NUM_FRAMES = 6
EVENT_RATE = 1_000
ROBOT_STATE_SHAPES = {
    "compensated_base_force": (40, 3),
    "compensated_base_torque": (40, 3),
    "gripper_efforts": (90, 2),
    "gripper_positions": (90, 2),
    "gripper_velocities": (90, 2),
    "joint_efforts": (90, 7),
    "joint_positions": (90, 7),
    "joint_velocities": (90, 7),
    "measured_force": (100, 3),
    "measured_torque": (100, 3),
    "pose": (60, 7),
    "velocity": (60, 6),
}
DURATION = 1.0


def _mp4(width: int, height: int, frames: int = NUM_FRAMES, fps: int = 30) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(frames):
            image = np.full((height, width, 3), index * 40, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _mp3(sample_rate: int, channels: int, seconds: float = 0.25) -> bytes:
    buffer = io.BytesIO()
    layout = "mono" if channels == 1 else "stereo"
    with av.open(buffer, "w", format="mp3") as container:
        stream = container.add_stream("mp3", rate=sample_rate, layout=layout)
        total, chunk = int(sample_rate * seconds), 1152
        tone = np.sin(np.linspace(0, 200 * np.pi, total, dtype=np.float32)) * 0.2
        for offset in range(0, total, chunk):
            block = np.tile(tone[offset : offset + chunk], (channels, 1)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _segment(group, index: int | None, text: str, start: float, end: float, success: bool) -> None:
    group.create_dataset("start", data=start)
    group.create_dataset("end", data=end)
    group.create_dataset("success", data=success)
    group.create_dataset("text", data=text.encode(), dtype=h5py.string_dtype())
    if index is not None:
        group.create_dataset("index", data=index)


def _write_recording(root: Path, name: str = FULL, *, with_hand: bool = True, with_low_level: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.h5"
    with h5py.File(path, "w") as h5:
        cameras = {"hama1": (32, 24), "hama2": (32, 24), "hand": (32, 24), "capture_node-camera-image": (24, 16)}
        for dataset, (width, height) in cameras.items():
            if dataset == "hand" and not with_hand:
                continue
            h5.create_dataset(dataset, data=np.void(_mp4(width, height)))
            h5.create_dataset(f"timestamps/{dataset}", data=T0 + np.arange(NUM_FRAMES) / 30.0)
        for microphone, (rate, channels) in {"hama1": (16_000, 2), "hama2": (16_000, 2), "hand": (48_000, 1)}.items():
            mp3 = np.frombuffer(_mp3(rate, channels), dtype=np.uint8).astype(np.int64)
            h5.create_dataset(f"{microphone}_audio", data=mp3)
        count = int(EVENT_RATE * DURATION)
        xs = np.arange(count) % 346
        h5.create_dataset("events", data=np.stack([xs, xs % 260, xs % 2], axis=1).astype(np.int64))
        h5.create_dataset("timestamps/events", data=T0 + np.arange(count) / EVENT_RATE)
        for field, (samples, width) in ROBOT_STATE_SHAPES.items():
            values = np.arange(samples * width, dtype=np.float64).reshape(samples, width)
            h5.create_dataset(f"robot_state/{field}", data=values)
            h5.create_dataset(f"timestamps/{field}", data=T0 + np.arange(samples) * (DURATION / samples))
        info = h5.create_group("segments_info")
        _segment(info.create_group("0"), 0, "No action.", T0, T0 + 0.2, True)
        pick = info.create_group("1")
        _segment(pick, 1, "Pick square peg 3.", T0 + 0.2, T0 + 0.9, False)
        if with_low_level:
            _segment(pick.create_group("low_level/0"), None, "Grasp", T0 + 0.2, T0 + 0.6, True)
            _segment(pick.create_group("low_level/1"), None, "Lift", T0 + 0.6, T0 + 0.9, False)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _write_recording(tmp_path)
    _write_recording(tmp_path / "nested", PARTIAL, with_hand=False, with_low_level=False)
    return tmp_path


def _decoded(mp3: bytes) -> np.ndarray:
    with av.open(io.BytesIO(mp3)) as container:
        stream = container.streams.audio[0]
        frames = [frame.to_ndarray().reshape(frame.layout.nb_channels, -1) for frame in container.decode(stream)]
    return np.concatenate(frames, axis=1)


def test_raw_catalogs_recordings_from_paths(dataset: Path) -> None:
    df = raw(str(dataset))
    assert [field.name for field in df.schema()] == ["recording", "file"]
    assert df.schema()["file"].dtype == DataType.file(MediaType.hdf5())  # type: ignore[attr-defined]

    result = df.select("recording", col("file").file_path().alias("path")).sort("recording").to_pydict()
    assert result["recording"] == [PARTIAL, FULL]
    assert result["path"] == [f"file://{dataset / 'nested' / f'{PARTIAL}.h5'}", f"file://{dataset / f'{FULL}.h5'}"]

    assert raw(str(dataset), recordings=FULL).select("recording").to_pydict() == {"recording": [FULL]}
    assert raw(f"{dataset}/*.h5").select("recording").to_pydict() == {"recording": [FULL]}


def test_segments_high_level_rows_nest_low_level_skills(dataset: Path) -> None:
    result = segments(raw(str(dataset))).sort(["recording", "segment_index"]).to_pydict()
    assert result["recording"] == [PARTIAL, PARTIAL, FULL, FULL]
    assert result["text"] == ["No action.", "Pick square peg 3.", "No action.", "Pick square peg 3."]
    assert result["success"] == [True, False, True, False]
    assert result["start"][3] == pytest.approx(T0 + 0.2)
    assert result["duration"][3] == pytest.approx(0.7)
    assert result["low_level"][1] == []
    assert [skill["text"] for skill in result["low_level"][3]] == ["Grasp", "Lift"]
    assert [skill["success"] for skill in result["low_level"][3]] == [True, False]


def test_segments_low_level_rows_carry_the_parent_action(dataset: Path) -> None:
    result = segments(raw(str(dataset)), level="low").sort(["recording", "skill_index"]).to_pydict()
    assert result["recording"] == [FULL, FULL]
    assert result["segment_index"] == [1, 1]
    assert result["segment_text"] == ["Pick square peg 3.", "Pick square peg 3."]
    assert result["text"] == ["Grasp", "Lift"]
    assert result["end"][1] == pytest.approx(T0 + 0.9)
    with pytest.raises(ValueError, match="level"):
        segments(raw(str(dataset)), level="mid")  # type: ignore[arg-type]


def test_robot_state_keeps_native_rates(dataset: Path) -> None:
    result = robot_state(raw(str(dataset), recordings=FULL)).to_pydict()
    for field in DEFAULT_ROBOT_STATE_FIELDS:
        assert result[field][0].shape == ROBOT_STATE_SHAPES[field]
        assert result[f"{field}_timestamps"][0].shape == (ROBOT_STATE_SHAPES[field][0],)
    assert result["measured_force"][0].dtype == np.float64


def test_robot_state_windows_every_stream_by_time(dataset: Path) -> None:
    result = robot_state(
        raw(str(dataset), recordings=FULL),
        fields=["measured_force", "pose"],
        start_time=T0 + 0.25,
        end_time=T0 + 0.5,
    ).to_pydict()
    assert result["measured_force"][0].shape == (25, 3)  # 100 Hz in the fixture
    assert result["pose"][0].shape == (15, 7)  # 60 Hz in the fixture
    assert result["measured_force"][0][0].tolist() == [75.0, 76.0, 77.0]
    assert result["measured_force_timestamps"][0][0] == pytest.approx(T0 + 0.25)


def test_absent_streams_yield_empty_or_no_rows(dataset: Path) -> None:
    with h5py.File(dataset / f"{FULL}.h5", "a") as h5:
        for name in ("robot_state/gripper_efforts", "hand_audio", "events", "segments_info"):
            del h5[name]
    episodes = raw(str(dataset), recordings=FULL)

    state = robot_state(episodes, fields=["gripper_efforts", "pose"]).to_pydict()
    assert state["gripper_efforts"][0].size == 0
    assert state["gripper_efforts_timestamps"][0].size == 0
    assert state["pose"][0].shape == ROBOT_STATE_SHAPES["pose"]

    assert audio(episodes, "hand").count_rows() == 0
    assert audio(episodes, "hand", decode=False).count_rows() == 0
    assert audio(episodes).select("microphone").sort("microphone").to_pydict() == {"microphone": ["hama1", "hama2"]}
    assert events(episodes).count_rows() == 0
    assert segments(episodes).count_rows() == 0


def test_robot_state_validates_fields(dataset: Path) -> None:
    episodes = raw(str(dataset))
    with pytest.raises(ValueError, match="at least one"):
        robot_state(episodes, fields=[])
    with pytest.raises(ValueError, match="Unknown robot_state"):
        robot_state(episodes, fields=["torque"])
    with pytest.raises(ValueError, match="file"):
        robot_state(daft.from_pydict({"recording": [FULL]}))


def test_audio_decodes_each_microphone(dataset: Path) -> None:
    result = audio(raw(str(dataset), recordings=FULL)).sort("microphone").to_pydict()
    assert result["microphone"] == ["hama1", "hama2", "hand"]
    assert result["sample_rate"] == [16_000, 16_000, 48_000]
    assert result["channels"] == [2, 2, 1]
    assert [samples.shape for samples in result["samples"]] == [
        (channels, samples) for channels, samples in zip(result["channels"], result["num_samples"])
    ]
    assert result["samples"][2].dtype == np.float32


def test_audio_without_decoding_returns_the_mp3_bitstream(dataset: Path) -> None:
    result = audio(raw(str(dataset), recordings=FULL), "hand", decode=False).to_pydict()
    with h5py.File(dataset / f"{FULL}.h5") as h5:
        expected = h5["hand_audio"][()].astype(np.uint8).tobytes()
    assert result == {"recording": [FULL], "microphone": ["hand"], "mp3": [expected]}
    with pytest.raises(ValueError, match="Unknown microphones"):
        audio(raw(str(dataset)), "boom")


def test_events_window_matches_a_full_read(dataset: Path) -> None:
    episodes = raw(str(dataset), recordings=FULL)
    full = events(episodes).to_pydict()
    assert full["num_events"] == [EVENT_RATE]
    assert full["events"][0].shape == (EVENT_RATE, 3)

    window = events(episodes, start_time=T0 + 0.1, end_time=T0 + 0.35).to_pydict()
    timestamps = full["timestamps"][0]
    keep = (timestamps >= T0 + 0.1) & (timestamps < T0 + 0.35)
    assert window["num_events"] == [int(keep.sum())]
    assert np.array_equal(window["events"][0], full["events"][0][keep])
    assert np.array_equal(window["timestamps"][0], timestamps[keep])

    empty = events(episodes, start_time=T0 + 5, end_time=T0 + 6).to_pydict()
    assert empty["num_events"] == [0]


def test_camera_frames_decode_with_per_frame_timestamps(dataset: Path) -> None:
    result = (
        camera_frames(raw(str(dataset)), cameras=["hand", "event_cam"])
        .sort(["recording", "camera", "frame_index"])
        .to_pydict()
    )
    # The partial recording has no hand camera, so it contributes event_cam frames only.
    pairs = list(zip(result["recording"], result["camera"]))
    assert pairs.count((PARTIAL, "event_cam")) == NUM_FRAMES
    assert pairs.count((PARTIAL, "hand")) == 0
    assert pairs.count((FULL, "hand")) == NUM_FRAMES
    hand = [index for index, pair in enumerate(pairs) if pair == (FULL, "hand")]
    assert [result["frame_index"][index] for index in hand] == list(range(NUM_FRAMES))
    assert result["timestamp"][hand[2]] == pytest.approx(T0 + 2 / 30)
    assert result["image"][hand[0]].shape == (24, 32, 3)
    event_cam = pairs.index((FULL, "event_cam"))
    assert (result["width"][event_cam], result["height"][event_cam]) == (24, 16)


def test_camera_frames_window_and_resize(dataset: Path) -> None:
    result = camera_frames(
        raw(str(dataset), recordings=FULL),
        "hama1",
        start_time=T0 + 2 / 30 - 1e-6,
        end_time=T0 + 4 / 30 - 1e-6,
        width=16,
        height=12,
    ).to_pydict()
    assert result["frame_index"] == [2, 3]
    assert [image.shape for image in result["image"]] == [(12, 16, 3), (12, 16, 3)]


def test_camera_frames_validates_arguments(dataset: Path) -> None:
    episodes = raw(str(dataset))
    with pytest.raises(ValueError, match="Unknown cameras"):
        camera_frames(episodes, "top")
    with pytest.raises(ValueError, match="together"):
        camera_frames(episodes, width=16)


def _write_port(root: Path, source: Path, *, real_pcm: bool = False) -> None:
    """A local stand-in for the robot-lev/reassemble side folders, built like the real port."""
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "splits.json").write_text(
        json.dumps({"0": {"recording": FULL, "split": "test"}, "1": {"recording": PARTIAL, "split": "unassigned"}})
    )
    with h5py.File(source) as h5:
        for index in (0, 1):
            folder = root / "audio" / f"episode_{index:06d}"
            folder.mkdir(parents=True)
            for microphone in MICROPHONES:
                payload = h5[f"{microphone}_audio"][()].astype("<i4")
                if real_pcm:
                    payload = (np.sin(np.arange(1600)) * 1e9).astype("<i4")
                with wave.open(str(folder / f"{microphone}.wav"), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(4)
                    wav.setframerate(16_000)
                    wav.writeframes(payload.tobytes())
        (root / "events").mkdir()
        for index in (0, 1):
            np.savez_compressed(
                root / "events" / f"episode_{index:06d}.npz",
                events=h5["events"][()],
                timestamps=h5["timestamps/events"][()],
            )


def test_lerobot_sidecars_catalog_the_port_side_files(dataset: Path, tmp_path: Path) -> None:
    port = tmp_path / "port"
    _write_port(port, dataset / f"{FULL}.h5")
    sidecars = lerobot_sidecars(str(port))
    assert sidecars.schema().column_names() == [
        "episode_index",
        "recording",
        "split",
        "hama1_audio",
        "hama2_audio",
        "hand_audio",
        "events_file",
    ]
    result = sidecars.select(
        "episode_index", "recording", "split", col("hand_audio").file_path().alias("hand")
    ).to_pydict()
    assert result["episode_index"] == [0, 1]
    assert result["recording"] == [FULL, PARTIAL]
    assert result["split"] == ["test", "unassigned"]
    assert result["hand"][0].endswith(f"{port}/audio/episode_000000/hand.wav")


def test_port_audio_recovers_the_original_mp3(dataset: Path, tmp_path: Path) -> None:
    port = tmp_path / "port"
    _write_port(port, dataset / f"{FULL}.h5")
    episode = lerobot_sidecars(str(port)).where(col("episode_index") == 0)

    mp3 = audio(episode, "hand", decode=False).to_pydict()
    original = audio(raw(str(dataset), recordings=FULL), "hand", decode=False).to_pydict()
    assert mp3["mp3"] == original["mp3"]

    decoded = audio(episode).sort("microphone").to_pydict()
    assert decoded["episode_index"] == [0, 0, 0]
    assert decoded["sample_rate"] == [16_000, 16_000, 48_000]
    assert np.array_equal(decoded["samples"][2], _decoded(original["mp3"][0]))


def test_port_audio_reads_real_pcm_as_pcm(dataset: Path, tmp_path: Path) -> None:
    port = tmp_path / "port"
    _write_port(port, dataset / f"{FULL}.h5", real_pcm=True)
    result = audio(lerobot_sidecars(str(port)).where(col("episode_index") == 0), "hand").to_pydict()
    assert (result["sample_rate"], result["channels"], result["num_samples"]) == ([16_000], [1], [1600])
    assert float(np.abs(result["samples"][0]).max()) <= 1.0


def test_port_events_match_the_hdf5_events(dataset: Path, tmp_path: Path) -> None:
    port = tmp_path / "port"
    _write_port(port, dataset / f"{FULL}.h5")
    window = {"start_time": T0 + 0.1, "end_time": T0 + 0.35}
    from_port = events(lerobot_sidecars(str(port)).where(col("episode_index") == 0), **window).to_pydict()
    from_h5 = events(raw(str(dataset), recordings=FULL), **window).to_pydict()
    assert from_port["num_events"] == from_h5["num_events"]
    assert np.array_equal(from_port["events"][0], from_h5["events"][0])
    assert np.array_equal(from_port["timestamps"][0], from_h5["timestamps"][0])


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, headers: dict[str, str]) -> None:
        super().__init__(data)
        self.headers = headers


def _serve(monkeypatch: pytest.MonkeyPatch, archive: bytes) -> list[tuple[int, int]]:
    requests: list[tuple[int, int]] = []

    def fake_range(url: str, start: int, end: int) -> _FakeResponse:
        requests.append((start, end))
        return _FakeResponse(archive[start : end + 1], {"Content-Range": f"bytes {start}-{end}/{len(archive)}"})

    monkeypatch.setattr(reassemble, "_http_range", fake_range)
    return requests


def _archive(dataset: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data/", b"")
        archive.write(dataset / f"{FULL}.h5", f"data/{FULL}.h5")
        archive.write(dataset / "nested" / f"{PARTIAL}.h5", f"data/{PARTIAL}.h5", compress_type=zipfile.ZIP_STORED)
    return buffer.getvalue()


def test_remote_catalog_reads_only_the_central_directory(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive(dataset)
    requests = _serve(monkeypatch, archive)
    result = remote_catalog("https://example.invalid/data.zip").to_pydict()
    assert result["recording"] == [PARTIAL, FULL]
    assert result["size_bytes"] == [
        os.path.getsize(dataset / "nested" / f"{PARTIAL}.h5"),
        (dataset / f"{FULL}.h5").stat().st_size,
    ]
    assert result["compressed_bytes"][1] < result["size_bytes"][1]
    assert sum(end - start + 1 for start, end in requests) < len(archive) // 4


def test_download_extracts_selected_members(dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _archive(dataset))
    dest = tmp_path / "downloaded"
    paths = download([FULL, PARTIAL], str(dest), url="https://example.invalid/data.zip")
    assert paths == [str(dest / f"{FULL}.h5"), str(dest / f"{PARTIAL}.h5")]
    assert (dest / f"{FULL}.h5").read_bytes() == (dataset / f"{FULL}.h5").read_bytes()
    assert (dest / f"{PARTIAL}.h5").read_bytes() == (dataset / "nested" / f"{PARTIAL}.h5").read_bytes()
    assert segments(raw(str(dest), recordings=FULL)).count_rows() == 2

    with pytest.raises(ValueError, match="not in the archive"):
        download("2030-01-01-00-00-00", str(dest), url="https://example.invalid/data.zip")
    with pytest.raises(ValueError, match="at least one"):
        download([], str(dest), url="https://example.invalid/data.zip")


def test_catalogs_match_the_release_layout() -> None:
    assert set(CAMERAS) == {"hama1", "hama2", "hand", "event_cam"}
    assert CAMERAS["event_cam"] == "capture_node-camera-image"
    assert set(ROBOT_STATE_FIELDS) == set(ROBOT_STATE_SHAPES)
    assert set(DEFAULT_ROBOT_STATE_FIELDS) <= set(ROBOT_STATE_FIELDS)


# --------------------------------------------------------------------------- #
# Real data (opt-in)
# --------------------------------------------------------------------------- #

# The smallest recording in the archive: 24 MB to transfer, 102 MB extracted.
PINNED_RECORDING = "2025-01-11-14-43-37"
PINNED_PORT = "hf://datasets/robot-lev/reassemble@37d242d3532fb17b6f749a26dad6d7aaacf810cd"
REASSEMBLE_ROOT = os.environ.get("REASSEMBLE_ROOT")


@pytest.mark.integration
@pytest.mark.skipif(
    not REASSEMBLE_ROOT, reason="set REASSEMBLE_ROOT to a directory for extracted recordings; download() fills it"
)
def test_reassemble_tuwien_smoke() -> None:
    assert REASSEMBLE_ROOT is not None
    download(PINNED_RECORDING, REASSEMBLE_ROOT)  # no-op when the file is already there
    episodes = raw(REASSEMBLE_ROOT, recordings=PINNED_RECORDING)

    high = segments(episodes).sort("segment_index").to_pydict()
    assert high["text"] == ["No action.", "Pick square peg 3.", "No action.", "No action."]
    assert high["success"] == [True, False, True, True]
    low = segments(episodes, level="low").sort("skill_index").to_pydict()
    assert list(zip(low["text"], low["success"])) == [("Grasp", True), ("Lift", False)]

    state = robot_state(episodes, fields=["measured_force", "joint_positions", "pose"]).to_pydict()
    assert state["measured_force"][0].shape == (23_240, 3)
    assert state["joint_positions"][0].shape == (22_549, 7)
    assert state["pose"][0].shape == (13_401, 7)

    sound = audio(episodes).sort("microphone").to_pydict()
    assert sound["sample_rate"] == [16_000, 16_000, 48_000]
    assert sound["num_samples"] == [372_096, 372_096, 1_123_200]

    assert events(episodes).select("num_events").to_pydict() == {"num_events": [2_551_295]}

    frames = camera_frames(episodes, ["hand", "event_cam"], width=64, height=48)
    counts = frames.groupby("camera").agg(col("frame_index").count().alias("n")).sort("camera").to_pydict()
    assert counts == {"camera": ["event_cam", "hand"], "n": [136, 642]}


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="set HF_TOKEN to read the Hugging Face LeRobot port")
def test_reassemble_lerobot_port_sidecars_smoke() -> None:
    sidecars = lerobot_sidecars(PINNED_PORT)
    catalog = sidecars.select("episode_index", "recording", "split").to_pydict()
    assert len(catalog["episode_index"]) == 149
    assert {split: catalog["split"].count(split) for split in set(catalog["split"])} == {
        "train": 111,
        "test": 37,
        "unassigned": 1,
    }
    episode = sidecars.where(col("recording") == PINNED_RECORDING)
    hand = audio(episode, "hand").to_pydict()
    assert (hand["episode_index"], hand["sample_rate"], hand["num_samples"]) == ([21], [48_000], [1_123_200])
    assert events(episode).select("num_events").to_pydict() == {"num_events": [2_551_295]}
