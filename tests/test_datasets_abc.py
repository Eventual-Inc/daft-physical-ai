from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path

import daft
import pytest
from daft import DataType, col

pytest.importorskip("mcap")

from mcap.writer import CompressionType, IndexType, Writer

from daft_physical_ai.datasets import abc

FIRST_ID = "11111111-1111-1111-1111-111111111111"
SECOND_ID = "22222222-2222-2222-2222-222222222222"
FIRST_DIR = f"data/train/fold_the_towel/episode_{FIRST_ID}"
SECOND_DIR = f"data/val/pick_up_the_cup/episode_{SECOND_ID}"

FRAME_COUNT = 8
GOP = 4
FRAME_STEP_NS = 1_000


# --------------------------------------------------------------------------- #
# Synthetic MCAP fixtures
# --------------------------------------------------------------------------- #


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _field(number: int, value: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _timestamp(timestamp_ns: int) -> bytes:
    seconds, nanos = divmod(timestamp_ns, 1_000_000_000)
    return _varint(1 << 3) + _varint(seconds) + _varint(2 << 3) + _varint(nanos)


def _annotation_payload(timestamp_ns: int, label: str) -> bytes:
    return _field(1, _timestamp(timestamp_ns)) + _field(2, label.encode())


def _video_payload(timestamp_ns: int, frame: bytes, fmt: str) -> bytes:
    return _field(1, _timestamp(timestamp_ns)) + _field(2, b"camera") + _field(3, frame) + _field(4, fmt.encode())


def _encode_frames(codec: str) -> list[bytes]:
    """Encode FRAME_COUNT flat gray frames (level 20*i) as Annex-B access units, keyframe every GOP."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    try:
        context = av.CodecContext.create(codec, "w")
    except ValueError:  # UnknownCodecError
        pytest.skip(f"PyAV build has no {codec} encoder")
    context.width, context.height, context.pix_fmt = 64, 48, "yuv420p"
    context.time_base = Fraction(1, 30)
    params = f"keyint={GOP}:min-keyint={GOP}:scenecut=0:bframes=0"
    if codec == "libx265":
        context.options = {"x265-params": f"{params}:log-level=none"}
    else:
        context.options = {"x264-params": params}
    packets = []
    for index in range(FRAME_COUNT):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 20 * index, dtype=np.uint8), format="rgb24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = index
        packets.extend(context.encode(frame))
    packets.extend(context.encode(None))
    packets.sort(key=lambda packet: packet.pts)
    assert len(packets) == FRAME_COUNT
    return [bytes(packet) for packet in packets]


def _write_episode(
    path: Path,
    *,
    metadata_name: str,
    metadata: dict[str, str],
    base_time: int,
    video: dict[str, tuple[str, list[bytes]]] | None = None,
    indexed: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        options = {} if indexed else {"index_types": IndexType.NONE, "use_summary_offsets": False}
        writer = Writer(
            output,
            chunk_size=256,
            compression=CompressionType.NONE,
            use_statistics=indexed,
            repeat_channels=indexed,
            repeat_schemas=indexed,
            **options,
        )
        writer.start()
        state_schema = writer.register_schema("yam.RobotState", "protobuf", b"state-schema")
        video_schema = writer.register_schema("foxglove.CompressedVideo", "protobuf", b"video-schema")
        state = writer.register_channel("/left-arm-state", "protobuf", state_schema)
        top = writer.register_channel("/top-camera", "protobuf", video_schema)
        channels = {"/top-camera": top}
        for topic in video or {}:
            if topic not in channels:
                channels[topic] = writer.register_channel(topic, "protobuf", video_schema)
        for sequence in range(2):
            writer.add_message(
                state,
                log_time=base_time + sequence * 10,
                publish_time=base_time + sequence * 10 + 1,
                sequence=sequence,
                # Quotes, backslashes, and non-UTF-8 bytes exercise the released reader's str(bytes) round trip.
                data=b"state-" + bytes([sequence, 0xFF, 0x27, 0x22, 0x5C]),
            )
        for topic, (fmt, frames) in (video or {}).items():
            for index, frame in enumerate(frames):
                log_time = base_time + index * FRAME_STEP_NS
                writer.add_message(
                    channels[topic],
                    log_time=log_time,
                    publish_time=log_time,
                    sequence=index,
                    data=_video_payload(log_time, frame, fmt),
                )
        writer.add_metadata(metadata_name, metadata)
        writer.finish()


def _write_annotations(path: Path, *, timestamp_ns: int, labels: list[str]) -> None:
    with path.open("wb") as output:
        writer = Writer(output, chunk_size=128, compression=CompressionType.NONE)
        writer.start()
        schema = writer.register_schema("yam.Annotation", "protobuf", b"annotation-schema")
        channel = writer.register_channel(abc.ANNOTATION_TOPIC, "protobuf", schema)
        for sequence, label in enumerate(labels):
            message_time = timestamp_ns + sequence * 10
            writer.add_message(
                channel,
                log_time=message_time,
                publish_time=message_time,
                sequence=sequence,
                data=_annotation_payload(message_time, label),
            )
        writer.add_message(channel, log_time=timestamp_ns + 100, publish_time=0, sequence=9, data=b"\xff\xff")
        writer.finish()


@pytest.fixture
def abc_root(tmp_path: Path) -> Path:
    _write_episode(
        tmp_path / FIRST_DIR / "episode.mcap",
        metadata_name="episode-metadata",
        metadata={
            "session_id": FIRST_ID,
            "operator_id": "operator-one",
            "task_name": "fold the towel",
            "duration": "1.25",
        },
        base_time=1_000,
    )
    _write_annotations(tmp_path / FIRST_DIR / "annotation.mcap", timestamp_ns=1_000, labels=["pick up towel", "fold"])
    _write_episode(
        tmp_path / SECOND_DIR / "episode.mcap",
        metadata_name="session-metadata",
        metadata={"session-uuid": SECOND_ID, "operator-id": "operator-two", "instruction": "pick up the cup"},
        base_time=2_000,
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# raw()
# --------------------------------------------------------------------------- #


def test_raw_builds_episode_catalog(abc_root: Path) -> None:
    episodes = abc.raw(str(abc_root))
    assert episodes.column_names == list(abc._CATALOG_COLUMNS)
    assert episodes.schema()["episode_mcap"].dtype == DataType.file()

    result = (
        episodes.with_column("episode_file", col("episode_mcap").file_path())
        .exclude("episode_mcap", "annotation_mcap")
        .sort("episode_id")
        .to_pydict()
    )
    assert result["split"] == ["train", "val"]
    assert result["task_slug"] == ["fold_the_towel", "pick_up_the_cup"]
    assert result["episode_id"] == [FIRST_ID, SECOND_ID]
    assert result["annotated"] == [True, False]
    assert result["episode_size"] == [
        (abc_root / FIRST_DIR / "episode.mcap").stat().st_size,
        (abc_root / SECOND_DIR / "episode.mcap").stat().st_size,
    ]
    assert result["annotation_size"][0] > 0
    assert result["annotation_size"][1] is None
    assert result["annotation_path"][1] is None
    assert result["episode_file"] == result["episode_path"]
    assert result["episode_dir"][0].endswith(FIRST_DIR)


def test_raw_annotation_file_column_is_null_when_absent(abc_root: Path) -> None:
    result = (
        abc.raw(str(abc_root))
        .select("episode_id", col("annotation_mcap").file_path().alias("annotation_file"))
        .sort("episode_id")
        .to_pydict()
    )
    assert result["annotation_file"][0].endswith(f"{FIRST_DIR}/annotation.mcap")
    assert result["annotation_file"][1] is None


def test_raw_filters_split_and_task(abc_root: Path) -> None:
    result = abc.raw(str(abc_root), split="train", tasks="fold_the_towel").select("episode_id").to_pydict()
    assert result == {"episode_id": [FIRST_ID]}


def test_raw_task_filter_is_exact_and_accepts_task_prefix(abc_root: Path) -> None:
    # ``*fold_the_towel`` also lists this suffix-sharing task; the slug filter must drop it.
    _write_episode(
        abc_root / "data/train/unfold_the_towel/episode_44444444-4444-4444-4444-444444444444/episode.mcap",
        metadata_name="episode-metadata",
        metadata={},
        base_time=4_000,
    )
    _write_episode(
        abc_root / "data/train/task=documented_task/episode_33333333-3333-3333-3333-333333333333/episode.mcap",
        metadata_name="episode-metadata",
        metadata={},
        base_time=3_000,
    )
    result = (
        abc.raw(str(abc_root), tasks=["fold_the_towel", "task=documented_task"])
        .select("task_slug", "episode_id")
        .sort("task_slug")
        .to_pydict()
    )
    assert result["task_slug"] == ["documented_task", "fold_the_towel"]


def test_raw_without_annotations(abc_root: Path) -> None:
    result = abc.raw(str(abc_root), include_annotations=False).select("annotated", "annotation_path").to_pydict()
    assert result == {"annotated": [False, False], "annotation_path": [None, None]}


def test_raw_rejects_bad_arguments(abc_root: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        abc.raw(str(abc_root), split="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact task slugs"):
        abc.raw(str(abc_root), tasks="fold_*")
    with pytest.raises(ValueError, match="at least one"):
        abc.raw(str(abc_root), tasks=[])


class _FakeRepoFile:
    def __init__(self, path: str, size: int) -> None:
        self.path = path
        self.size = size


def test_raw_hf_listing_uses_recursive_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_api = pytest.importorskip("huggingface_hub.hf_api")
    from huggingface_hub.utils import EntryNotFoundError

    monkeypatch.setattr(hf_api, "RepoFile", _FakeRepoFile)
    calls: list[tuple[str, str, str | None]] = []
    tree = {
        "data/train/task=clip_socks": [
            _FakeRepoFile(f"data/train/task=clip_socks/episode_{FIRST_ID}/episode.mcap", 10),
            _FakeRepoFile(f"data/train/task=clip_socks/episode_{FIRST_ID}/annotation.mcap", 3),
            _FakeRepoFile(f"data/train/task=clip_socks/episode_{FIRST_ID}/README.md", 1),
        ]
    }

    class FakeApi:
        def __init__(self, token=None, library_name=None) -> None:
            self.token = token

        def list_repo_tree(self, repo_id, path_in_repo, recursive, revision, repo_type):
            assert recursive and repo_type == "dataset"
            calls.append((repo_id, path_in_repo, revision))
            if path_in_repo not in tree:
                raise EntryNotFoundError("missing")
            return iter(tree[path_in_repo])

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    episodes = abc.raw("hf://datasets/XDOF/ABC-130k@abc123", split="train", tasks="clip_socks")
    result = episodes.select("task_slug", "episode_id", "episode_size", "annotation_size").to_pydict()

    assert calls == [
        ("XDOF/ABC-130k", "data/train/clip_socks", "abc123"),
        ("XDOF/ABC-130k", "data/train/task=clip_socks", "abc123"),
    ]
    assert result == {
        "task_slug": ["clip_socks"],
        "episode_id": [FIRST_ID],
        "episode_size": [10],
        "annotation_size": [3],
    }

    calls.clear()
    empty = abc.raw("hf://datasets/XDOF/ABC-130k", split="val", tasks="missing").to_pydict()
    assert all(values == [] for values in empty.values())
    assert set(empty) == set(abc._CATALOG_COLUMNS)


def test_hf_io_config_picks_up_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    resolved = abc._resolve_hf_io_config(None, ["hf://datasets/XDOF/ABC-130k"])
    assert resolved is not None and resolved.hf.token == "hf_test"
    assert abc._resolve_hf_io_config(None, ["/local/mirror"]) is None


# --------------------------------------------------------------------------- #
# metadata()
# --------------------------------------------------------------------------- #


def test_metadata_normalizes_current_and_legacy_records(abc_root: Path) -> None:
    result = abc.metadata(abc.raw(str(abc_root))).sort("episode_id").to_pydict()

    assert result["session_id"] == [FIRST_ID, SECOND_ID]
    assert result["operator_id"] == ["operator-one", "operator-two"]
    assert result["task_name"] == ["fold the towel", "pick up the cup"]
    assert result["duration_seconds"][0] == 1.25
    assert result["duration_seconds"][1] == pytest.approx(10 / 1_000_000_000)
    assert result["message_count"] == [2, 2]
    assert result["message_start_time"] == [1_000, 2_000]
    assert result["message_end_time"] == [1_010, 2_010]
    assert all(count >= 1 for count in result["chunk_count"])
    assert result["topics"] == [["/left-arm-state", "/top-camera"]] * 2
    assert result["video_topics"] == [["/top-camera"]] * 2
    assert result["indexed"] == [True, True]
    assert json.loads(result["episode_metadata_json"][0])["name"] == "episode-metadata"
    assert json.loads(result["episode_metadata_json"][1])["metadata"]["instruction"] == "pick up the cup"
    # Catalog columns pass through.
    assert result["annotated"] == [True, False]


def test_metadata_falls_back_to_a_scan_without_summary(tmp_path: Path) -> None:
    _write_episode(
        tmp_path / FIRST_DIR / "episode.mcap",
        metadata_name="episode-metadata",
        metadata={"task_name": "fold the towel"},
        base_time=5_000,
        indexed=False,
    )
    result = abc.metadata(abc.raw(str(tmp_path))).to_pydict()
    assert result["indexed"] == [False]
    assert result["chunk_count"] == [None]
    assert result["message_count"] == [2]
    assert result["message_start_time"] == [5_000]
    assert result["task_name"] == ["fold the towel"]
    assert result["topics"] == [["/left-arm-state", "/top-camera"]]


def test_metadata_requires_catalog() -> None:
    with pytest.raises(ValueError, match="episode_mcap"):
        abc.metadata(daft.from_pydict({"x": [1]}))


# --------------------------------------------------------------------------- #
# messages() / annotations()
# --------------------------------------------------------------------------- #


def test_messages_reads_bounded_paths_with_identity(abc_root: Path) -> None:
    episodes = abc.raw(str(abc_root))
    result = abc.messages(episodes, topics="/left-arm-state", start_time=1_005)
    assert result.schema()["data"].dtype == DataType.binary()
    assert result.schema()["log_time"].dtype == DataType.int64()
    assert result.schema()["sequence"].dtype == DataType.int64()

    rows = result.sort(["episode_id", "log_time"]).to_pydict()
    state = [b"state-" + bytes([sequence, 0xFF, 0x27, 0x22, 0x5C]) for sequence in (0, 1)]
    assert rows["data"] == [state[1], state[0], state[1]]
    assert rows["log_time"] == [1_010, 2_000, 2_010]
    assert rows["split"] == ["train", "val", "val"]
    assert rows["task_slug"] == ["fold_the_towel", "pick_up_the_cup", "pick_up_the_cup"]
    assert rows["episode_id"] == [FIRST_ID, SECOND_ID, SECOND_ID]
    assert rows["file_kind"] == ["episode"] * 3
    assert rows["source_path"][0].endswith(f"{FIRST_DIR}/episode.mcap")


def test_messages_end_time_is_exclusive(abc_root: Path) -> None:
    episodes = abc.raw(str(abc_root)).where(col("episode_id") == FIRST_ID)
    rows = abc.messages(episodes, topics=None, start_time=1_000, end_time=1_010).to_pydict()
    assert rows["log_time"] == [1_000]


def test_messages_handles_empty_catalog(abc_root: Path) -> None:
    episodes = abc.raw(str(abc_root)).where(col("episode_id") == "missing")
    result = abc.messages(episodes).to_pydict()
    assert all(values == [] for values in result.values())
    assert {"source_path", "topic", "data", "episode_id", "file_kind"} <= set(result)


def test_messages_rejects_bad_files_argument(abc_root: Path) -> None:
    with pytest.raises(ValueError, match="files"):
        abc.messages(abc.raw(str(abc_root)), files="video")  # type: ignore[arg-type]


def test_normalize_messages_accepts_native_reader_types() -> None:
    # Daft main's native read_mcap: source_path present, unsigned ints, binary data.
    native = daft.from_pydict(
        {
            "source_path": ["/x/data/train/t/episode_e/episode.mcap"],
            "topic": ["/a"],
            "log_time": [5],
            "publish_time": [6],
            "sequence": [7],
            "data": [b"\x00\xff"],
        }
    ).with_columns(
        {
            "log_time": col("log_time").cast(DataType.uint64()),
            "publish_time": col("publish_time").cast(DataType.uint64()),
            "sequence": col("sequence").cast(DataType.uint32()),
        }
    )
    result = abc._normalize_messages(native, "/ignored")
    assert [field.dtype for field in result.schema()] == [
        DataType.string(),
        DataType.string(),
        DataType.int64(),
        DataType.int64(),
        DataType.int64(),
        DataType.binary(),
    ]
    assert result.to_pydict()["source_path"] == ["/x/data/train/t/episode_e/episode.mcap"]
    assert result.to_pydict()["data"] == [b"\x00\xff"]


def test_annotations_decode_free_form_labels(abc_root: Path) -> None:
    episodes = abc.raw(str(abc_root))
    result = abc.annotations(episodes).sort("sequence").to_pydict()

    assert result["label"] == ["pick up towel", "fold", None]
    assert result["timestamp_ns"] == [1_000, 1_010, None]
    assert result["log_time"] == [1_000, 1_010, 1_100]
    assert result["episode_id"] == [FIRST_ID] * 3
    assert result["file_kind"] == ["annotation"] * 3


def test_annotations_empty_when_no_episode_is_annotated(abc_root: Path) -> None:
    episodes = abc.raw(str(abc_root)).where(~col("annotated"))
    result = abc.annotations(episodes).to_pydict()
    assert result["label"] == []


# --------------------------------------------------------------------------- #
# camera_frames()
# --------------------------------------------------------------------------- #


@pytest.fixture
def video_root(tmp_path: Path) -> Path:
    _write_episode(
        tmp_path / FIRST_DIR / "episode.mcap",
        metadata_name="episode-metadata",
        metadata={},
        base_time=10_000,
        video={
            "/top-camera": ("h264", _encode_frames("libx264")),
            "/left-wrist-camera": ("h265", _encode_frames("libx265")),
        },
    )
    return tmp_path


def _gray_levels(images: list) -> list[int]:
    return [round(float(image.mean()) / 20) for image in images]


def test_camera_frames_decodes_h264_and_h265(video_root: Path) -> None:
    frames = abc.camera_frames(abc.raw(str(video_root)), cameras=["top", "left_wrist"])
    assert frames.schema()["data"].dtype == DataType.image("RGB")

    result = frames.sort(["topic", "log_time"]).to_pydict()
    # Absent top streams (/top-left-camera, /top-right-camera) contribute no rows.
    assert result["topic"] == ["/left-wrist-camera"] * FRAME_COUNT + ["/top-camera"] * FRAME_COUNT
    assert result["camera"] == ["left_wrist"] * FRAME_COUNT + ["top_mono"] * FRAME_COUNT
    assert result["format"] == ["h265"] * FRAME_COUNT + ["h264"] * FRAME_COUNT
    assert result["timestamp_ns"] == result["log_time"]
    assert result["frame_id"] == ["camera"] * (2 * FRAME_COUNT)
    assert result["is_key_frame"][:FRAME_COUNT] == [index % GOP == 0 for index in range(FRAME_COUNT)]
    assert _gray_levels(result["data"]) == list(range(FRAME_COUNT)) * 2
    assert set(result["width"]) == {64} and set(result["height"]) == {48}
    assert result["episode_id"] == [FIRST_ID] * (2 * FRAME_COUNT)


def test_camera_frames_window_starting_mid_gop(video_root: Path) -> None:
    # Frames 5..6: the decoder must warm up from the keyframe at index 4.
    start = 10_000 + 5 * FRAME_STEP_NS
    end = 10_000 + 7 * FRAME_STEP_NS
    result = (
        abc.camera_frames(abc.raw(str(video_root)), cameras="top_mono", start_time=start, end_time=end)
        .sort("log_time")
        .to_pydict()
    )
    assert result["log_time"] == [start, start + FRAME_STEP_NS]
    assert result["sequence"] == [5, 6]
    assert result["is_key_frame"] == [False, False]
    assert _gray_levels(result["data"]) == [5, 6]


def test_camera_frames_skips_frames_before_the_first_keyframe(tmp_path: Path) -> None:
    # A stream that opens mid-GOP: frames 0..1 have no reference and must be dropped, not crash the decoder.
    _write_episode(
        tmp_path / FIRST_DIR / "episode.mcap",
        metadata_name="episode-metadata",
        metadata={},
        base_time=0,
        video={"/top-camera": ("h264", _encode_frames("libx264")[GOP - 2 :])},
    )
    result = abc.camera_frames(abc.raw(str(tmp_path)), cameras="top_mono").sort("log_time").to_pydict()
    assert result["sequence"] == list(range(2, FRAME_COUNT - GOP + 2))
    assert result["is_key_frame"][0] is True
    assert _gray_levels(result["data"]) == list(range(GOP, FRAME_COUNT))


def test_camera_frames_resizes(video_root: Path) -> None:
    result = (
        abc.camera_frames(abc.raw(str(video_root)), cameras="left_wrist", width=32, height=16)
        .select("width", "height", "data")
        .to_pydict()
    )
    assert set(result["width"]) == {32} and set(result["height"]) == {16}
    assert result["data"][0].shape == (16, 32, 3)


def test_camera_frames_validates_arguments(abc_root: Path) -> None:
    episodes = abc.raw(str(abc_root))
    with pytest.raises(ValueError, match="Unknown camera"):
        abc.camera_frames(episodes, cameras="overhead")
    with pytest.raises(ValueError, match="together"):
        abc.camera_frames(episodes, width=32)


# --------------------------------------------------------------------------- #
# Real ABC-130k (gated): run with HF_TOKEN set
# --------------------------------------------------------------------------- #

PINNED_ROOT = "hf://datasets/XDOF/ABC-130k@29136bc9b9e38d320b00ffcddbbe4cd0e3278c58"
PINNED_TASK = "clip_the_socks_to_the_hanger"
PINNED_EPISODE = "5b33995f-ba4a-49f8-bfb7-c6c034df0865"

requires_hf_token = pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="ABC-130k is gated; set HF_TOKEN")


def _pinned_episode() -> daft.DataFrame:
    return abc.raw(PINNED_ROOT, split="train", tasks=PINNED_TASK).where(col("episode_id") == PINNED_EPISODE).limit(1)


@pytest.mark.integration
@requires_hf_token
def test_abc_huggingface_smoke() -> None:
    episodes = _pinned_episode()
    sniffed = abc.metadata(episodes).select("episode_id", "message_count", "chunk_count", "indexed").to_pydict()
    assert sniffed == {"episode_id": [PINNED_EPISODE], "message_count": [9_849], "chunk_count": [18], "indexed": [True]}

    rows = abc.messages(
        episodes,
        topics="/left-arm-state",
        start_time=1_750_450_520_000_000_000,
        end_time=1_750_450_521_000_000_000,
    ).to_pydict()
    assert rows["episode_id"] and set(rows["episode_id"]) == {PINNED_EPISODE}
    assert set(rows["topic"]) == {"/left-arm-state"}
    assert all(isinstance(payload, bytes) and payload for payload in rows["data"])

    labels = abc.annotations(episodes).select("episode_id", "timestamp_ns", "label").to_pydict()
    assert len(labels["label"]) == 9
    assert all(timestamp is not None for timestamp in labels["timestamp_ns"])
    assert all(labels["label"])


@pytest.mark.integration
@requires_hf_token
def test_abc_huggingface_camera_smoke() -> None:
    result = (
        abc.camera_frames(
            _pinned_episode(),
            cameras="left_wrist",
            start_time=1_750_450_525_202_761_905,
            end_time=1_750_450_525_503_013_648,
        )
        .select("episode_id", "camera", "topic", "format", "width", "height")
        .to_pydict()
    )
    assert result["episode_id"] and set(result["episode_id"]) == {PINNED_EPISODE}
    assert set(result["camera"]) == {"left_wrist"}
    assert set(result["format"]) == {"h264"}
    assert set(zip(result["width"], result["height"])) == {(640, 480)}
