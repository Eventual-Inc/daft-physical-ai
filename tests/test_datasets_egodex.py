from __future__ import annotations

from pathlib import Path

import daft
import pytest
from daft import DataType, MediaType, col

import daft_physical_ai.datasets.egodex as egodex_module
from daft_physical_ai.datasets.egodex import (
    DEFAULT_TRAJECTORY_FIELDS,
    JOINTS,
    TRAJECTORY_FIELDS,
    TRANSFORM_JOINTS,
    camera_frames,
    raw,
    trajectory,
)

NUM_FRAMES = 4


def _write_episode(
    root: Path,
    *,
    task: str = "fold_towel",
    episode_id: int = 0,
    with_video: bool = True,
    attrs: dict[str, object] | None = None,
) -> Path:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    task_dir = root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = task_dir / f"{episode_id}.hdf5"

    with h5py.File(hdf5_path, "w") as h5:
        h5.create_dataset("camera/intrinsic", data=np.eye(3, dtype=np.float32) * 736.0)
        for joint in JOINTS:
            h5.create_dataset(f"confidences/{joint}", data=np.linspace(0, 1, NUM_FRAMES, dtype=np.float32))
        for joint in TRANSFORM_JOINTS:
            h5.create_dataset(f"transforms/{joint}", data=np.tile(np.eye(4, dtype=np.float32), (NUM_FRAMES, 1, 1)))
        for key, value in (attrs or {"llm_description": "Fold the towel.", "llm_verbs": ["fold"]}).items():
            h5.attrs[key] = value
    if with_video:
        (task_dir / f"{episode_id}.mp4").write_bytes(b"")
    return hdf5_path


def _write_video(path: Path) -> None:
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 64, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def test_raw_catalogs_episodes_metadata_and_lazy_file_columns(tmp_path: Path) -> None:
    first = _write_episode(tmp_path)
    _write_episode(tmp_path, task="stack_cups", episode_id=3, with_video=False)

    df = raw(str(tmp_path))
    assert [field.name for field in df.schema()] == ["task", "episode_id", "metadata", "trajectory", "video"]
    assert df.schema()["metadata"].dtype == egodex_module._METADATA_DTYPE
    assert df.schema()["trajectory"].dtype == DataType.file(MediaType.hdf5())
    assert df.schema()["video"].dtype == DataType.file(MediaType.video())

    result = (
        df.select(
            "task",
            "episode_id",
            "metadata",
            col("trajectory").file_path().alias("trajectory_path"),
            col("video").file_path().alias("video_path"),
        )
        .sort("task")
        .collect()
        .to_pydict()
    )
    assert result["task"] == ["fold_towel", "stack_cups"]
    assert result["trajectory_path"] == [f"file://{first}", f"file://{tmp_path / 'stack_cups' / '3.hdf5'}"]
    assert result["video_path"] == [f"file://{first.with_suffix('.mp4')}", None]
    assert result["metadata"][0]["llm_description"] == "Fold the towel."
    assert result["metadata"][0]["llm_verbs"] == ["fold"]
    assert result["metadata"][0]["llm_objects"] == []


def test_raw_filters_from_paths_before_reading_metadata(tmp_path: Path) -> None:
    _write_episode(tmp_path, task="fold_towel", episode_id=0)
    _write_episode(tmp_path, task="fold_towel", episode_id=1)
    _write_episode(tmp_path, task="stack_cups", episode_id=0)

    result = raw(str(tmp_path), tasks="fold_towel", episode_ids=[1]).select("task", "episode_id").collect().to_pydict()
    assert result == {"task": ["fold_towel"], "episode_id": [1]}


def test_trajectory_reads_only_requested_fields(tmp_path: Path) -> None:
    _write_episode(tmp_path)
    result = trajectory(raw(str(tmp_path)), fields=["camera/intrinsic", "transforms/leftHand", "confidences/leftHand"])

    assert [field.name for field in result.schema()] == [
        "task",
        "episode_id",
        "metadata",
        "camera/intrinsic",
        "transforms/leftHand",
        "confidences/leftHand",
        "video",
    ]
    values = result.collect().to_pydict()
    assert _as_list(values["camera/intrinsic"][0]) == [[736.0, 0.0, 0.0], [0.0, 736.0, 0.0], [0.0, 0.0, 736.0]]
    assert len(_as_list(values["transforms/leftHand"][0])) == NUM_FRAMES
    assert _as_list(values["confidences/leftHand"][0]) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_trajectory_validates_its_input(tmp_path: Path) -> None:
    _write_episode(tmp_path)
    episodes = raw(str(tmp_path))
    with pytest.raises(ValueError, match="at least one"):
        trajectory(episodes, fields=[])
    with pytest.raises(ValueError, match="Unknown trajectory"):
        trajectory(episodes, fields=["transforms/not-a-joint"])
    with pytest.raises(ValueError, match="trajectory"):
        trajectory(daft.from_pydict({"task": ["fold_towel"]}))


def test_camera_frames_decodes_video_and_preserves_episode_granularity(tmp_path: Path) -> None:
    pytest.importorskip("av")
    _write_episode(tmp_path, with_video=False)
    _write_video(tmp_path / "fold_towel" / "0.mp4")

    result = camera_frames(raw(str(tmp_path)), width=16, height=12).select("video_frames").collect().to_pydict()
    assert len(result["video_frames"]) == 1
    assert len(result["video_frames"][0]) > 0


def test_camera_frames_handles_a_missing_video(tmp_path: Path) -> None:
    _write_episode(tmp_path, with_video=False)
    result = camera_frames(raw(str(tmp_path))).select("video_frames").collect().to_pydict()
    assert result["video_frames"] == [[]]


def test_field_catalog_matches_the_egodex_layout() -> None:
    assert len(JOINTS) == 68
    assert TRANSFORM_JOINTS[0] == "camera"
    assert len(TRAJECTORY_FIELDS) == 138
    assert set(DEFAULT_TRAJECTORY_FIELDS) <= set(TRAJECTORY_FIELDS)
