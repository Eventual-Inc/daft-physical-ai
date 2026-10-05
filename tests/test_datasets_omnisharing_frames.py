from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from daft import DataType
from daft.exceptions import DaftCoreException

from daft_physical_ai.datasets import omnisharing
from daft_physical_ai.datasets.omnisharing._frames import nearest_timestamp
from tests.omnisharing_datagen import episode_filename, write_df2_episode


@pytest.fixture
def frame_root(tmp_path: Path) -> Path:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=4)
    return tmp_path


def test_nearest_timestamp_is_signed_deterministic_and_validated() -> None:
    assert nearest_timestamp([0, 10], 5) == (0, -5)
    assert nearest_timestamp([0, 10], 6) == (1, 4)
    assert nearest_timestamp([10, 20], 0) == (0, 10)
    assert nearest_timestamp([10, 20], 30) == (1, -10)
    with pytest.raises(ValueError, match="at least one"):
        nearest_timestamp([], 0)
    with pytest.raises(ValueError, match="monotonically"):
        nearest_timestamp([2, 1], 0)


def test_frames_requires_explicit_valid_fields(frame_root: Path) -> None:
    episodes = omnisharing.raw(str(frame_root))
    with pytest.raises(ValueError, match="explicit field whitelist"):
        omnisharing.frames(episodes, [])
    with pytest.raises(ValueError, match="Unknown frame"):
        omnisharing.frames(episodes, "observation/lefthand/nope")
    assert "observation/lefthand/tactile" in omnisharing.FRAME_FIELDS


def test_frames_emits_exact_order_and_only_explicit_episode_columns(frame_root: Path) -> None:
    episodes = omnisharing.episode_metadata(omnisharing.raw(str(frame_root)))
    result = omnisharing.frames(
        episodes,
        ["observation/lefthand/joints", "observation/lefthand/joints"],
        include_columns=["instruction", "episode_key", "instruction"],
    )
    assert result.column_names == [
        "instruction",
        "episode_key",
        "frame_index",
        "timestamp_us",
        "observation/lefthand/joints",
    ]
    assert result.schema()["observation/lefthand/joints"].dtype == DataType.tensor(DataType.float32())
    assert len(result.to_pylist()) == 4


def test_frames_applies_action_lead_exactly_once(frame_root: Path) -> None:
    import h5py

    path = next(frame_root.glob("*.hdf5"))
    with h5py.File(path, "a") as h5:
        observation = h5["dataset/observation/lefthand/joints/data"]
        action = h5["dataset/action/lefthand/joints/data"]
        observation[...] = np.arange(4, dtype=np.float32)[:, None]
        action[...] = (10 + np.arange(4, dtype=np.float32))[:, None]
    rows = omnisharing.frames(
        omnisharing.raw(str(frame_root)),
        ["observation/lefthand/joints", "action/lefthand/joints"],
    ).to_pylist()
    assert [np.asarray(row["observation/lefthand/joints"])[0] for row in rows] == [0, 1, 2, 3]
    assert [np.asarray(row["action/lefthand/joints"])[0] for row in rows] == [11, 12, 13, 13]


def test_frames_aligns_independent_camera_clocks(frame_root: Path) -> None:
    rows = omnisharing.frames(
        omnisharing.raw(str(frame_root)),
        "observation/lefthand/joints",
        align_cameras=["RGB_Camera0", ("RGBD_0", "color"), ("RGBD_0", "left")],
    ).to_pylist()
    assert [row["RGB_Camera0/frame_index"] for row in rows] == [0, 1, 2, 3]
    assert [row["RGB_Camera0/timestamp_residual_us"] for row in rows] == [0, -2000, -4000, -6000]
    assert [row["RGBD_0/color/timestamp_residual_us"] for row in rows] == [0, 0, 0, 0]
    assert [row["RGBD_0/left/timestamp_residual_us"] for row in rows] == [0, -2000, -4000, -6000]


def test_frames_falls_back_to_camera_clock_for_missing_eye_clock(frame_root: Path) -> None:
    import h5py

    path = next(frame_root.glob("*.hdf5"))
    with h5py.File(path, "a") as h5:
        del h5["dataset/observation/image/RGBD_0/left_timestamp"]
    rows = omnisharing.frames(
        omnisharing.raw(str(frame_root)),
        "observation/lefthand/joints",
        align_cameras=[("RGBD_0", "left")],
    ).to_pylist()
    assert [row["RGBD_0/left/timestamp_residual_us"] for row in rows] == [0, 0, 0, 0]


def test_frames_missing_optional_camera_returns_null_alignment(frame_root: Path) -> None:
    rows = omnisharing.frames(
        omnisharing.raw(str(frame_root)),
        "observation/lefthand/joints",
        align_cameras=["RGB_Camera99"],
    ).to_pylist()
    assert [row["RGB_Camera99/frame_index"] for row in rows] == [None] * 4
    assert [row["RGB_Camera99/timestamp_residual_us"] for row in rows] == [None] * 4


def test_frames_expand_multiple_episodes(tmp_path: Path) -> None:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=2)
    write_df2_episode(tmp_path / episode_filename(2, "213000", 93, 110056), n_frames=3)
    rows = omnisharing.frames(
        omnisharing.raw(str(tmp_path)),
        "observation/lefthand/joints",
        include_columns="episode_key",
    ).to_pylist()
    assert len(rows) == 5
    assert sorted({row["episode_key"] for row in rows}) == [
        "episode_1_212953_93_110056_glove",
        "episode_2_213000_93_110056_glove",
    ]


def test_frames_rejects_field_length_mismatch(frame_root: Path) -> None:
    import h5py

    path = next(frame_root.glob("*.hdf5"))
    with h5py.File(path, "a") as h5:
        group = h5["dataset/observation/lefthand/joints"]
        del group["data"]
        group.create_dataset("data", data=np.zeros((3, 29), dtype=np.float32))
    with pytest.raises(DaftCoreException, match="observation/lefthand/joints/data has 3 frames but aligned_timestamp"):
        omnisharing.frames(omnisharing.raw(str(frame_root)), "observation/lefthand/joints").collect()


def test_frames_validates_include_columns(frame_root: Path) -> None:
    episodes = omnisharing.raw(str(frame_root))
    with pytest.raises(ValueError, match="Unknown include_columns"):
        omnisharing.frames(episodes, "observation/lefthand/joints", include_columns="missing")
    with pytest.raises(ValueError, match="cannot broadcast"):
        omnisharing.frames(episodes, "observation/lefthand/joints", include_columns="episode")


def test_frame_plan_construction_performs_no_reads(
    frame_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daft.file.hdf5 import Hdf5File

    def forbidden(*args, **kwargs):
        raise AssertionError("planning opened HDF5")

    monkeypatch.setattr(Hdf5File, "open", forbidden)
    result = omnisharing.frames(
        omnisharing.raw(str(frame_root)),
        "observation/lefthand/tactile",
        align_cameras=[("RGBD_0", "right")],
        include_columns="capture_key",
    )
    assert result.column_names == [
        "capture_key",
        "frame_index",
        "timestamp_us",
        "observation/lefthand/tactile",
        "RGBD_0/right/frame_index",
        "RGBD_0/right/timestamp_residual_us",
    ]
