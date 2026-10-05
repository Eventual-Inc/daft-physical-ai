from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from daft import DataType, col, lit
from daft.exceptions import DaftCoreException

from daft_physical_ai.datasets import omnisharing
from tests.omnisharing_datagen import HANDPOSE_ORDER as DISK_HANDPOSE_ORDER
from tests.omnisharing_datagen import JOINT_NAMES, episode_filename, write_df2_episode


@pytest.fixture
def signal_root(tmp_path: Path) -> Path:
    write_df2_episode(tmp_path / "DF-2" / episode_filename(1, "212953", 93, 110056), n_frames=4)
    write_df2_episode(
        tmp_path / "DF-2R" / episode_filename(1, "212953", 93, 110056, "mano"),
        n_frames=5,
        n_joints=17,
        tactile_width=3750,
    )
    write_df2_episode(tmp_path / "DF-1" / episode_filename(1, "212953", 93, 110056, None), n_frames=2)
    return tmp_path


def test_episode_metadata_has_fixed_order_and_values(signal_root: Path) -> None:
    result = omnisharing.episode_metadata(omnisharing.raw(str(signal_root), stage="DF-2"))
    assert result.column_names[-10:] == [
        "generated_time",
        "data_id",
        "vendor",
        "instruction",
        "frame_count",
        "audio_sample_rate",
        "audio_sample_count",
        "camera_names",
        "object_names",
        "task_labels_json",
    ]
    row = result.to_pylist()[0]
    assert row["frame_count"] == 4
    assert row["audio_sample_rate"] == 8000
    assert row["audio_sample_count"] == 160
    assert row["camera_names"] == [
        "RGBD_0",
        "RGBD_1",
        "RGB_Camera0",
        "RGB_Camera1",
        "RGB_Camera2",
        "RGB_Camera4",
        "RGB_Camera6",
    ]
    assert row["object_names"] == ["obj1", "obj2"]
    assert json.loads(row["task_labels_json"])["任务物品"] == ["燕京啤酒+红色啤酒架"]


def test_episode_metadata_missing_optional_values_are_null_or_empty(tmp_path: Path) -> None:
    path = write_df2_episode(
        tmp_path / episode_filename(1, "212953", 93, 110056),
        n_frames=3,
        n_objects=0,
        include_audio=False,
        include_meta=False,
    )
    import h5py

    with h5py.File(path, "a") as h5:
        del h5["dataset/observation/image"]
    row = omnisharing.episode_metadata(omnisharing.raw(str(tmp_path))).to_pylist()[0]
    assert row["vendor"] is None
    assert row["instruction"] is None
    assert row["audio_sample_rate"] is None
    assert row["audio_sample_count"] is None
    assert row["camera_names"] == []
    assert row["object_names"] == []
    assert row["task_labels_json"] == "{}"


def test_trajectory_uses_slash_fields_request_order_and_deduplication(signal_root: Path) -> None:
    episodes = omnisharing.raw(str(signal_root), stage="DF-2")
    fields = [
        "action/righthand/handpose",
        "observation/lefthand/joints",
        "action/righthand/handpose",
    ]
    result = omnisharing.trajectory(episodes, fields)
    assert result.column_names[-3:] == [*fields[:2], "observation/lefthand/joint_names"]
    assert result.schema()[fields[0]].dtype == DataType.tensor(DataType.float32())
    assert result.schema()["observation/lefthand/joint_names"].dtype == DataType.list(DataType.string())
    row = result.to_pylist()[0]
    assert np.asarray(row[fields[0]]).shape == (4, 7)
    assert np.asarray(row[fields[1]]).shape == (4, 29)
    assert row["observation/lefthand/joint_names"] == JOINT_NAMES


def test_trajectory_accepts_single_string_and_df2r_width(signal_root: Path) -> None:
    result = omnisharing.trajectory(
        omnisharing.raw(str(signal_root), stage="DF-2R"),
        "observation/lefthand/joints",
    )
    row = result.to_pylist()[0]
    assert np.asarray(row["observation/lefthand/joints"]).shape == (5, 17)
    assert row["observation/lefthand/joint_names"] == JOINT_NAMES[:17]


def test_trajectory_joint_names_follow_each_stage_layout(signal_root: Path) -> None:
    episodes = omnisharing.raw(str(signal_root)).where(col("stage") != lit("DF-1"))
    rows = omnisharing.trajectory(episodes, ["observation/righthand/joints", "action/lefthand/joints"]).sort("stage")
    df2, df2r = rows.to_pylist()
    for row, width in ((df2, 29), (df2r, 17)):
        for prefix in ("observation/righthand", "action/lefthand"):
            names = row[f"{prefix}/joint_names"]
            assert len(names) == width == np.asarray(row[f"{prefix}/joints"]).shape[1]
    assert df2["observation/righthand/joint_names"] != df2r["observation/righthand/joint_names"]


def test_handpose_fields_do_not_emit_joint_names(signal_root: Path) -> None:
    result = omnisharing.trajectory(omnisharing.raw(str(signal_root), stage="DF-2"), "observation/lefthand/handpose")
    assert not any(name.endswith("joint_names") for name in result.column_names)


def test_trajectory_rejects_joint_names_that_do_not_match_width(tmp_path: Path) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        h5["dataset/observation/lefthand/joints/data"].attrs["joint_names"] = np.array(JOINT_NAMES[:5], dtype=object)
    with pytest.raises(DaftCoreException, match="declares 5 joint_names for 29 joint columns"):
        omnisharing.trajectory(omnisharing.raw(str(tmp_path)), "observation/lefthand/joints").collect()


def test_trajectory_catalog_and_handpose_order_are_exported() -> None:
    assert len(omnisharing.TRAJECTORY_FIELDS) == 8
    assert omnisharing.DEFAULT_TRAJECTORY_FIELDS == omnisharing.TRAJECTORY_FIELDS
    assert omnisharing.HANDPOSE_ORDER == ("x", "y", "z", "qw", "qx", "qy", "qz")
    assert DISK_HANDPOSE_ORDER == "[x, y, z, qw, qx, qy, qz]"


@pytest.mark.parametrize("fields", [[], ["observation/lefthand/nope"]])
def test_trajectory_validates_selectors(signal_root: Path, fields: list[str]) -> None:
    with pytest.raises(ValueError):
        omnisharing.trajectory(omnisharing.raw(str(signal_root), stage="DF-2"), fields)


_MATERIALIZERS = {
    "episode_metadata": omnisharing.episode_metadata,
    "trajectory": lambda df: omnisharing.trajectory(df, "observation/lefthand/joints"),
    "tactile": lambda df: omnisharing.tactile(df, "lefthand", split_by_sensor=True),
    "audio": omnisharing.audio,
    "objects": omnisharing.objects,
    "cameras": omnisharing.cameras,
    "camera_payloads": lambda df: omnisharing.camera_payloads(df, "RGB_Camera0"),
    "camera_frames": lambda df: omnisharing.camera_frames(df, "RGB_Camera0"),
    "depth_frames": lambda df: omnisharing.depth_frames(df, "RGBD_0"),
    "stereo_extrinsics": lambda df: omnisharing.stereo_extrinsics(df, "RGBD_0"),
    "frames": lambda df: omnisharing.frames(df, "observation/lefthand/joints"),
}


@pytest.mark.parametrize("name", sorted(_MATERIALIZERS))
def test_materializers_reject_df1_clearly(signal_root: Path, name: str) -> None:
    # Daft 0.7.20 masks a nested-dtype UDF's exception with "Need at least 1
    # series to perform concat" when every row fails; the reader must not.
    episodes = omnisharing.raw(str(signal_root), stage="DF-1")
    with pytest.raises(DaftCoreException, match="DF-1 episodes are catalog-only") as raised:
        _MATERIALIZERS[name](episodes).collect()
    assert "concat" not in str(raised.value)


def test_read_errors_name_the_episode_in_a_mixed_batch(signal_root: Path) -> None:
    rows = omnisharing.raw(str(signal_root))
    with pytest.raises(DaftCoreException, match=r"DF-1/episode_1_212953_93_110056\.hdf5: ValueError"):
        omnisharing.trajectory(rows, "observation/lefthand/joints").collect()


def test_missing_tactile_dataset_fails_clearly(tmp_path: Path) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        del h5["dataset/observation/lefthand/tactile"]
    for split in (False, True):
        with pytest.raises(
            DaftCoreException,
            match="KeyError: Required OmniSharing tactile dataset is missing: dataset/observation/lefthand/tactile/data",
        ):
            omnisharing.tactile(omnisharing.raw(str(tmp_path)), "lefthand", split_by_sensor=split).collect()
    # The other hand is untouched and still reads.
    row = omnisharing.tactile(omnisharing.raw(str(tmp_path)), "righthand").to_pylist()[0]
    assert np.asarray(row["observation/righthand/tactile"]).shape == (3, 3465)


def test_trajectory_fails_for_missing_required_path(tmp_path: Path) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        del h5["dataset/observation/lefthand/joints/data"]
    with pytest.raises(DaftCoreException, match="trajectory dataset is missing: dataset/observation/lefthand/joints"):
        omnisharing.trajectory(omnisharing.raw(str(tmp_path)), "observation/lefthand/joints").collect()


def test_trajectory_fails_for_wrong_dtype(tmp_path: Path) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        node = h5["dataset/observation/lefthand/joints/data"]
        values = node[()].astype(np.float64)
        del h5["dataset/observation/lefthand/joints/data"]
        h5["dataset/observation/lefthand/joints"].create_dataset("data", data=values)
    with pytest.raises(DaftCoreException, match="must have dtype float32, found float64"):
        omnisharing.trajectory(omnisharing.raw(str(tmp_path)), "observation/lefthand/joints").collect()


def test_trajectory_validates_handpose_layout(tmp_path: Path) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        h5["dataset/observation/lefthand/handpose"].attrs["order"] = "[x, y, z, qx, qy, qz, qw]"
    with pytest.raises(DaftCoreException, match="hand-pose order must be"):
        omnisharing.trajectory(omnisharing.raw(str(tmp_path)), "observation/lefthand/handpose").collect()


def test_trajectory_validates_consistent_lengths(tmp_path: Path) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        group = h5["dataset/action/lefthand/joints"]
        del group["data"]
        group.create_dataset("data", data=np.zeros((4, 29), dtype=np.float32))
    with pytest.raises(DaftCoreException, match="inconsistent frame counts"):
        omnisharing.trajectory(
            omnisharing.raw(str(tmp_path)),
            ["observation/lefthand/joints", "action/lefthand/joints"],
        ).collect()


def test_signal_plan_construction_performs_no_reads(signal_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daft.file.hdf5 import Hdf5File

    def forbidden(*args, **kwargs):
        raise AssertionError("planned signal read opened HDF5")

    monkeypatch.setattr(Hdf5File, "open", forbidden)
    episodes = omnisharing.raw(str(signal_root), stage="DF-2")
    assert omnisharing.episode_metadata(episodes).schema()["frame_count"].dtype == DataType.int64()
    assert omnisharing.trajectory(episodes, "observation/lefthand/joints").schema()[
        "observation/lefthand/joints"
    ].dtype == DataType.tensor(DataType.float32())
