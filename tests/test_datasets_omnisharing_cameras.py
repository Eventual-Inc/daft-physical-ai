from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from daft import DataType
from daft.exceptions import DaftCoreException

from daft_physical_ai.datasets import omnisharing
from daft_physical_ai.datasets.omnisharing._cameras import _is_checked_camera
from tests.omnisharing_datagen import (
    RGB_CAMERA_IDS,
    RGBD_CAMERA_IDS,
    episode_filename,
    left_to_color_matrix,
    write_df2_episode,
)


@pytest.fixture
def camera_root(tmp_path: Path) -> Path:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=4)
    return tmp_path


def test_camera_inventory_has_exact_long_form_schema(camera_root: Path) -> None:
    result = omnisharing.cameras(omnisharing.raw(str(camera_root)))
    assert result.column_names == [
        "episode_key",
        "camera",
        "kind",
        "stream",
        "h5path",
        "codec",
        "payload_bytes",
        "clock_h5path",
        "clock_length",
        "width",
        "height",
        "intrinsics",
        "extrinsics",
        "distortion",
        "relative_to",
        "calibration_date",
        "is_checked_camera",
    ]
    assert result.schema()["intrinsics"].dtype == DataType.tensor(DataType.float32())
    rows = result.to_pylist()
    assert len(rows) == len(RGB_CAMERA_IDS) + len(RGBD_CAMERA_IDS) * len(omnisharing.RGBD_STREAMS)
    assert {row["camera"] for row in rows if row["kind"] == "rgb"} == {
        "RGB_Camera0",
        "RGB_Camera1",
        "RGB_Camera2",
        "RGB_Camera4",
        "RGB_Camera6",
    }


def test_camera_inventory_detects_codecs_clocks_and_reference_frames(camera_root: Path) -> None:
    rows = omnisharing.cameras(omnisharing.raw(str(camera_root))).to_pylist()
    rgb4 = next(row for row in rows if row["camera"] == "RGB_Camera4")
    left = next(row for row in rows if row["camera"] == "RGBD_0" and row["stream"] == "left")
    right = next(row for row in rows if row["camera"] == "RGBD_0" and row["stream"] == "right")
    assert rgb4["codec"] == "h26x-annexb"
    assert rgb4["clock_length"] == 11
    assert rgb4["relative_to"] == "RGB_Camera6"
    assert rgb4["is_checked_camera"] is True
    assert left["codec"] == "matroska"
    assert left["clock_length"] == 5
    assert left["clock_h5path"].endswith("RGBD_0/left_timestamp")
    assert right["clock_length"] == 4
    assert left["relative_to"] == "RGBD_0"
    assert np.asarray(left["extrinsics"]).shape == (4, 4)


def _add_high_camera_ids(path: Path, checked: str) -> None:
    """Give the episode RGB_Camera10..12, as real releases have, and set checked_cam_name."""
    import h5py

    with h5py.File(path, "a") as h5:
        image = h5["dataset/observation/image"]
        for camera_id in (10, 11, 12):
            image.copy("RGB_Camera1", f"RGB_Camera{camera_id}")
        image.attrs["checked_cam_name"] = checked


@pytest.mark.parametrize(
    "checked,expected",
    [("Camera1", "RGB_Camera1"), ("Camera12", "RGB_Camera12"), ("Camera4", "RGB_Camera4")],
)
def test_checked_camera_matches_exact_camera_id(tmp_path: Path, checked: str, expected: str) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    _add_high_camera_ids(path, checked)
    rows = omnisharing.cameras(omnisharing.raw(str(tmp_path))).to_pylist()
    assert {row["camera"] for row in rows if row["is_checked_camera"]} == {expected}


def test_is_checked_camera_rejects_prefix_and_substring_matches() -> None:
    assert _is_checked_camera("RGB_Camera1", "Camera1")
    assert _is_checked_camera("RGB_Camera1", "RGB_Camera1")
    assert not _is_checked_camera("RGB_Camera10", "Camera1")
    assert not _is_checked_camera("RGB_Camera11", "Camera1")
    assert not _is_checked_camera("RGB_Camera1", "Camera10")
    assert not _is_checked_camera("RGBD_1", "Camera1")
    assert not _is_checked_camera("RGB_Camera1", "")


@pytest.mark.parametrize(
    "selector,columns",
    [
        ("RGB_Camera0", ["RGB_Camera0/payload", "RGB_Camera0/codec"]),
        (("RGBD_0", "left"), ["RGBD_0/left/payload", "RGBD_0/left/codec"]),
    ],
)
def test_camera_payloads_accept_single_selectors(camera_root: Path, selector, columns: list[str]) -> None:
    result = omnisharing.camera_payloads(omnisharing.raw(str(camera_root)), selector)
    assert result.column_names[-2:] == columns
    row = result.to_pylist()[0]
    assert isinstance(row[columns[0]], bytes)
    assert row[columns[1]] in {"h26x-annexb", "matroska"}


def test_camera_payloads_deduplicate_and_use_one_file_session(
    camera_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daft.file.hdf5 import Hdf5File

    original = Hdf5File.open
    calls = 0

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Hdf5File, "open", counted)
    result = omnisharing.camera_payloads(
        omnisharing.raw(str(camera_root)),
        ["RGB_Camera0", ("RGBD_0", "color"), "RGB_Camera0"],
    )
    row = result.to_pylist()[0]
    assert row["RGB_Camera0/codec"] == "h26x-annexb"
    assert row["RGBD_0/color/codec"] == "matroska"
    assert calls == 1


@pytest.mark.parametrize("selector", [[], "RGBD_0", [("RGBD_0", "infrared")]])
def test_camera_payloads_validate_selectors(camera_root: Path, selector) -> None:
    with pytest.raises(ValueError):
        omnisharing.camera_payloads(omnisharing.raw(str(camera_root)), selector)


def test_missing_payload_is_optional(camera_root: Path) -> None:
    row = omnisharing.camera_payloads(
        omnisharing.raw(str(camera_root)),
        "RGB_Camera99",
    ).to_pylist()[0]
    assert row["RGB_Camera99/payload"] is None
    assert row["RGB_Camera99/codec"] is None


def test_depth_reads_only_requested_indices_in_requested_order(camera_root: Path) -> None:
    import h5py

    path = next(camera_root.glob("*.hdf5"))
    with h5py.File(path, "r") as h5:
        node = h5["dataset/observation/image/RGBD_0/aligned_depth"]
        expected = np.stack([node[index] for index in (3, 0, 2)])
    row = omnisharing.depth_frames(
        omnisharing.raw(str(camera_root)),
        ["RGBD_0", "RGBD_0"],
        frame_indices=[3, 0, 2],
    ).to_pylist()[0]
    np.testing.assert_array_equal(row["RGBD_0/depth"], expected)
    assert row["RGBD_0/depth_frame_indices"] == [3, 0, 2]


def test_missing_depth_is_typed_empty(camera_root: Path) -> None:
    row = omnisharing.depth_frames(omnisharing.raw(str(camera_root)), "RGBD_1", frame_indices=0).to_pylist()[0]
    assert np.asarray(row["RGBD_1/depth"]).shape == (0,)
    assert row["RGBD_1/depth_frame_indices"] == []


def test_depth_validates_bounds_and_arguments(camera_root: Path) -> None:
    episodes = omnisharing.raw(str(camera_root))
    with pytest.raises(ValueError):
        omnisharing.depth_frames(episodes, [], frame_indices=0)
    with pytest.raises(ValueError):
        omnisharing.depth_frames(episodes, "RGBD_0", frame_indices=[])
    with pytest.raises(IndexError):
        omnisharing.depth_frames(episodes, "RGBD_0", frame_indices=-1)
    with pytest.raises(DaftCoreException, match=r"aligned_depth has 4 frames; requested out-of-range indices \[99\]"):
        omnisharing.depth_frames(episodes, "RGBD_0", frame_indices=[1, 99], strict=True).collect()


def test_depth_reads_available_indices_by_default(camera_root: Path) -> None:
    row = omnisharing.depth_frames(
        omnisharing.raw(str(camera_root)),
        ["RGBD_0", "RGBD_1"],
        frame_indices=[3, 99, 1],
    ).to_pylist()[0]
    assert row["RGBD_0/depth_frame_indices"] == [3, 1]
    assert np.asarray(row["RGBD_0/depth"]).shape == (2, 8, 12)
    assert row["RGBD_1/depth_frame_indices"] == []


def test_depth_with_no_available_index_is_typed_empty(camera_root: Path) -> None:
    row = omnisharing.depth_frames(omnisharing.raw(str(camera_root)), "RGBD_0", frame_indices=[50, 99]).to_pylist()[0]
    assert np.asarray(row["RGBD_0/depth"]).shape == (0,)
    assert row["RGBD_0/depth_frame_indices"] == []


@pytest.fixture
def uneven_root(tmp_path: Path) -> Path:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    write_df2_episode(tmp_path / episode_filename(2, "213630", 115, 110092), n_frames=6)
    return tmp_path


def test_depth_lenient_mode_is_decided_per_episode(uneven_root: Path) -> None:
    rows = (
        omnisharing.depth_frames(omnisharing.raw(str(uneven_root)), "RGBD_0", frame_indices=[0, 4])
        .sort("episode_index")
        .to_pylist()
    )
    assert [row["RGBD_0/depth_frame_indices"] for row in rows] == [[0], [0, 4]]
    assert [np.asarray(row["RGBD_0/depth"]).shape[0] for row in rows] == [1, 2]


def test_depth_strict_mode_names_the_short_episode(uneven_root: Path) -> None:
    with pytest.raises(DaftCoreException, match=r"episode_1_212953_93_110056_glove\.hdf5: IndexError"):
        omnisharing.depth_frames(
            omnisharing.raw(str(uneven_root)),
            "RGBD_0",
            frame_indices=[0, 4],
            strict=True,
        ).collect()


def test_depth_strict_mode_ignores_cameras_without_depth(camera_root: Path) -> None:
    row = omnisharing.depth_frames(
        omnisharing.raw(str(camera_root)), "RGBD_1", frame_indices=99, strict=True
    ).to_pylist()[0]
    assert row["RGBD_1/depth_frame_indices"] == []


def test_stereo_extrinsics_are_stable_per_camera(camera_root: Path) -> None:
    row = omnisharing.stereo_extrinsics(omnisharing.raw(str(camera_root)), ["RGBD_1", "RGBD_0", "RGBD_1"]).to_pylist()[
        0
    ]
    np.testing.assert_allclose(row["RGBD_0/left_to_color"], left_to_color_matrix(0))
    np.testing.assert_allclose(row["RGBD_1/left_to_color"], left_to_color_matrix(1))
    assert row["RGBD_0/calibration_date"] == "20250925171218"


def test_missing_stereo_calibration_is_typed_empty(camera_root: Path) -> None:
    row = omnisharing.stereo_extrinsics(omnisharing.raw(str(camera_root)), "RGBD_99").to_pylist()[0]
    assert np.asarray(row["RGBD_99/left_to_color"]).shape == (0,)
    assert row["RGBD_99/calibration_date"] is None


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"{{{not json", "contains malformed calibration JSON"),
        (b"not-json", "contains malformed calibration JSON"),
        (json.dumps([1, 2, 3]).encode(), "must contain a left_to_color calibration matrix"),
        (json.dumps({"calib_date": "20250101000000"}).encode(), "must contain a left_to_color calibration matrix"),
        (
            json.dumps({"calib_date": "x", "left_to_color": [[1, 2], [3, 4]]}).encode(),
            r"left_to_color must have shape \(4, 4\)",
        ),
        (json.dumps({"calib_date": "x", "left_to_color": "nope"}).encode(), "left_to_color must be numeric"),
    ],
)
def test_present_malformed_stereo_calibration_fails(tmp_path: Path, payload: bytes, message: str) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        camera = h5["dataset/observation/image/RGBD_0"]
        del camera["inner_extrinsic"]
        camera.create_dataset("inner_extrinsic", data=np.array([payload]))
    with pytest.raises(DaftCoreException, match=f"RGBD_0/inner_extrinsic {message}"):
        omnisharing.stereo_extrinsics(omnisharing.raw(str(tmp_path)), "RGBD_0").collect()


def test_malformed_calibration_on_one_camera_names_that_camera(tmp_path: Path) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        camera = h5["dataset/observation/image/RGBD_1"]
        del camera["inner_extrinsic"]
        camera.create_dataset("inner_extrinsic", data=np.array([b"[]"]))
    episodes = omnisharing.raw(str(tmp_path))
    np.testing.assert_allclose(
        omnisharing.stereo_extrinsics(episodes, "RGBD_0").to_pylist()[0]["RGBD_0/left_to_color"],
        left_to_color_matrix(0),
    )
    with pytest.raises(DaftCoreException, match="RGBD_1/inner_extrinsic must contain"):
        omnisharing.stereo_extrinsics(episodes, ["RGBD_0", "RGBD_1"]).collect()


def test_camera_apis_construct_schemas_without_reads(
    camera_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daft.file.hdf5 import Hdf5File

    def forbidden(*args, **kwargs):
        raise AssertionError("planning opened HDF5")

    monkeypatch.setattr(Hdf5File, "open", forbidden)
    episodes = omnisharing.raw(str(camera_root))
    assert "camera" in omnisharing.cameras(episodes).column_names
    assert "RGB_Camera0/payload" in omnisharing.camera_payloads(episodes, "RGB_Camera0").column_names
    assert "RGBD_0/depth" in omnisharing.depth_frames(episodes, "RGBD_0").column_names
    assert "RGBD_0/left_to_color" in omnisharing.stereo_extrinsics(episodes, "RGBD_0").column_names
