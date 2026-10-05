from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from daft import DataType
from daft.exceptions import DaftCoreException

from daft_physical_ai.datasets import omnisharing
from tests.omnisharing_datagen import (
    PLAYABLE_FRAMES,
    PLAYABLE_HEIGHT,
    PLAYABLE_WIDTH,
    episode_filename,
    write_df2_episode,
)


@pytest.fixture
def video_root(tmp_path: Path) -> Path:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=4)
    return tmp_path


def test_camera_frames_has_video_frames_compatible_schema_and_typed_missing(video_root: Path) -> None:
    result = omnisharing.camera_frames(omnisharing.raw(str(video_root)), "RGB_Camera99")
    dtype = result.schema()["RGB_Camera99/frames"].dtype
    expected = DataType.list(
        DataType.struct(
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
    )
    assert dtype == expected
    assert result.to_pylist()[0]["RGB_Camera99/frames"] == []


def test_camera_frames_requires_explicit_valid_selectors(video_root: Path) -> None:
    episodes = omnisharing.raw(str(video_root))
    with pytest.raises(ValueError):
        omnisharing.camera_frames(episodes, [])
    with pytest.raises(ValueError, match="must include a stream"):
        omnisharing.camera_frames(episodes, "RGBD_0")
    result = omnisharing.camera_frames(
        episodes,
        [("RGBD_0", "left"), ("RGBD_0", "left")],
    )
    assert result.column_names.count("RGBD_0/left/frames") == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_time": -1},
        {"start_time": 1, "end_time": 1},
        {"width": 10},
        {"width": 0, "height": 10},
        {"sample_interval_seconds": 0},
        {"is_key_frame": "yes"},
        {"max_frames": 0},
        {"max_frames": -1},
        {"max_frames": True},
    ],
)
def test_camera_frames_validates_droid_compatible_options(video_root: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        omnisharing.camera_frames(omnisharing.raw(str(video_root)), "RGB_Camera0", **kwargs)


def test_present_unsupported_stream_fails_clearly(video_root: Path) -> None:
    import h5py

    path = next(video_root.glob("*.hdf5"))
    with h5py.File(path, "a") as h5:
        node = h5["dataset/observation/image/RGB_Camera0/data"]
        node[:16] = np.arange(16, dtype=np.uint8) + 80
    with pytest.raises(DaftCoreException, match="RGB_Camera0/data has unsupported embedded video codec 'unknown'"):
        omnisharing.camera_frames(omnisharing.raw(str(video_root)), "RGB_Camera0").collect()


def test_camera_frame_plan_construction_performs_no_reads(
    video_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daft.file.hdf5 import Hdf5File

    def forbidden(*args, **kwargs):
        raise AssertionError("planning opened HDF5")

    monkeypatch.setattr(Hdf5File, "open", forbidden)
    result = omnisharing.camera_frames(
        omnisharing.raw(str(video_root)),
        [("RGBD_0", "color")],
        start_time=0.1,
        end_time=0.5,
        width=32,
        height=24,
        sample_interval_seconds=0.2,
        max_frames=None,
    )
    assert "RGBD_0/color/frames" in result.column_names


def test_camera_frames_caps_frames_by_default() -> None:
    import inspect

    assert inspect.signature(omnisharing.camera_frames).parameters["max_frames"].default == 1


RUN_DECODE = os.environ.get("OMNISHARING_RUN_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not RUN_DECODE, reason="set OMNISHARING_RUN_INTEGRATION=1 to exercise local codecs")
@pytest.mark.parametrize(
    "selector,column",
    [
        ("RGB_Camera0", "RGB_Camera0/frames"),
        (("RGBD_0", "color"), "RGBD_0/color/frames"),
    ],
)
def test_actual_annexb_and_matroska_decode(tmp_path: Path, selector, column: str) -> None:
    write_df2_episode(
        tmp_path / episode_filename(1, "212953", 93, 110056),
        n_frames=4,
        playable=True,
    )
    frames = omnisharing.camera_frames(
        omnisharing.raw(str(tmp_path)),
        selector,
        width=32,
        height=24,
        sample_interval_seconds=0.15,
        max_frames=None,
    ).to_pylist()[0][column]
    assert frames
    assert list(frames[0]) == [
        "frame_index",
        "frame_time",
        "frame_time_base",
        "frame_pts",
        "frame_dts",
        "frame_duration",
        "is_key_frame",
        "data",
    ]
    assert np.asarray(frames[0]["data"]).shape == (24, 32, 3)


@pytest.mark.integration
@pytest.mark.skipif(not RUN_DECODE, reason="set OMNISHARING_RUN_INTEGRATION=1 to exercise local codecs")
@pytest.mark.parametrize(
    "selector,column", [("RGB_Camera0", "RGB_Camera0/frames"), (("RGBD_0", "color"), "RGBD_0/color/frames")]
)
@pytest.mark.parametrize("max_frames,expected", [(None, PLAYABLE_FRAMES), (1, 1), (2, 2), (99, PLAYABLE_FRAMES)])
def test_max_frames_caps_decoded_frames(tmp_path: Path, selector, column: str, max_frames, expected: int) -> None:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=4, playable=True)
    kwargs = {} if max_frames == 1 else {"max_frames": max_frames}
    frames = omnisharing.camera_frames(omnisharing.raw(str(tmp_path)), selector, **kwargs).to_pylist()[0][column]
    assert len(frames) == expected
    assert [frame["frame_index"] for frame in frames] == list(range(expected))
    assert np.asarray(frames[0]["data"]).shape == (PLAYABLE_HEIGHT, PLAYABLE_WIDTH, 3)
