from __future__ import annotations

import os

import numpy as np
import pytest
from daft import col

from daft_physical_ai.datasets import omnisharing

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OMNISHARING_RUN_INTEGRATION") != "1",
        reason="set OMNISHARING_RUN_INTEGRATION=1 for public OmniSharing validation",
    ),
]

DATASET = "paxini/Omnisharing_DB_SampleData"
PINNED_EPISODE = "episode_1203_213135_115_110092_glove"


@pytest.fixture(scope="module")
def episode():
    return omnisharing.raw(DATASET, stage="DF-2").where(col("episode_key").endswith(PINNED_EPISODE)).limit(1)


def test_public_catalog_and_pinned_metadata(episode) -> None:
    row = omnisharing.episode_metadata(episode).to_pylist()[0]
    assert row["frame_count"] == 207
    assert row["audio_sample_rate"] == 8000
    assert row["audio_sample_count"] == 60416
    assert row["vendor"] == "paxini"
    assert len(row["camera_names"]) == 14


def test_public_signals_and_objects(episode) -> None:
    row = omnisharing.trajectory(
        episode,
        ["observation/lefthand/joints", "observation/lefthand/handpose"],
    ).to_pylist()[0]
    assert np.asarray(row["observation/lefthand/joints"]).shape == (207, 29)
    assert len(row["observation/lefthand/joint_names"]) == 29
    assert row["observation/lefthand/joint_names"][0] == "J1J"
    assert np.asarray(row["observation/lefthand/handpose"]).shape == (207, 7)
    tactile = omnisharing.tactile(episode, "lefthand").to_pylist()[0]
    assert np.asarray(tactile["observation/lefthand/tactile"]).shape == (207, 3465)
    assert omnisharing.objects(episode).to_pylist()[0]["n_objects"] == 0


def test_public_checked_camera_is_a_single_exact_match(episode) -> None:
    rows = omnisharing.cameras(episode).to_pylist()
    cameras = {row["camera"] for row in rows}
    assert {"RGB_Camera1", "RGB_Camera10", "RGB_Camera11", "RGB_Camera12"} <= cameras
    assert len({row["camera"] for row in rows if row["is_checked_camera"]}) == 1


def test_public_depth_stereo_and_camera_families(episode) -> None:
    depth = omnisharing.depth_frames(episode, ["RGBD_0", "RGBD_1"], frame_indices=[0, 100, 500]).to_pylist()[0]
    assert np.asarray(depth["RGBD_0/depth"]).shape == (2, 720, 1280)
    assert depth["RGBD_0/depth_frame_indices"] == [0, 100]
    assert np.asarray(depth["RGBD_1/depth"]).shape == (0,)
    matrix = omnisharing.stereo_extrinsics(episode, "RGBD_0").to_pylist()[0]["RGBD_0/left_to_color"]
    assert np.asarray(matrix).shape == (4, 4)
    decoded = omnisharing.camera_frames(
        episode,
        ["RGB_Camera0", ("RGBD_0", "color")],
        end_time=0.1,
        width=64,
        height=48,
    ).to_pylist()[0]
    assert len(decoded["RGB_Camera0/frames"]) == 1
    assert len(decoded["RGBD_0/color/frames"]) == 1
    assert np.asarray(decoded["RGB_Camera0/frames"][0]["data"]).shape == (48, 64, 3)


def test_public_frame_alignment(episode) -> None:
    rows = omnisharing.frames(
        episode,
        "observation/lefthand/joints",
        align_cameras=["RGB_Camera0", ("RGBD_0", "left")],
        include_columns="episode_key",
    ).to_pylist()
    assert len(rows) == 207
    assert rows[0]["RGB_Camera0/frame_index"] != rows[0]["frame_index"]
