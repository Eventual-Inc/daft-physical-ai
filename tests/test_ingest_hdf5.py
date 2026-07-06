"""HDF5 ingest tests - robomimic/LIBERO demos.

Self-contained: synthetic h5py fixtures, no real datasets. The adapter routes
through ``Episode.to_step_rows`` -> ``ROLLOUT_SCHEMA``, so these assert both the
parsing AND that ingested demos land on the same canonical schema the rollout
path emits.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from daft_physical_ai.episodes import assert_emits_schema, write_rows
from daft_physical_ai.ingest import Hdf5Ingestor


def _write_demo(
    parent,
    name,
    *,
    n,
    gripper=0.5,
    native=False,
    with_obs=True,
    with_rewards=True,
    with_dones=True,
    success=True,
):
    g = parent.create_group(name)
    actions = np.zeros((n, 7), dtype=np.float32)
    actions[:, -1] = gripper
    for i in range(n):
        actions[i, :6] = 0.01 * i
    g.create_dataset("actions", data=actions)
    if with_rewards:
        rewards = np.zeros(n, dtype=np.float32)
        if success:
            rewards[-1] = 1.0
        g.create_dataset("rewards", data=rewards)
    if with_dones:
        dones = np.zeros(n, dtype=np.int64)
        if success:
            dones[-1] = 1
        g.create_dataset("dones", data=dones)
    if with_obs:
        obs = g.create_group("obs")
        eef = np.tile(np.array([0.1, 0.2, 0.3], np.float32), (n, 1))
        gqpos = np.tile(np.array([0.04, -0.04], np.float32), (n, 1))  # diff -> 0.08
        img = np.zeros((n, 8, 8, 3), np.uint8)
        if native:
            obs.create_dataset("agentview_rgb", data=img)
            obs.create_dataset("eye_in_hand_rgb", data=img)
            obs.create_dataset("ee_pos", data=eef)
            obs.create_dataset("ee_ori", data=np.zeros((n, 3), np.float32))  # axis-angle
            obs.create_dataset("gripper_states", data=gqpos)
        else:
            obs.create_dataset("agentview_image", data=img)
            obs.create_dataset("robot0_eye_in_hand_image", data=img)
            obs.create_dataset("robot0_eef_pos", data=eef)
            obs.create_dataset("robot0_eef_quat", data=np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1)))
            obs.create_dataset("robot0_gripper_qpos", data=gqpos)


def _write_robomimic(
    path,
    *,
    native=False,
    with_obs=True,
    with_rewards=True,
    with_dones=True,
    problem_info=True,
    suite="libero_goal",
    bddl="/x/put_bowl.bddl",
    demos=None,
    mask=None,
):
    if demos is None:
        demos = [("demo_0", 3, True), ("demo_10", 5, True), ("demo_2", 4, True)]
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        if problem_info:
            data.attrs["problem_info"] = json.dumps(
                {
                    "problem_name": "KitchenDemo",
                    "domain_name": suite,
                    "language_instruction": "put the bowl on the plate",
                }
            )
            data.attrs["bddl_file_name"] = bddl
        for name, n, success in demos:
            _write_demo(
                data,
                name,
                n=n,
                native=native,
                with_obs=with_obs,
                with_rewards=with_rewards,
                with_dones=with_dones,
                success=success,
            )
        if mask is not None:
            mask_group = f.create_group("mask")
            for split, demo_ids in mask.items():
                mask_group.create_dataset(split, data=np.array(demo_ids, dtype="S"))


def test_integer_demo_sort(tmp_path) -> None:
    _write_robomimic(tmp_path / "demos.hdf5")  # demo_0, demo_10, demo_2 on disk
    episodes = list(Hdf5Ingestor().load(str(tmp_path / "demos.hdf5")))

    assert [int(ep.episode_id.rsplit("/", 1)[1]) for ep in episodes] == [0, 2, 10]


def test_success_derivation(tmp_path) -> None:
    _write_robomimic(tmp_path / "d.hdf5", demos=[("demo_0", 3, True), ("demo_1", 3, False)])
    episodes = {int(ep.episode_id.rsplit("/", 1)[1]): ep for ep in Hdf5Ingestor().load(str(tmp_path / "d.hdf5"))}

    assert episodes[0].success is True
    assert episodes[1].success is False


def test_spine_projection_lands_on_rollout_schema(tmp_path) -> None:
    _write_robomimic(tmp_path / "d.hdf5", demos=[("demo_0", 3, True)])
    episode = next(iter(Hdf5Ingestor().load(str(tmp_path / "d.hdf5"))))

    assert episode.suite == "libero_goal"
    rows: list[dict[str, Any]] = episode.to_step_rows(run_id="test")
    out = write_rows(rows, tmp_path / "ep.parquet")
    assert_emits_schema(out)  # schema parity with ROLLOUT_SCHEMA

    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert row["step_idx"] == i
        assert row["num_steps"] == 3
        assert row["instruction"] == "put the bowl on the plate"
        assert row["bddl_file"] == "/x/put_bowl.bddl"
        assert row["control_mode"] == "relative"
        assert row["source"] == "hdf5"
        assert len(row["state"]) == 8
        assert abs(row["gripper_state"] - 0.08) < 1e-5  # qpos[0]-qpos[1]
        assert abs(row["gripper_action"] - row["action"][-1]) < 1e-6


def test_libero_task_name_and_suite_like_the_real_release(tmp_path) -> None:
    # Real LIBERO files record domain_name="robosuite" (the engine); the suite
    # is only in the bddl path. The stem minus `_demo` is the benchmark task
    # name rollouts record - the demos<->rollouts join key. problem_name (a
    # scene label) stays in metadata.
    path = tmp_path / "pick_up_the_black_bowl_and_place_it_on_the_plate_demo.hdf5"
    _write_robomimic(
        path,
        suite="robosuite",
        bddl="libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl.bddl",
        demos=[("demo_0", 3, True)],
    )
    episode = next(iter(Hdf5Ingestor().load(str(path))))

    assert episode.suite == "libero_spatial"
    assert episode.task_name == "pick_up_the_black_bowl_and_place_it_on_the_plate"
    assert episode.metadata["problem_name"] == "KitchenDemo"


def test_native_libero_obs_aliases(tmp_path) -> None:
    _write_robomimic(tmp_path / "n.hdf5", native=True, demos=[("demo_0", 3, True)])
    episode = next(iter(Hdf5Ingestor().load(str(tmp_path / "n.hdf5"))))
    rows: list[dict[str, Any]] = episode.to_step_rows(run_id="test")

    assert len(rows[0]["state"]) == 8  # ee_ori already axis-angle, no quat conversion
    assert rows[0]["state"][3:6] == [0.0, 0.0, 0.0]
    assert_emits_schema(write_rows(rows, tmp_path / "n.parquet"))


def test_missing_obs_and_rewards_tolerated(tmp_path) -> None:
    _write_robomimic(
        tmp_path / "raw.hdf5",
        with_obs=False,
        with_rewards=False,
        with_dones=False,
        demos=[("demo_0", 4, True)],
    )
    episode = next(iter(Hdf5Ingestor().load(str(tmp_path / "raw.hdf5"))))

    assert episode.success is False
    assert episode.num_steps == 4
    rows = episode.to_step_rows(run_id="test")
    assert rows[0]["state"] is None
    assert rows[0]["eef_pos"] is None
    assert rows[0]["gripper_state"] is None
    assert rows[0]["action"] is not None
    assert_emits_schema(write_rows(rows, tmp_path / "raw.parquet"))


def test_split_mask_and_limit(tmp_path) -> None:
    _write_robomimic(
        tmp_path / "m.hdf5",
        demos=[("demo_0", 3, True), ("demo_1", 3, True), ("demo_2", 3, True)],
        mask={"train": [b"demo_0", b"demo_2"]},
    )

    train = list(Hdf5Ingestor().load(str(tmp_path / "m.hdf5"), split="train"))
    assert [int(ep.episode_id.rsplit("/", 1)[1]) for ep in train] == [0, 2]

    limited = list(Hdf5Ingestor().load(str(tmp_path / "m.hdf5"), limit=1))
    assert len(limited) == 1
