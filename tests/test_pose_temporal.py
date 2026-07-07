from __future__ import annotations

from typing import cast

import daft
import numpy as np
import pytest
from daft import col, lit

from daft_physical_ai.pose import (
    EpisodeFeatureComputer,
    add_temporal_features,
    state_frame_features,
)


def spatial_features() -> daft.Expression:
    # the @daft.func wrapper returns an Expression at plan-build time, not the
    # dict its body is annotated with
    return cast("daft.Expression", state_frame_features(col("observation.state")))


def make_state(n: int, seed: int) -> np.ndarray:
    """A wandering synthetic 48-D state so every rate is nonzero."""
    rng = np.random.default_rng(seed)
    state = np.zeros((n, 48))
    for base in (0, 24):
        walk = rng.normal(0, 0.02, (n, 3)).cumsum(axis=0)
        state[:, base : base + 3] = [0.3, 1.0, 0.4] + walk  # wrist wanders
        state[:, base + 3 : base + 9] = [1, 0, 0, 0, 1, 0]  # identity rot6d
        tips = rng.normal(0, 0.01, (n, 15)).cumsum(axis=0) + 0.1
        state[:, base + 9 : base + 24] = state[:, base : base + 3].repeat(5, axis=0).reshape(n, 15) + tips
    return state


def frame_table(episodes: dict[int, np.ndarray]) -> daft.DataFrame:
    rows = [
        {"episode_index": episode, "frame_index": i, "observation.state": state[i].tolist()}
        for episode, state in episodes.items()
        for i in range(len(state))
    ]
    return daft.from_pylist(rows)


def test_window_rates_match_numpy_computer_per_episode() -> None:
    episodes = {0: make_state(8, seed=0), 1: make_state(5, seed=1)}
    frames = frame_table(episodes).select("episode_index", "frame_index", spatial_features())

    in_dag = (
        add_temporal_features(frames, fps=30.0)
        .sort(["episode_index", "frame_index"])
        .select("episode_index", "curl_rate_L", "wrist_vert_vel_R", "wrist_speed_L")
        .to_pydict()
    )

    computer = EpisodeFeatureComputer()
    for episode, state in episodes.items():
        tracks = computer.compute(state=state)
        idx = [i for i, e in enumerate(in_dag["episode_index"]) if e == episode]
        np.testing.assert_allclose(
            [in_dag["curl_rate_L"][i] for i in idx], np.asarray(tracks["curl_rate_L"]), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            [in_dag["wrist_vert_vel_R"][i] for i in idx],
            np.asarray(tracks["wrist_vert_vel_R"]),
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [in_dag["wrist_speed_L"][i] for i in idx], np.asarray(tracks["wrist_speed_L"]), rtol=1e-4, atol=1e-6
        )


def test_rates_do_not_bleed_across_episodes() -> None:
    # each episode's last frame must have rate 0: lead(1) is null at the boundary
    episodes = {0: make_state(4, seed=2), 1: make_state(4, seed=3)}
    frames = frame_table(episodes).select("episode_index", "frame_index", spatial_features())

    data = (
        add_temporal_features(frames)
        .sort(["episode_index", "frame_index"])
        .select("episode_index", "frame_index", "curl_rate_L", "wrist_speed_R")
        .to_pydict()
    )

    for i, (episode, frame) in enumerate(zip(data["episode_index"], data["frame_index"])):
        if frame == 3:  # last frame of each 4-frame episode
            assert data["curl_rate_L"][i] == 0.0, (episode, frame)
            assert data["wrist_speed_R"][i] == 0.0, (episode, frame)


def test_state_only_tables_get_state_only_rates() -> None:
    frames = frame_table({0: make_state(4, seed=4)}).select("episode_index", "frame_index", spatial_features())
    out = add_temporal_features(frames)

    names = set(out.column_names)
    assert {"curl_rate_L", "wrist_vert_vel_L", "wrist_speed_L"} <= names
    assert "arm_ext_rate_L" not in names  # skeleton-derived inputs absent
    assert "roll_L" not in names


def test_scenario_predicates_as_column_expressions() -> None:
    # grasping/lifting thresholds work as plain expressions over the rate columns
    state = make_state(6, seed=5)
    state[3, 1] = state[2, 1] + 0.05  # force a right... left wrist_y jump on frame 2->3
    frames = frame_table({0: state}).select("episode_index", "frame_index", spatial_features())

    lifted = (
        add_temporal_features(frames, fps=30.0)
        .where(col("wrist_vert_vel_L") >= lit(0.20))
        .select("frame_index")
        .to_pydict()
    )

    assert 2 in lifted["frame_index"]


def test_forearm_roll_requires_both_rotation_columns() -> None:
    frames = frame_table({0: make_state(3, seed=6)}).select("episode_index", "frame_index", spatial_features())
    with_axis = frames.with_column(
        "forearm_axis_L",
        col("wrist_rot6d_L").apply(lambda r: [1.0, 0.0, 0.0], return_dtype=daft.DataType.list(daft.DataType.float64())),
    )

    out = add_temporal_features(with_axis)
    assert "roll_L" in set(out.column_names)
    rolls = out.sort("frame_index").select("roll_L").to_pydict()["roll_L"]
    assert rolls == pytest.approx([0.0, 0.0, 0.0])  # identity rotation throughout: no roll
