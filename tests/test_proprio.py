from __future__ import annotations

from typing import Any

import daft
import pytest
from daft import col

from daft_physical_ai.proprio import is_active, motion_energy, motion_scale


def _frames(states: dict[int, list[list[float]]]) -> daft.DataFrame:
    """Build a frame-level DataFrame from {episode_index: [state per frame]}."""
    rows: dict[str, Any] = {"episode_index": [], "frame_index": [], "state": []}
    for episode, frames in states.items():
        for i, state in enumerate(frames):
            rows["episode_index"].append(episode)
            rows["frame_index"].append(i)
            rows["state"].append(state)
    return daft.from_pydict(rows)


def test_motion_scale_returns_one_positive_value_per_dim() -> None:
    df = _frames({0: [[0.0, 0.0], [0.1, 0.5], [0.2, 1.0], [0.3, 1.5]]})
    scale = motion_scale(df, "state", dims=2)
    assert len(scale) == 2
    assert all(s > 0 for s in scale)
    # dim 1 steps five times further than dim 0, so its scale is larger
    assert scale[1] > scale[0]


def test_motion_scale_measures_step_size_not_step_variation() -> None:
    # Constant velocity: every delta is identical, so their standard deviation
    # is zero even though the arm is plainly moving. RMS reports the real step.
    df = _frames({0: [[float(i)] for i in range(6)]})
    assert motion_scale(df, "state", dims=1)[0] == pytest.approx(1.0)


def test_motion_scale_never_returns_zero_for_a_frozen_dim() -> None:
    df = _frames({0: [[0.0, 1.0], [0.1, 1.0], [0.2, 1.0]]})
    scale = motion_scale(df, "state", dims=2)
    assert scale[1] > 0  # would divide by zero otherwise


def test_motion_energy_is_zero_on_the_first_frame() -> None:
    df = _frames({0: [[0.0], [1.0], [2.0]]})
    out = df.with_column("e", motion_energy(col("state"), dims=1, scale=[1.0])).sort("frame_index")
    assert out.to_pydict()["e"][0] == 0.0


def test_motion_energy_is_zero_while_still_and_positive_while_moving() -> None:
    df = _frames({0: [[0.0], [0.0], [0.0], [5.0], [10.0]]})
    energy = df.with_column("e", motion_energy(col("state"), dims=1, scale=[1.0]))
    e = energy.sort("frame_index").to_pydict()["e"]
    assert e[1] == pytest.approx(0.0)
    assert e[2] == pytest.approx(0.0)
    assert e[3] == pytest.approx(5.0)
    assert e[4] == pytest.approx(5.0)


def test_motion_energy_does_not_diff_across_episode_boundaries() -> None:
    # Episode 1 starts far from where episode 0 ended. Without partitioning, its
    # first frame would show a huge fake spike.
    df = _frames({0: [[0.0], [0.0]], 1: [[100.0], [100.0]]})
    out = df.with_column("e", motion_energy(col("state"), dims=1, scale=[1.0]))
    assert max(out.to_pydict()["e"]) == pytest.approx(0.0)


def test_motion_energy_combines_dims_as_a_euclidean_norm() -> None:
    df = _frames({0: [[0.0, 0.0], [3.0, 4.0]]})
    out = df.with_column("e", motion_energy(col("state"), dims=2, scale=[1.0, 1.0]))
    assert max(out.to_pydict()["e"]) == pytest.approx(5.0)


def test_motion_energy_scale_normalizes_each_dim() -> None:
    # Same physical step in both dims, but dim 1 is scaled down 10x.
    df = _frames({0: [[0.0, 0.0], [1.0, 10.0]]})
    out = df.with_column("e", motion_energy(col("state"), dims=2, scale=[1.0, 10.0]))
    assert max(out.to_pydict()["e"]) == pytest.approx(2.0**0.5)


def test_motion_energy_rejects_a_mismatched_scale() -> None:
    with pytest.raises(ValueError, match="expected dims=3"):
        motion_energy(col("state"), dims=3, scale=[1.0])


def test_is_active_ignores_a_single_frame_spike() -> None:
    df = daft.from_pydict(
        {
            "episode_index": [0] * 6,
            "frame_index": list(range(6)),
            "e": [0.0, 0.0, 9.0, 0.0, 0.0, 0.0],  # one frame over, run of 1
        }
    )
    out = df.with_column("a", is_active(col("e"), threshold=0.1, run=3)).sort("frame_index")
    assert out.to_pydict()["a"] == [False] * 6


def test_is_active_marks_every_frame_of_a_qualifying_run() -> None:
    df = daft.from_pydict(
        {
            "episode_index": [0] * 7,
            "frame_index": list(range(7)),
            "e": [0.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0],
        }
    )
    out = df.with_column("a", is_active(col("e"), threshold=0.1, run=3)).sort("frame_index")
    assert out.to_pydict()["a"] == [False, True, True, True, True, False, False]


def test_is_active_run_of_one_is_a_plain_threshold() -> None:
    df = daft.from_pydict({"episode_index": [0] * 3, "frame_index": [0, 1, 2], "e": [0.0, 9.0, 0.0]})
    out = df.with_column("a", is_active(col("e"), threshold=0.1, run=1)).sort("frame_index")
    assert out.to_pydict()["a"] == [False, True, False]


def test_is_active_runs_do_not_span_episodes() -> None:
    # Two frames over at the end of episode 0, one at the start of episode 1.
    # Neither episode has a run of 3, so nothing is active.
    df = daft.from_pydict(
        {
            "episode_index": [0, 0, 0, 1, 1, 1],
            "frame_index": [0, 1, 2, 0, 1, 2],
            "e": [0.0, 9.0, 9.0, 9.0, 0.0, 0.0],
        }
    )
    out = df.with_column("a", is_active(col("e"), threshold=0.1, run=3))
    assert not any(out.to_pydict()["a"])


def test_is_active_rejects_a_zero_run() -> None:
    with pytest.raises(ValueError, match="run must be >= 1"):
        is_active(col("e"), run=0)
