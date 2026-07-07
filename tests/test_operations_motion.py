from __future__ import annotations

import daft
import numpy as np
import pytest

from daft_physical_ai.episodes.schema import empty_step_row, validate_rows
from daft_physical_ai.operations import motion_trim, noop_mask
from daft_physical_ai.operations.motion import trim_span


def actions_from(motion: list[float], gripper: list[float]) -> np.ndarray:
    acts = np.zeros((len(motion), 7))
    acts[:, 0] = motion
    acts[:, -1] = gripper
    return acts


def test_noop_mask_idle_prefix_and_interior() -> None:
    acts = actions_from(motion=[0.0, 0.0, 0.5, 0.0, 0.5, 0.0], gripper=[-1] * 6)

    assert noop_mask(acts).tolist() == [True, True, False, True, False, True]


def test_gripper_toggle_is_not_a_noop() -> None:
    # zero motion but the gripper command changes: load-bearing, keep it
    acts = actions_from(motion=[0.0, 0.0, 0.0], gripper=[-1, 1, 1])

    assert noop_mask(acts).tolist() == [True, False, True]


def test_no_noops_at_all() -> None:
    acts = actions_from(motion=[0.5, 0.5, 0.5], gripper=[-1, -1, -1])
    span = trim_span(acts)

    assert span.frames_removed == 0
    assert span.trim_fraction == 0.0
    assert span.noop_fraction == 0.0


def test_trim_span_reports_prefix_suffix_and_interior() -> None:
    acts = actions_from(motion=[0.0, 0.0, 0.5, 0.0, 0.5, 0.0], gripper=[-1] * 6)
    span = trim_span(acts)

    assert (span.start_step, span.end_step) == (2, 4)
    assert span.frames_removed == 3  # two idle prefix + one idle suffix
    assert span.trim_fraction == pytest.approx(0.5)
    assert span.noop_fraction == pytest.approx(4 / 6)  # interior no-op counted too


def test_all_idle_episode_trims_to_nothing() -> None:
    span = trim_span(actions_from(motion=[0.0, 0.0], gripper=[-1, -1]))

    assert span.frames_removed == 2
    assert span.trim_fraction == 1.0


def make_rows(episode: int, motion: list[float]) -> list[dict[str, object]]:
    rows = []
    for step_idx, value in enumerate(motion):
        row = empty_step_row()
        row.update(
            schema_version="rollout-v1",
            episode_id=f"hdf5/demo/{episode}",
            run_id="test",
            model="libero_demo",
            policy_type="hdf5",
            source="hdf5",
            suite="libero_spatial",
            task_name="put_bowl",
            instruction="put the bowl on the plate",
            success=True,
            num_steps=len(motion),
            step_idx=step_idx,
            action=[value, 0, 0, 0, 0, 0, -1.0],
        )
        rows.append(row)
    return rows


def test_motion_trim_over_step_rows() -> None:
    rows = make_rows(0, [0.0, 0.0, 0.5, 0.5]) + make_rows(1, [0.5, 0.5, 0.5, 0.5])
    df = daft.from_arrow(validate_rows(rows))

    spans = motion_trim(df).sort("episode_id").to_pydict()

    assert spans["episode_id"] == ["hdf5/demo/0", "hdf5/demo/1"]
    assert spans["start_step"] == [2, 0]
    assert spans["frames_removed"] == [2, 0]
