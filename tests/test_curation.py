from __future__ import annotations

import daft

from daft_physical_ai.curation import acquisition_map, preference_pairs, sft_view
from daft_physical_ai.episodes.schema import empty_step_row, validate_rows
from daft_physical_ai.evals import label_failures
from daft_physical_ai.operations import motion_trim


def make_rows(
    policy_type: str,
    init_state_id: int,
    *,
    task_id: int = 0,
    success: bool = True,
    motion: list[float] | None = None,
    gripper_states: list[float] | None = None,
) -> list[dict[str, object]]:
    motion = motion if motion is not None else [0.5, 0.5, 0.5, 0.5]
    rows = []
    for step_idx, value in enumerate(motion):
        row = empty_step_row()
        row.update(
            schema_version="rollout-v1",
            episode_id=f"libero_spatial/{task_id}/{init_state_id}/7",
            run_id=f"run-{policy_type}",
            model=f"{policy_type}-7b",
            policy_type=policy_type,
            source="libero",
            suite="libero_spatial",
            task_id=task_id,
            task_name=f"task_{task_id}",
            init_state_id=init_state_id,
            seed=7,
            instruction="put the bowl on the plate",
            success=success,
            num_steps=len(motion),
            step_idx=step_idx,
            action=[value, 0, 0, 0, 0, 0, 1.0],
            gripper_action=1.0,
            gripper_state=(gripper_states or [0.02] * len(motion))[step_idx],
            eef_pos=[0.0, 0.0, 0.2],
        )
        rows.append(row)
    return rows


def test_sft_view_filters_failures_and_exclusions() -> None:
    rows = make_rows("hdf5", 0, success=True)
    rows += make_rows("hdf5", 1, success=False)
    rows += make_rows("hdf5", 2, success=True)
    df = daft.from_arrow(validate_rows(rows))

    view = sft_view(df, exclude_episode_ids=["libero_spatial/0/2/7"]).to_pydict()

    assert set(view["episode_id"]) == {"libero_spatial/0/0/7"}
    assert set(view["curation_weight"]) == {1.0}


def test_sft_view_applies_trim_spans() -> None:
    # two idle prefix steps then motion: the trimmed view drops step 0-1
    rows = make_rows("hdf5", 0, motion=[0.0, 0.0, 0.5, 0.5])
    df = daft.from_arrow(validate_rows(rows))
    spans = motion_trim(df)

    view = sft_view(df, trim_spans=spans).sort("step_idx").to_pydict()

    assert view["step_idx"] == [2, 3]


def test_preference_pairs_covers_both_directions() -> None:
    rows = make_rows("openvla", 0, success=False) + make_rows("vla_jepa", 0, success=True)
    rows += make_rows("openvla", 1, success=True) + make_rows("vla_jepa", 1, success=False)
    rows += make_rows("openvla", 2, success=True) + make_rows("vla_jepa", 2, success=True)
    df = daft.from_arrow(validate_rows(rows))

    pairs = preference_pairs(df, "openvla", "vla_jepa").sort("episode_id").to_pydict()

    assert pairs["episode_id"] == ["libero_spatial/0/0/7", "libero_spatial/0/1/7"]
    assert pairs["chosen_policy"] == ["vla_jepa", "openvla"]
    assert pairs["rejected_policy"] == ["openvla", "vla_jepa"]


def test_acquisition_map_ranks_the_failing_task_first() -> None:
    air = [0.001] * 4  # closed on air -> no_grasp
    rows: list[dict[str, object]] = []
    for init_state_id in range(3):  # task 5 fails three times
        rows += make_rows("openvla", init_state_id, task_id=5, success=False, gripper_states=air)
    rows += make_rows("openvla", 0, task_id=1, success=False, gripper_states=air)
    rows += make_rows("openvla", 1, task_id=1, success=True)
    df = daft.from_arrow(validate_rows(rows))

    ranked = acquisition_map(label_failures(df)).to_pydict()

    assert ranked["task_id"][0] == 5
    assert ranked["failures"][0] == 3
    assert ranked["init_state_ids"][0] == [0, 1, 2]
    assert ranked["terminal_failure"][0] == "no_grasp"
