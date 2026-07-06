from __future__ import annotations

import daft
import pytest

from daft_physical_ai.episodes.schema import TERMINAL_FAILURE_LABELS, empty_step_row, validate_rows
from daft_physical_ai.evals import classify_failure, detect_regrasp, label_failures


def test_detect_regrasp_loop() -> None:
    detection = detect_regrasp(
        object_z=[0.0, 0.0, 0.08, 0.11, 0.02, 0.02, 0.09],
        gripper_closed=[False, True, True, True, True, False, True],
    )

    assert detection.label == "re_grasp"
    assert detection.regrasp_count == 1
    assert [event.kind for event in detection.events] == ["grasp", "lift", "drop", "re-grasp", "lift"]


def test_detect_drop_without_recovery() -> None:
    detection = detect_regrasp(
        object_z=[0.0, 0.0, 0.08, 0.02, 0.01],
        gripper_closed=[False, True, True, True, False],
    )

    assert detection.label == "drop_no_recover"
    assert detection.regrasp_count == 0


def test_detect_no_grasp() -> None:
    detection = detect_regrasp(
        object_z=[0.0, 0.01, 0.0],
        gripper_closed=[False, False, False],
    )

    assert detection.label == "no_grasp"


def test_detect_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        detect_regrasp(object_z=[0.0], gripper_closed=[])


HELD = 0.02  # measured finger separation while holding an object
AIR = 0.001  # fingers met: closed on air
OPEN = 0.08


def test_classify_re_grasp_fumble_loop() -> None:
    # two open->close command transitions, holding something both times
    ga = [-1, 1, 1, -1, -1, 1, 1]
    gs = [OPEN, HELD, HELD, OPEN, OPEN, HELD, HELD]
    z = [0.2, 0.2, 0.3, 0.25, 0.2, 0.2, 0.3]
    label, features = classify_failure(ga, gs, z)

    assert label == "re_grasp"
    assert features.close_cycles == 2
    assert features.ever_held


def test_classify_no_grasp_closed_on_air() -> None:
    ga = [-1, 1, 1, 1]
    gs = [OPEN, AIR, AIR, AIR]
    z = [0.2, 0.2, 0.2, 0.2]
    label, features = classify_failure(ga, gs, z)

    assert label == "no_grasp"
    assert not features.ever_held
    assert features.closed_on_air_frac == 1.0


def test_classify_grasp_no_lift() -> None:
    ga = [-1, 1, 1, 1]
    gs = [OPEN, HELD, HELD, HELD]
    z = [0.2, 0.2, 0.205, 0.21]  # holds but never lifts min_lift=0.02
    label, _ = classify_failure(ga, gs, z)

    assert label == "grasp_no_lift"


def test_classify_missed_target_single_clean_lift() -> None:
    ga = [-1, 1, 1, 1]
    gs = [OPEN, HELD, HELD, HELD]
    z = [0.2, 0.2, 0.3, 0.35]
    label, features = classify_failure(ga, gs, z)

    assert label == "missed_target"
    assert features.close_cycles == 1
    assert features.max_lift == pytest.approx(0.15)


def test_classify_timeout_when_held_without_close_transition() -> None:
    # commanded closed from step 0: no open->close transition is ever observed
    ga = [1, 1, 1]
    gs = [HELD, HELD, HELD]
    z = [0.2, 0.3, 0.35]
    label, features = classify_failure(ga, gs, z)

    assert label == "timeout"
    assert features.close_cycles == 0


def test_classify_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        classify_failure([1.0], [0.02, 0.02], [0.2])


def test_grasp_no_lift_in_schema_taxonomy() -> None:
    assert "grasp_no_lift" in TERMINAL_FAILURE_LABELS


def make_failure_rows(
    policy_type: str, init_state_id: int, ga: list[float], gs: list[float], z: list[float]
) -> list[dict[str, object]]:
    rows = []
    for step_idx, (action, state, height) in enumerate(zip(ga, gs, z)):
        row = empty_step_row()
        row.update(
            schema_version="rollout-v1",
            episode_id=f"libero_spatial/0/{init_state_id}/7",
            run_id=f"run-{policy_type}",
            model=f"{policy_type}-7b",
            policy_type=policy_type,
            source="libero",
            suite="libero_spatial",
            task_id=0,
            task_name="put_bowl",
            init_state_id=init_state_id,
            seed=7,
            instruction="put the bowl on the plate",
            success=False,
            num_steps=len(ga),
            step_idx=step_idx,
            gripper_action=action,
            gripper_state=state,
            eef_pos=[0.0, 0.0, height],
        )
        rows.append(row)
    return rows


def test_label_failures_groups_by_attempt_and_labels() -> None:
    # the same spec attempted by two policies must yield two labeled attempts
    rows = make_failure_rows("openvla", 0, [-1, 1, -1, 1], [OPEN, HELD, OPEN, HELD], [0.2, 0.3, 0.2, 0.3])
    rows += make_failure_rows("vla_jepa", 0, [-1, 1, 1, 1], [OPEN, AIR, AIR, AIR], [0.2, 0.2, 0.2, 0.2])
    df = daft.from_arrow(validate_rows(rows))

    labels = label_failures(df).sort("policy_type").to_pydict()

    assert labels["episode_id"] == ["libero_spatial/0/0/7", "libero_spatial/0/0/7"]
    assert labels["terminal_failure"] == ["re_grasp", "no_grasp"]
