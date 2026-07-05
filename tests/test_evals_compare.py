from __future__ import annotations

import daft

from daft_physical_ai.episodes.schema import empty_step_row, validate_rows
from daft_physical_ai.evals import (
    compare_policies,
    episode_outcomes,
    failure_counts,
    success_rates,
)


def make_rows(
    policy_type: str,
    spec_success: list[bool],
    *,
    terminal_failure: str = "re_grasp",
    steps: int = 2,
) -> list[dict[str, object]]:
    """Step rows for one policy attempting spec ids 0..n on one task."""
    rows: list[dict[str, object]] = []
    for init_state_id, success in enumerate(spec_success):
        for step_idx in range(steps):
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
                success=success,
                terminal_failure=None if success else terminal_failure,
                num_steps=steps,
                step_idx=step_idx,
            )
            rows.append(row)
    return rows


def make_two_policy_frame() -> daft.DataFrame:
    rows = make_rows("openvla", [True, False, False, False])
    rows += make_rows("vla_jepa", [True, True, True, False], terminal_failure="drop_no_recover")
    return daft.from_arrow(validate_rows(rows))


def test_episode_outcomes_groups_by_policy_not_episode_id() -> None:
    # Both policies run the same 4 specs; grouping must keep 8 attempts, not
    # chimera them into 4 phantom episodes.
    outcomes = episode_outcomes(make_two_policy_frame())
    assert outcomes.count_rows() == 8


def test_success_rates_per_policy() -> None:
    rates = success_rates(make_two_policy_frame()).sort("policy_type").to_pydict()
    assert rates["policy_type"] == ["openvla", "vla_jepa"]
    assert rates["success_rate"] == [0.25, 0.75]
    assert rates["episodes"] == [4, 4]


def test_failure_counts_labels_per_policy() -> None:
    counts = failure_counts(make_two_policy_frame()).sort("policy_type").to_pydict()
    by_policy = dict(zip(counts["policy_type"], zip(counts["terminal_failure"], counts["episodes"])))
    assert by_policy["openvla"] == ("re_grasp", 3)
    assert by_policy["vla_jepa"] == ("drop_no_recover", 1)


def test_compare_policies_pairs_specs() -> None:
    paired = compare_policies(make_two_policy_frame(), "openvla", "vla_jepa")
    data = paired.sort("episode_id").to_pydict()

    assert len(data["episode_id"]) == 4
    assert data["success_left"] == [True, False, False, False]
    assert data["success_right"] == [True, True, True, False]

    left_only_failures = [
        episode_id
        for episode_id, left, right in zip(data["episode_id"], data["success_left"], data["success_right"])
        if right and not left
    ]
    assert left_only_failures == ["libero_spatial/0/1/7", "libero_spatial/0/2/7"]
