from __future__ import annotations

import daft

from daft_physical_ai.episodes.schema import empty_step_row, validate_rows
from daft_physical_ai.evals import validate_run


def make_run_rows(
    *,
    policy_type: str = "openvla",
    suite: str = "libero_spatial",
    tasks: int = 10,
    trials: int = 2,
    seed: int = 7,
    steps: int = 3,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id in range(tasks):
        for init_state_id in range(trials):
            for step_idx in range(steps):
                row = empty_step_row()
                row.update(
                    schema_version="rollout-v1",
                    episode_id=f"{suite}/{task_id}/{init_state_id}/{seed}",
                    run_id="run-protocol",
                    model=f"{policy_type}-7b",
                    policy_type=policy_type,
                    source="libero",
                    suite=suite,
                    task_id=task_id,
                    init_state_id=init_state_id,
                    seed=seed,
                    instruction="put the bowl on the plate",
                    success=init_state_id % 2 == 0,
                    num_steps=steps,
                    step_idx=step_idx,
                )
                rows.append(row)
    return rows


def to_frame(rows: list[dict[str, object]]) -> daft.DataFrame:
    return daft.from_arrow(validate_rows(rows))


def test_faithful_run_passes() -> None:
    report = validate_run(
        to_frame(make_run_rows()),
        suite="libero_spatial",
        policy_type="openvla",
        trials_per_task=2,
    )

    assert report.ok
    assert report.n_tasks == 10
    assert report.n_attempts == 20
    assert report.expected_attempts == 20  # 10 tasks x the trials_per_task under validation


def test_scope_filters_other_policies_and_suites() -> None:
    rows = make_run_rows() + make_run_rows(policy_type="vla_jepa", seed=11, trials=1)
    report = validate_run(to_frame(rows), suite="libero_spatial", policy_type="openvla", trials_per_task=2)

    assert report.ok  # vla_jepa's off-seed short run must not leak into openvla's report


def test_missing_task_and_trial_count_flagged() -> None:
    rows = [row for row in make_run_rows() if row["task_id"] != 9]
    rows = [row for row in rows if not (row["task_id"] == 0 and row["init_state_id"] == 1)]
    report = validate_run(to_frame(rows), suite="libero_spatial", policy_type="openvla", trials_per_task=2)

    codes = {issue.code for issue in report.issues}
    assert codes == {"missing_tasks", "trial_count"}


def test_wrong_seed_flagged() -> None:
    report = validate_run(
        to_frame(make_run_rows(seed=42)),
        suite="libero_spatial",
        policy_type="openvla",
        trials_per_task=2,
    )

    assert [issue.code for issue in report.issues] == ["seed_mismatch"]


def test_step_cap_exceeded_flagged() -> None:
    rows = make_run_rows(steps=251)  # spatial cap is 250
    report = validate_run(to_frame(rows), suite="libero_spatial", policy_type="openvla", trials_per_task=2)

    codes = [issue.code for issue in report.issues]
    assert codes == ["step_cap_exceeded"]


def test_doubled_parquet_part_flagged_as_step_count_mismatch() -> None:
    rows = make_run_rows()
    report = validate_run(
        to_frame(rows + rows),  # every episode written twice
        suite="libero_spatial",
        policy_type="openvla",
        trials_per_task=2,
    )

    assert [issue.code for issue in report.issues] == ["step_count_mismatch"]


def test_empty_scope_reports_empty_run() -> None:
    report = validate_run(to_frame(make_run_rows()), suite="libero_goal", policy_type="openvla")

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["empty_run"]
