"""The canonical LIBERO evaluation protocol, as checkable constants.

The published VLA eval protocol originates in OpenVLA's ``run_libero_eval.py``
and is inherited verbatim by openpi, starVLA, and allenai's evaluation harness:
50 trials per task, seed 7, 10 zero-action settle steps, per-suite step caps.
``validate_run`` turns "we reproduced the benchmark" from a vibe into a check a
reviewer can run against the rollout parquet itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import daft
from daft import col, lit

#: Trials per task in the canonical protocol (50 init-state layouts per task).
TRIALS_PER_TASK = 50
#: The canonical eval seed.
PROTOCOL_SEED = 7
#: Zero-action settle steps before the policy takes control (objects drop in).
NUM_STEPS_WAIT = 10

#: The four suites in the standard published VLA eval; ``libero_10`` is LIBERO-Long.
CORE_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

#: Published per-suite step caps. Where published caps differ across policies
#: (OpenVLA uses 220 for spatial, VLA-JEPA 250) the non-truncating cap is used
#: so cross-policy comparison never clips one side.
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 250,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

SUITE_NUM_TASKS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


@dataclass(frozen=True)
class ProtocolIssue:
    """One way a run deviates from the protocol."""

    code: str
    message: str


@dataclass(frozen=True)
class ProtocolReport:
    """Outcome of ``validate_run`` for one (suite, policy) scope."""

    suite: str
    policy_type: str
    n_tasks: int
    n_attempts: int
    expected_attempts: int | None
    issues: tuple[ProtocolIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def _preview(values: Sequence[object], limit: int = 5) -> str:
    shown = ", ".join(str(value) for value in values[:limit])
    return shown if len(values) <= limit else f"{shown}, ... ({len(values)} total)"


def validate_run(
    df: daft.DataFrame,
    *,
    suite: str,
    policy_type: str,
    trials_per_task: int = TRIALS_PER_TASK,
    seed: int = PROTOCOL_SEED,
) -> ProtocolReport:
    """Check that step rows for one (suite, policy) reproduce the protocol.

    Verifies task coverage, trials per task, the eval seed, per-suite step
    caps, and step-count consistency. A doubled parquet part (the same episode
    written twice) surfaces as a step-count mismatch: the duplicate rows repeat
    ``step_idx`` values, so row count exceeds ``max(step_idx) + 1``.
    """
    scoped = df.where((col("suite") == lit(suite)) & (col("policy_type") == lit(policy_type)))
    attempts = scoped.groupby("model", "episode_id").agg(
        col("task_id").any_value(),
        col("init_state_id").any_value(),
        col("seed").any_value(),
        col("num_steps").any_value(),
        col("step_idx").max().alias("max_step_idx"),
        col("step_idx").count().alias("n_rows"),
    )
    data = attempts.to_pydict()
    episode_ids: list[str] = data["episode_id"]

    issues: list[ProtocolIssue] = []
    if not episode_ids:
        issues.append(ProtocolIssue("empty_run", f"no rows for suite={suite!r} policy_type={policy_type!r}"))
        return ProtocolReport(
            suite=suite,
            policy_type=policy_type,
            n_tasks=0,
            n_attempts=0,
            expected_attempts=None,
            issues=tuple(issues),
        )

    expected_tasks = SUITE_NUM_TASKS.get(suite)
    max_steps = SUITE_MAX_STEPS.get(suite)
    if expected_tasks is None:
        issues.append(ProtocolIssue("unknown_suite", f"{suite!r} has no published task count or step cap"))

    observed_tasks = sorted({task for task in data["task_id"] if task is not None})
    if expected_tasks is not None:
        missing = sorted(set(range(expected_tasks)) - set(observed_tasks))
        if missing:
            issues.append(ProtocolIssue("missing_tasks", f"tasks never attempted: {_preview(missing)}"))

    trial_counts: dict[int, int] = {}
    for task in data["task_id"]:
        if task is not None:
            trial_counts[task] = trial_counts.get(task, 0) + 1
    off_protocol = sorted(
        f"task {task}: {count}/{trials_per_task}" for task, count in trial_counts.items() if count != trials_per_task
    )
    if off_protocol:
        issues.append(ProtocolIssue("trial_count", f"trials per task off: {_preview(off_protocol)}"))

    bad_seeds = sorted({s for s in data["seed"] if s != seed})
    if bad_seeds:
        issues.append(ProtocolIssue("seed_mismatch", f"expected seed {seed}, found {_preview(bad_seeds)}"))

    if max_steps is not None:
        over_cap = [episode_id for episode_id, n_rows in zip(episode_ids, data["n_rows"]) if n_rows > max_steps]
        if over_cap:
            issues.append(
                ProtocolIssue(
                    "step_cap_exceeded",
                    f"episodes exceed the {max_steps}-step {suite} cap: {_preview(over_cap)}",
                )
            )

    inconsistent = [
        episode_id
        for episode_id, num_steps, max_step_idx, n_rows in zip(
            episode_ids, data["num_steps"], data["max_step_idx"], data["n_rows"]
        )
        if n_rows != max_step_idx + 1 or (num_steps is not None and num_steps != n_rows)
    ]
    if inconsistent:
        issues.append(
            ProtocolIssue(
                "step_count_mismatch",
                f"row count disagrees with step_idx/num_steps (truncated or doubled part): {_preview(inconsistent)}",
            )
        )

    return ProtocolReport(
        suite=suite,
        policy_type=policy_type,
        n_tasks=len(observed_tasks),
        n_attempts=len(episode_ids),
        expected_attempts=None if expected_tasks is None else expected_tasks * trials_per_task,
        issues=tuple(issues),
    )
