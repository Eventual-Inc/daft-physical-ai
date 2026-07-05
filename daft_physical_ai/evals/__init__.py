"""Policy evaluation and benchmark reproduction over canonical episode tables.

This package is the analysis half of the eval loop. Rollout parquet in the
`daft_physical_ai.episodes` step-row schema comes in; policy-level answers come
out: success rates, per-spec policy comparisons, failure labels, and checks
that a run faithfully reproduces the published benchmark protocol.

Rollout *generation* (simulators, policy checkpoints, GPU images) stays outside
the package and lands here as parquet. The integration contract is the schema:
`episode_id` names the evaluation spec (``suite/task_id/init_state_id/seed``),
so any harness that writes the schema gets this analysis layer for free, and
the same spec run by two policies joins into a paired comparison.

Roadmap: the LIBERO rollout runner and its ``Policy`` seam are planned to
promote into a ``daft-physical-ai[libero]`` extra - LIBERO ships as a plain
wheel now and co-resolves with modern policy stacks in one Python >=3.12
process - once the lerobot release that carries the policy port is published
and can be pinned in package metadata.
"""

from __future__ import annotations

from .compare import (
    ATTEMPT_KEYS,
    compare_policies,
    episode_outcomes,
    failure_counts,
    success_rates,
)
from .failures import FailureEvent, FailureLabel, RegraspDetection, detect_regrasp
from .protocol import (
    CORE_SUITES,
    NUM_STEPS_WAIT,
    PROTOCOL_SEED,
    SUITE_MAX_STEPS,
    SUITE_NUM_TASKS,
    TRIALS_PER_TASK,
    ProtocolIssue,
    ProtocolReport,
    validate_run,
)

__all__ = [
    "ATTEMPT_KEYS",
    "CORE_SUITES",
    "NUM_STEPS_WAIT",
    "PROTOCOL_SEED",
    "SUITE_MAX_STEPS",
    "SUITE_NUM_TASKS",
    "TRIALS_PER_TASK",
    "FailureEvent",
    "FailureLabel",
    "ProtocolIssue",
    "ProtocolReport",
    "RegraspDetection",
    "compare_policies",
    "detect_regrasp",
    "episode_outcomes",
    "failure_counts",
    "success_rates",
    "validate_run",
]
