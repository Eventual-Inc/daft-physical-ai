"""Cross-policy comparison recipes over canonical step-row tables.

Contract: ``episode_id`` names the evaluation *spec*
(``suite/task_id/init_state_id/seed``), not the attempt. Two policies
legitimately run the same ``episode_id``, so every aggregation here groups by
``ATTEMPT_KEYS`` - grouping by ``episode_id`` alone would chimera different
policies' attempts into one phantom episode.

Every function takes and returns a lazy Daft DataFrame, so these compose with
any filter/join before ``.collect()``.
"""

from __future__ import annotations

from collections.abc import Sequence

import daft
from daft import col, lit

#: One attempt = one policy running one evaluation spec.
ATTEMPT_KEYS: tuple[str, ...] = ("policy_type", "model", "episode_id")


def episode_outcomes(df: daft.DataFrame) -> daft.DataFrame:
    """Collapse step rows to one row per attempt.

    Episode-level fields are denormalized onto every step row, so a single
    ``any_value`` per group recovers them without a join.
    """
    return df.groupby(*ATTEMPT_KEYS).agg(
        col("suite").any_value(),
        col("task_id").any_value(),
        col("task_name").any_value(),
        col("success").any_value(),
        col("terminal_failure").any_value(),
        col("num_steps").any_value(),
    )


def success_rates(df: daft.DataFrame, by: Sequence[str] = ("policy_type",)) -> daft.DataFrame:
    """Success rate and attempt count per group (default: per policy)."""
    outcomes = episode_outcomes(df)
    return outcomes.groupby(*by).agg(
        col("success").cast(daft.DataType.float64()).mean().alias("success_rate"),
        col("episode_id").count().alias("episodes"),
    )


def failure_counts(df: daft.DataFrame, by: Sequence[str] = ("policy_type",)) -> daft.DataFrame:
    """Failed-attempt counts per terminal-failure label per group."""
    failures = episode_outcomes(df).where(~col("success"))
    return failures.groupby(*by, "terminal_failure").agg(
        col("episode_id").count().alias("episodes"),
    )


def compare_policies(df: daft.DataFrame, left: str, right: str) -> daft.DataFrame:
    """Pair two policies' attempts on the same evaluation specs.

    Returns one row per ``episode_id`` both policies attempted, with
    ``success_left``/``success_right`` (and failure/step columns) side by side -
    the per-spec view behind "where does `left` fail that `right` does not".

    Assumes one model per policy type in ``df``; with several, filter on
    ``model`` first or the spec join fans out.
    """
    outcomes = episode_outcomes(df)
    left_outcomes = outcomes.where(col("policy_type") == lit(left)).select(
        "episode_id",
        "suite",
        "task_id",
        "task_name",
        col("success").alias("success_left"),
        col("terminal_failure").alias("terminal_failure_left"),
        col("num_steps").alias("num_steps_left"),
    )
    right_outcomes = outcomes.where(col("policy_type") == lit(right)).select(
        "episode_id",
        col("success").alias("success_right"),
        col("terminal_failure").alias("terminal_failure_right"),
        col("num_steps").alias("num_steps_right"),
    )
    return left_outcomes.join(right_outcomes, on="episode_id", how="inner")
