"""Curation: turn analyzed episode tables into training-set views.

The bridge between evaluation and training. Each function is a thin, lazy
combinator over the canonical step-row schema, and each maps to one way a
training regime consumes graded episodes:

- `sft_view` - the imitation-learning path: only behavior worth imitating
  (successes, quality-gated, idle steps trimmed). Curated datasets are views:
  the output selects and annotates rows, media stays wherever the source
  dataset keeps it.
- `preference_pairs` - the contrastive path: the same evaluation spec attempted
  by two policies with opposite outcomes is a ready-made (chosen, rejected)
  pair, because ``episode_id`` names the spec, not the attempt.
- `acquisition_map` - the collection path: failures ranked by where they
  concentrate, i.e. which (task, init states) to aim the next teleop or
  generation budget at.

Demonstrations and rollouts join on ``(suite, task_name)`` - never ``task_id``
(null on demos) or ``episode_id`` (different id spaces).
"""

from __future__ import annotations

from collections.abc import Sequence

import daft
from daft import col, lit

from .evals.compare import compare_policies

#: Columns that identify one attempt in trim-span/label sidecars.
_SPAN_KEYS = ("episode_id", "policy_type", "model")


def sft_view(
    df: daft.DataFrame,
    *,
    require_success: bool = True,
    exclude_episode_ids: Sequence[str] = (),
    trim_spans: daft.DataFrame | None = None,
    weight: float = 1.0,
) -> daft.DataFrame:
    """Step rows worth imitating, as a lazy view with provenance columns.

    Filters to successful episodes (behavior cloning must not see failures),
    drops explicitly excluded episodes (e.g. fumbly successes flagged by a
    quality pass), and - when ``trim_spans`` from
    `daft_physical_ai.operations.motion_trim` is given - keeps only each
    episode's active ``[start_step, end_step]`` window. Adds
    ``curation_weight`` (default 1.0) for weighted regimes.
    """
    view = df
    if require_success:
        view = view.where(col("success"))
    if exclude_episode_ids:
        view = view.where(~col("episode_id").is_in(list(exclude_episode_ids)))
    if trim_spans is not None:
        windows = trim_spans.select(
            *_SPAN_KEYS,
            col("start_step").alias("_trim_start"),
            col("end_step").alias("_trim_end"),
        )
        view = (
            view.join(windows, on=list(_SPAN_KEYS), how="inner")
            .where((col("step_idx") >= col("_trim_start")) & (col("step_idx") <= col("_trim_end")))
            .exclude("_trim_start", "_trim_end")
        )
    return view.with_column("curation_weight", lit(weight))


def preference_pairs(df: daft.DataFrame, left: str, right: str) -> daft.DataFrame:
    """One (chosen, rejected) row per spec the two policies disagree on.

    Built on `daft_physical_ai.evals.compare_policies`: both policies attempted
    the same ``episode_id`` (the spec), one succeeded, one failed. The output is
    an episode-level manifest - resolve it back against the step rows (join on
    ``episode_id`` + the policy column) to materialize trajectories.
    """
    paired = compare_policies(df, left, right)
    spec_cols = (col("episode_id"), col("suite"), col("task_id"), col("task_name"))
    right_wins = paired.where(col("success_right") & ~col("success_left")).select(
        *spec_cols,
        lit(right).alias("chosen_policy"),
        lit(left).alias("rejected_policy"),
        col("num_steps_right").alias("chosen_num_steps"),
        col("num_steps_left").alias("rejected_num_steps"),
    )
    left_wins = paired.where(col("success_left") & ~col("success_right")).select(
        *spec_cols,
        lit(left).alias("chosen_policy"),
        lit(right).alias("rejected_policy"),
        col("num_steps_left").alias("chosen_num_steps"),
        col("num_steps_right").alias("rejected_num_steps"),
    )
    return right_wins.concat(left_wins)


def acquisition_map(labels: daft.DataFrame) -> daft.DataFrame:
    """Rank where failures concentrate: the "collect these next" table.

    Takes the output of `daft_physical_ai.evals.label_failures` and returns one
    row per (policy, suite, task, failure label) with the failure count and the
    init states it happened on, ranked by count. This is the acquisition
    function for the next data-collection or generation pass: point demos at
    the top rows.
    """
    return (
        labels.groupby("policy_type", "suite", "task_id", "task_name", "terminal_failure")
        .agg(
            col("episode_id").count().alias("failures"),
            col("init_state_id").list_agg().alias("init_state_ids"),
        )
        .with_column("init_state_ids", col("init_state_ids").list_sort())
        .sort("failures", desc=True)
    )
