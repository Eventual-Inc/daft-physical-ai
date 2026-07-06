"""Motion trimming: find and remove the idle steps in an episode.

Raw teleop demonstrations carry no-op steps - the operator settling in before
moving, pausing mid-task, or idling after release. Training on them teaches
idling; the community's fix is dataset-level cleaning (OpenVLA's LIBERO
fine-tunes train on a hand-built "no-noops" RLDS variant). ``noop_mask`` is
that cleaning rule as a function, and ``motion_trim`` applies it across a
step-row table in one Daft groupby.

A step is a no-op when its commanded motion is negligible (``||action[:-1]|| <
atol``) AND the gripper command does not change vs the previous step - a pause
that toggles the gripper is load-bearing, not idle.
"""

from __future__ import annotations

from dataclasses import dataclass

import daft
import numpy as np
from daft import col

from ..evals.compare import ATTEMPT_KEYS


@dataclass(frozen=True)
class TrimSpan:
    """The kept [start_step, end_step] window of one episode after trimming."""

    start_step: int
    end_step: int
    num_steps: int
    frames_removed: int
    trim_fraction: float
    noop_fraction: float


def noop_mask(actions: np.ndarray, *, atol: float = 1e-3) -> np.ndarray:
    """Boolean mask of no-op steps for an (N, action_dim) array.

    The last action dimension is the gripper command; the rest is motion.
    """
    acts = np.asarray(actions, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[0] == 0:
        raise ValueError("actions must be a non-empty (N, action_dim) array")
    still = np.linalg.norm(acts[:, :-1], axis=1) < atol
    gripper = acts[:, -1]
    gripper_held = np.concatenate([[True], gripper[1:] == gripper[:-1]])
    return still & gripper_held


def trim_span(actions: np.ndarray, *, atol: float = 1e-3) -> TrimSpan:
    """Compute the kept window and no-op statistics for one episode."""
    mask = noop_mask(actions, atol=atol)
    n = len(mask)
    active = np.flatnonzero(~mask)
    if len(active) == 0:  # an all-idle episode trims to nothing
        return TrimSpan(0, -1, n, n, 1.0, 1.0)
    start, end = int(active[0]), int(active[-1])
    kept = end - start + 1
    return TrimSpan(
        start_step=start,
        end_step=end,
        num_steps=n,
        frames_removed=n - kept,
        trim_fraction=(n - kept) / n,
        noop_fraction=float(mask.sum() / n),
    )


def motion_trim(df: daft.DataFrame, *, atol: float = 1e-3) -> daft.DataFrame:
    """Per-episode trim spans for a step-row table.

    Returns one row per attempt (grouped by ``ATTEMPT_KEYS``) with the kept
    ``[start_step, end_step]`` window, ``frames_removed`` / ``trim_fraction``
    (idle prefix + suffix), and ``noop_fraction`` (all idle steps, interior
    included - the number the RLDS "no-noops" variants filter on). Requires the
    ``action`` column to be populated.
    """
    grouped = df.groupby(*ATTEMPT_KEYS).agg(
        col("suite").any_value(),
        col("task_name").any_value(),
        col("step_idx").list_agg().alias("step_idxs"),
        col("action").list_agg().alias("actions"),
    )
    data = grouped.to_pydict()

    rows: list[dict[str, object]] = []
    for i, episode_id in enumerate(data["episode_id"]):
        if any(action is None for action in data["actions"][i]):
            raise ValueError(f"{episode_id}: null actions; cannot compute motion trim")
        order = np.argsort(np.asarray(data["step_idxs"][i]))
        actions = np.asarray(data["actions"][i], dtype=np.float64)[order]
        span = trim_span(actions, atol=atol)
        rows.append(
            {
                "episode_id": episode_id,
                "policy_type": data["policy_type"][i],
                "model": data["model"][i],
                "suite": data["suite"][i],
                "task_name": data["task_name"][i],
                "start_step": span.start_step,
                "end_step": span.end_step,
                "num_steps": span.num_steps,
                "frames_removed": span.frames_removed,
                "trim_fraction": span.trim_fraction,
                "noop_fraction": span.noop_fraction,
            }
        )
    return daft.from_pylist(rows)
