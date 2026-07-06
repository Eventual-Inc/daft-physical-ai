"""Failure labeling from per-step episode signals.

Two labelers share the vocabulary of
`daft_physical_ai.episodes.schema.TERMINAL_FAILURE_LABELS`:

- `classify_failure` / `label_failures` - the production path for real
  rollouts. Keys off gripper command/state and end-effector height, which
  rollout writers populate on every step; works when ``object_poses`` is null.
- `detect_regrasp` - the object-pose path. Needs object height, so it applies
  to episodes that carry ``object_poses`` (synthetic demos, sim datasets with
  object state).

Neither needs a simulator or model in the loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import daft
import numpy as np
from daft import col

from .compare import ATTEMPT_KEYS

FailureLabel = Literal["re_grasp", "drop_no_recover", "missed_target", "no_grasp"]
RolloutFailureLabel = Literal["re_grasp", "no_grasp", "grasp_no_lift", "missed_target", "timeout"]


@dataclass(frozen=True)
class FailureEvent:
    step_idx: int
    kind: Literal["grasp", "lift", "drop", "re-grasp"]


@dataclass(frozen=True)
class RegraspDetection:
    label: FailureLabel
    events: tuple[FailureEvent, ...]
    regrasp_count: int


def detect_regrasp(
    object_z: Sequence[float],
    gripper_closed: Sequence[bool],
    *,
    lift_threshold: float = 0.05,
    drop_threshold: float = 0.03,
) -> RegraspDetection:
    """Classify a trajectory from object height and gripper state.

    The detector looks for this timeline:

    1. gripper closes,
    2. object rises above ``lift_threshold``,
    3. lifted object falls below ``drop_threshold``,
    4. gripper closes again after the drop.

    Any post-drop close is labeled ``re_grasp``. A lift/drop without a post-drop
    close is ``drop_no_recover``. A close without a drop is ``missed_target``.
    No close is ``no_grasp``.
    """
    if len(object_z) != len(gripper_closed):
        raise ValueError("object_z and gripper_closed must have the same length")

    events: list[FailureEvent] = []
    lifted = False
    drops = 0
    regrasps = 0
    was_closed = False

    for step_idx, (z_value, is_closed) in enumerate(zip(object_z, gripper_closed)):
        z = float(z_value)
        closed = bool(is_closed)

        if closed and not was_closed:
            events.append(FailureEvent(step_idx, "re-grasp" if drops > 0 else "grasp"))
            if drops > 0:
                regrasps += 1
        if z > lift_threshold and not lifted:
            lifted = True
            events.append(FailureEvent(step_idx, "lift"))
        if lifted and z < drop_threshold:
            lifted = False
            drops += 1
            events.append(FailureEvent(step_idx, "drop"))
        was_closed = closed

    if regrasps >= 1:
        label: FailureLabel = "re_grasp"
    elif drops >= 1:
        label = "drop_no_recover"
    elif any(bool(value) for value in gripper_closed):
        label = "missed_target"
    else:
        label = "no_grasp"

    return RegraspDetection(label=label, events=tuple(events), regrasp_count=regrasps)


@dataclass(frozen=True)
class FailureFeatures:
    """Per-episode gripper/lift statistics behind a `classify_failure` label."""

    steps: int
    close_cycles: int
    held_frac: float
    ever_held: bool
    max_lift: float
    closed_on_air_frac: float


def classify_failure(
    gripper_action: Sequence[float],
    gripper_state: Sequence[float],
    eef_z: Sequence[float],
    *,
    hold_mm: float = 0.004,
    air_mm: float = 0.002,
    min_lift: float = 0.02,
) -> tuple[RolloutFailureLabel, FailureFeatures]:
    """Label a failed episode from gripper command/state and end-effector height.

    The heuristic (thresholds tuned on real LIBERO rollouts): a "close cycle" is
    a command transition from open to close; the gripper "holds" when commanded
    closed with more than ``hold_mm`` measured finger separation (fingers meeting
    below ``air_mm`` means it closed on air). Two or more close cycles while ever
    holding is the fumble loop (``re_grasp``); never holding is ``no_grasp``;
    holding without lifting ``min_lift`` is ``grasp_no_lift``; one clean
    hold-and-lift that still failed is ``missed_target``; the rest is ``timeout``.
    """
    ga = np.asarray(gripper_action, dtype=np.float64)
    gs = np.asarray(gripper_state, dtype=np.float64)
    z = np.asarray(eef_z, dtype=np.float64)
    if not (len(ga) == len(gs) == len(z)):
        raise ValueError("gripper_action, gripper_state, and eef_z must have the same length")
    if len(ga) == 0:
        raise ValueError("cannot classify an empty episode")

    closes = np.flatnonzero((ga[1:] > 0) & (ga[:-1] <= 0)) + 1
    closed = ga > 0
    held = closed & (gs > hold_mm)
    features = FailureFeatures(
        steps=len(ga),
        close_cycles=len(closes),
        held_frac=float(held.sum() / max(closed.sum(), 1)),
        ever_held=bool(held.any()),
        max_lift=float(z[held].max() - z.min()) if held.any() else 0.0,
        closed_on_air_frac=float(((gs < air_mm) & closed).sum() / max(closed.sum(), 1)),
    )

    label: RolloutFailureLabel
    if features.close_cycles >= 2 and features.ever_held:
        label = "re_grasp"  # grasp -> lose -> re-attempt (the fumble loop)
    elif not features.ever_held:
        label = "no_grasp"  # never got the object between the fingers
    elif features.max_lift < min_lift:
        label = "grasp_no_lift"  # held it but never lifted
    elif features.close_cycles == 1:
        label = "missed_target"  # held + lifted + still failed => wrong placement
    else:
        label = "timeout"
    return label, features


def label_failures(
    df: daft.DataFrame,
    *,
    hold_mm: float = 0.004,
    air_mm: float = 0.002,
    min_lift: float = 0.02,
) -> daft.DataFrame:
    """Label every failed attempt in a step-row table via `classify_failure`.

    Returns one row per failed attempt (grouped by ``ATTEMPT_KEYS`` - shared
    specs must never chimera across policies) with the label in
    ``terminal_failure`` plus the `FailureFeatures` columns. Requires
    ``gripper_action``, ``gripper_state``, and ``eef_pos`` to be populated on
    failure steps, as rollout writers do.
    """
    grouped = (
        df.where(~col("success"))
        .groupby(*ATTEMPT_KEYS)
        .agg(
            col("suite").any_value(),
            col("task_id").any_value(),
            col("task_name").any_value(),
            col("init_state_id").any_value(),
            col("num_steps").any_value(),
            col("step_idx").list_agg().alias("step_idxs"),
            col("gripper_action").list_agg().alias("gripper_actions"),
            col("gripper_state").list_agg().alias("gripper_states"),
            col("eef_pos").list_agg().alias("eef_positions"),
        )
    )
    data = grouped.to_pydict()

    rows: list[dict[str, object]] = []
    for i, episode_id in enumerate(data["episode_id"]):
        signals = (data["gripper_actions"][i], data["gripper_states"][i], data["eef_positions"][i])
        if any(value is None for track in signals for value in track):
            raise ValueError(f"{episode_id}: null gripper/eef signals; cannot classify")
        order = np.argsort(np.asarray(data["step_idxs"][i]))
        ga = np.asarray(signals[0], dtype=np.float64)[order]
        gs = np.asarray(signals[1], dtype=np.float64)[order]
        z = np.asarray([pos[2] for pos in signals[2]], dtype=np.float64)[order]
        label, features = classify_failure(ga, gs, z, hold_mm=hold_mm, air_mm=air_mm, min_lift=min_lift)
        rows.append(
            {
                "episode_id": episode_id,
                "policy_type": data["policy_type"][i],
                "model": data["model"][i],
                "suite": data["suite"][i],
                "task_id": data["task_id"][i],
                "task_name": data["task_name"][i],
                "init_state_id": data["init_state_id"][i],
                "num_steps": data["num_steps"][i],
                "terminal_failure": label,
                "steps": features.steps,
                "close_cycles": features.close_cycles,
                "held_frac": features.held_frac,
                "ever_held": features.ever_held,
                "max_lift": features.max_lift,
                "closed_on_air_frac": features.closed_on_air_frac,
            }
        )
    return daft.from_pylist(rows)
