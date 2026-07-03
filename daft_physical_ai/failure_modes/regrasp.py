"""Detect slip-then-regrasp loops from per-step episode signals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

FailureLabel = Literal["re_grasp", "drop_no_recover", "missed_target", "no_grasp"]


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
