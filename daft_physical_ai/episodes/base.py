"""Normalized episode and step representation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .schema import SCHEMA_VERSION, empty_step_row

PRIMARY = "primary"
WRIST = "wrist"

DEFAULT_CAMERA_ROLE_MAPS: dict[str, dict[str, str]] = {
    "lerobot": {
        "observation.images.image": PRIMARY,
        "observation.images.image2": WRIST,
        "observation.images.wrist_image": WRIST,
        "observation.images.agentview": PRIMARY,
    },
    "droid": {
        "exterior_image_1_left": PRIMARY,
        "wrist_image_left": WRIST,
    },
    "hdf5": {
        "agentview_image": PRIMARY,
        "robot0_eye_in_hand_image": WRIST,
        "agentview_rgb": PRIMARY,
        "eye_in_hand_rgb": WRIST,
    },
    "egodex": {},
}


@dataclass(frozen=True)
class Step:
    """One normalized timestep in an episode."""

    timestep: int
    images: dict[str, np.ndarray] = field(default_factory=dict)
    state: np.ndarray | None = None
    action: np.ndarray | None = None
    reward: float | None = None
    done: bool = False
    is_terminal: bool = False
    eef_pos: np.ndarray | None = None
    gripper_state: float | None = None
    object_poses: dict[str, list[float]] = field(default_factory=dict)
    timestamp: float | None = None


@dataclass(frozen=True)
class Episode:
    """An ordered trajectory plus episode-level metadata."""

    episode_id: str
    source: str
    instruction: str
    steps: tuple[Step, ...]
    success: bool
    terminal_failure: str | None = None
    model: str = "dataset"
    policy_type: str = "dataset"
    suite: str | None = None
    task_id: int | None = None
    task_name: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_step_rows(
        self,
        *,
        run_id: str = "ingest",
        frame_path_for=None,
        wrist_path_for=None,
        video_path: str | None = None,
    ) -> list[dict[str, object]]:
        """Flatten this episode into rows matching the canonical schema."""
        rows: list[dict[str, object]] = []
        for step in self.steps:
            row = empty_step_row()
            row.update(
                schema_version=SCHEMA_VERSION,
                episode_id=self.episode_id,
                run_id=run_id,
                model=self.model,
                policy_type=self.policy_type,
                source=self.source,
                suite=self.suite,
                task_id=self.task_id,
                task_name=self.task_name,
                instruction=self.instruction,
                bddl_file=self.metadata.get("bddl_file"),
                init_state_id=self.metadata.get("init_state_id"),
                seed=self.metadata.get("seed"),
                control_mode=self.metadata.get("control_mode"),
                success=self.success,
                terminal_failure=self.terminal_failure,
                num_steps=self.num_steps,
                step_idx=step.timestep,
                action=None if step.action is None else _as_f32_list(step.action),
                reward=None if step.reward is None else float(step.reward),
                done=bool(step.done),
                state=None if step.state is None else _as_f32_list(step.state),
                eef_pos=None if step.eef_pos is None else _as_f32_list(step.eef_pos),
                gripper_state=step.gripper_state,
                gripper_action=(None if step.action is None else float(np.asarray(step.action).ravel()[-1])),
                object_poses=json.dumps(step.object_poses) if step.object_poses else None,
                frame_path=frame_path_for(self.episode_id, step.timestep) if frame_path_for else None,
                wrist_path=wrist_path_for(self.episode_id, step.timestep) if wrist_path_for else None,
                video_path=video_path or self.metadata.get("video_path"),
                embedding=None,
            )
            rows.append(row)
        return rows


def _as_f32_list(arr) -> list[float]:
    """Cast an array-like value to a flat float32 list."""
    return [float(value) for value in np.asarray(arr, dtype=np.float32).ravel()]


class Ingestor(ABC):
    """Abstract dataset adapter that yields normalized episodes."""

    source: str = "base"

    def __init__(self, camera_role_map: dict[str, str] | None = None) -> None:
        self.camera_role_map = camera_role_map or DEFAULT_CAMERA_ROLE_MAPS.get(self.source, {})

    @abstractmethod
    def load(self, path: str, *, limit: int | None = None) -> Iterator[Episode]:
        """Yield normalized episodes from ``path``."""
        raise NotImplementedError(self.load.__doc__)
