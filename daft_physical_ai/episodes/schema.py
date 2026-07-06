"""Canonical rollout and dataset episode schema.

The contract is one row per step. Episode-level fields such as success,
instruction, source, and terminal failure are denormalized onto every row so a
single Daft parquet scan can filter failures without a side table or join.

The on-disk format uses portable Arrow primitives only. Media columns store
paths or URLs rather than inline bytes, which keeps parquet files small and lets
Daft decode media lazily from path columns.
"""

from __future__ import annotations

import pyarrow as pa

ACTION_DIM = 7
STATE_DIM = 8
EMBEDDING_DIM = 1024
SCHEMA_VERSION = "rollout-v1"

# Mirrored by the VLA-JEPA harness's ``harness/schema.py`` — keep the two in sync
# when adding labels (values only, so SCHEMA_VERSION is unaffected).
TERMINAL_FAILURE_LABELS = (
    "re_grasp",
    "no_grasp",
    "grasp_no_lift",
    "drop_no_recover",
    "wrong_object",
    "missed_target",
    "timeout",
    "collision",
    "unlabeled",
)


def rollout_schema() -> pa.Schema:
    """Return the canonical one-row-per-step Arrow schema."""
    f32 = pa.float32()
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("policy_type", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("suite", pa.string(), nullable=True),
            pa.field("task_id", pa.int32(), nullable=True),
            pa.field("task_name", pa.string(), nullable=True),
            pa.field("instruction", pa.string(), nullable=False),
            pa.field("bddl_file", pa.string(), nullable=True),
            pa.field("init_state_id", pa.int32(), nullable=True),
            pa.field("seed", pa.int64(), nullable=True),
            pa.field("control_mode", pa.string(), nullable=True),
            pa.field("success", pa.bool_(), nullable=False),
            pa.field("terminal_failure", pa.string(), nullable=True),
            pa.field("num_steps", pa.int32(), nullable=False),
            pa.field("step_idx", pa.int32(), nullable=False),
            pa.field("action", pa.list_(f32), nullable=True),
            pa.field("reward", f32, nullable=True),
            pa.field("done", pa.bool_(), nullable=True),
            pa.field("state", pa.list_(f32), nullable=True),
            pa.field("eef_pos", pa.list_(f32), nullable=True),
            pa.field("gripper_state", f32, nullable=True),
            pa.field("gripper_action", f32, nullable=True),
            pa.field("object_poses", pa.string(), nullable=True),
            pa.field("frame_path", pa.string(), nullable=True),
            pa.field("wrist_path", pa.string(), nullable=True),
            pa.field("video_path", pa.string(), nullable=True),
            pa.field("embedding", pa.list_(f32), nullable=True),
        ],
        metadata={
            b"schema_version": SCHEMA_VERSION.encode(),
            b"action_dim": str(ACTION_DIM).encode(),
            b"state_dim": str(STATE_DIM).encode(),
            b"embedding_dim": str(EMBEDDING_DIM).encode(),
        },
    )


ROLLOUT_SCHEMA: pa.Schema = rollout_schema()
COLUMNS: tuple[str, ...] = tuple(ROLLOUT_SCHEMA.names)


def empty_step_row() -> dict[str, object]:
    """Return a row dict with every schema column set to ``None``."""
    return {name: None for name in COLUMNS}


def validate_rows(rows: list[dict[str, object]]) -> pa.Table:
    """Build a typed Arrow table from row dicts under ``ROLLOUT_SCHEMA``."""
    return pa.Table.from_pylist(rows, schema=ROLLOUT_SCHEMA)
