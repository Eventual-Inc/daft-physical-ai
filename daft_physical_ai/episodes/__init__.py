"""Canonical episode tables for physical-AI datasets and rollouts."""

from __future__ import annotations

from .base import Episode, Ingestor, Step
from .schema import (
    ACTION_DIM,
    COLUMNS,
    EMBEDDING_DIM,
    ROLLOUT_SCHEMA,
    SCHEMA_VERSION,
    STATE_DIM,
    TERMINAL_FAILURE_LABELS,
    empty_step_row,
    validate_rows,
)
from .writer import RolloutWriter, assert_emits_schema, write_episode, write_rows

__all__ = [
    "ACTION_DIM",
    "COLUMNS",
    "EMBEDDING_DIM",
    "ROLLOUT_SCHEMA",
    "SCHEMA_VERSION",
    "STATE_DIM",
    "TERMINAL_FAILURE_LABELS",
    "Episode",
    "Ingestor",
    "RolloutWriter",
    "Step",
    "assert_emits_schema",
    "empty_step_row",
    "validate_rows",
    "write_episode",
    "write_rows",
]
