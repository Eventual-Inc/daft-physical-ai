"""Deterministic episode operations over canonical step-row tables.

Operations here understand trajectory semantics (idle spans, gripper events)
but stay model-free and simulator-free: pure NumPy cores wrapped in Daft
groupbys, so they run identically on demonstrations and rollouts.
"""

from __future__ import annotations

from .motion import TrimSpan, motion_trim, noop_mask

__all__ = ["TrimSpan", "motion_trim", "noop_mask"]
