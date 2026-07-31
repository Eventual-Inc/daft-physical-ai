"""Physical-AI dataset access, hand tracking, reward scoring, and motion trimming for Daft DataFrames.

`track_hands(images, method=...)` takes an image column (a Daft expression) and
returns a hand-pose column, so it composes with any Daft pipeline. Every method
returns the same output schema - see `HANDS_DTYPE`.

`score_rewards(...)` takes episode-metadata columns and returns a reward column
(per-frame task progress + success probability) scored against a Robometer eval
server you run - see `REWARD_DTYPE`.

`motion_energy(...)` / `is_active(...)` (in `daft_physical_ai.proprio`) score
per-frame motion from the robot's own state columns, and `trim_windows(...)`
(in `daft_physical_ai.trim`) reduces the flags to one trim window per episode -
see `TRIM_FIELDS`. Nothing decodes video.
"""

from __future__ import annotations

from . import datasets
from .hands import HAND_DTYPE, HANDS_DTYPE
from .rewards import REWARD_DTYPE, REWARD_FRAME_DTYPE

__all__ = ["HANDS_DTYPE", "HAND_DTYPE", "REWARD_DTYPE", "REWARD_FRAME_DTYPE", "datasets"]
