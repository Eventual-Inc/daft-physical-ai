"""Physical-AI dataset access, hand tracking, and reward scoring for Daft DataFrames.

`track_hands(images, method=...)` takes an image column (a Daft expression) and
returns a hand-pose column, so it composes with any Daft pipeline. Every method
returns the same output schema - see `HANDS_DTYPE`.

`score_rewards(...)` takes episode-metadata columns and returns a reward column
(per-frame task progress + success probability) scored against a Robometer eval
server you run - see `REWARD_DTYPE`.
"""

from __future__ import annotations

from . import datasets
from .hands import HAND_DTYPE, HANDS_DTYPE
from .rewards import REWARD_DTYPE, REWARD_FRAME_DTYPE

__all__ = ["HANDS_DTYPE", "HAND_DTYPE", "REWARD_DTYPE", "REWARD_FRAME_DTYPE", "datasets"]
