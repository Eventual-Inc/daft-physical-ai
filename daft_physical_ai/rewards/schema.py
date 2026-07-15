"""Shared output schema for reward scoring - server-agnostic.

Every episode gets per-frame task progress, per-frame success probability, and
the sampled frame references, so downstream code (filtering, RL post-training,
label QA) is identical regardless of where the model is served.
"""

from __future__ import annotations

from daft import DataType

# One sampled frame: its index relative to the episode start and its timestamp.
REWARD_FRAME_DTYPE = DataType.struct(
    {
        "index": DataType.int64(),
        "timestamp_s": DataType.float64(),
    }
)

# One episode's scores. Column names match Macrodata's write-back columns
# (`reward_score`, `robometer_success`, `reward_frames`) for direct contrast.
REWARD_DTYPE = DataType.struct(
    {
        "reward_score": DataType.list(DataType.float64()),  # per-frame task progress, 0-1
        "robometer_success": DataType.list(DataType.float64()),  # per-frame success probability
        "reward_frames": DataType.list(REWARD_FRAME_DTYPE),
    }
)

__all__ = ["REWARD_DTYPE", "REWARD_FRAME_DTYPE"]
