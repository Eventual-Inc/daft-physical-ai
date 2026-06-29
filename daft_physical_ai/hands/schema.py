"""Shared output schema for hand tracking - method-agnostic.

Every method (`mediapipe`, `wilor`, ...) returns the same column type so
downstream code is identical regardless of which model produced it.
"""

from __future__ import annotations

from daft import DataType

# One detected hand. `kp3d` is null for 2D-only methods (e.g. MediaPipe).
HAND_DTYPE = DataType.struct(
    {
        "handedness": DataType.string(),  # "left" | "right" | "unknown"
        "confidence": DataType.float32(),
        "kp2d": DataType.list(DataType.list(DataType.float32())),  # [[x, y], ...] image-space (21 pts)
        "kp3d": DataType.list(DataType.list(DataType.float32())),  # [[x, y, z], ...] (21 pts) or null
    }
)

# One frame -> a list of detected hands (0, 1, or 2).
HANDS_DTYPE = DataType.list(HAND_DTYPE)

__all__ = ["HANDS_DTYPE", "HAND_DTYPE"]
