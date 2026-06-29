"""Hand tracking for Daft DataFrames.

`track_hands(images, method=...)` takes an image column (a Daft expression) and
returns a hand-pose column (also an expression), so it composes with any Daft
pipeline and runs lazily. The output schema is the same for every method - see
`schema.HANDS_DTYPE`.
"""

from __future__ import annotations

from typing import cast

from daft import Expression

from .schema import HAND_DTYPE, HANDS_DTYPE

__all__ = ["HANDS_DTYPE", "HAND_DTYPE", "track_hands"]


def track_hands(
    images: Expression,
    *,
    method: str,
    mano_path: str | None = None,
    model_path: str | None = None,
    num_hands: int = 2,
    min_confidence: float = 0.3,
) -> Expression:
    """Detect hands in an image column.

    Args:
        images: a Daft image column (expression).
        method: ``"mediapipe"`` (CPU, 2D) or ``"wilor"`` (GPU, 3D - not yet implemented).
        mano_path: path to MANO weights, required by ``"wilor"`` (research-gated, not bundled).
        model_path: override the MediaPipe model path (defaults to a local cache).
        num_hands: max hands to detect per frame.
        min_confidence: minimum detection confidence.

    Returns:
        An expression yielding ``list[struct{handedness, confidence, kp2d, kp3d?}]``
        per frame (``kp3d`` is null for 2D-only methods).
    """
    if method == "mediapipe":
        from ._mediapipe import MediaPipeHands

        tracker = MediaPipeHands(model_path=model_path, num_hands=num_hands, min_confidence=min_confidence)
        return cast(Expression, tracker.track(images))

    if method == "wilor":
        raise NotImplementedError("method='wilor' is not implemented yet (see AGENTS.md roadmap).")

    raise ValueError(f"unknown method {method!r}; expected 'mediapipe' or 'wilor'")
