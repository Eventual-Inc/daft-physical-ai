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
    wilor_root: str | None = None,
    device: str = "cuda",
    model_path: str | None = None,
    num_hands: int = 2,
    min_confidence: float = 0.3,
) -> Expression:
    """Detect hands in an image column.

    Args:
        images: a Daft image column (expression).
        method: ``"mediapipe"`` (CPU, 2D) or ``"wilor"`` (GPU, 3D MANO keypoints).
        mano_path: path to MANO weights, required by ``"wilor"`` (research-gated, not bundled).
        wilor_root: where the WiLoR repo + weights live (``"wilor"`` only; defaults to
            ``$DAFT_PHYSICAL_AI_WILOR_ROOT`` or ``/WiLoR``, fetched on first use).
        device: torch device for ``"wilor"`` (default ``"cuda"``; ``"cpu"`` is slow).
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
        if not mano_path:
            raise ValueError("method='wilor' requires mano_path (MANO_RIGHT.pkl; research-gated).")
        # Import torch here, in the caller's process, BEFORE Daft lazily runs the class-UDF
        # on a worker. Importing torch/CUDA for the first time inside the worker segfaults;
        # loading it in the main process first makes the worker's import a safe no-op. This is
        # a Daft+torch interaction, not environment-specific (applies to local GPU, Ray, etc.).
        try:
            import torch
        except ImportError as err:
            raise ImportError(
                "method='wilor' requires the WiLoR extras: `pip install \"daft-physical-ai[wilor]\"`."
            ) from err
        from ._wilor import WiLoRHands

        wilor_tracker = WiLoRHands(mano_path=mano_path, wilor_root=wilor_root, device=device)
        return cast(Expression, wilor_tracker.track(images))

    raise ValueError(f"unknown method {method!r}; expected 'mediapipe' or 'wilor'")
