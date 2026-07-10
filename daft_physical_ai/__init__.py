"""Physical-AI dataset access and hand tracking for Daft DataFrames.

`track_hands(images, method=...)` takes an image column (a Daft expression) and
returns a hand-pose column, so it composes with any Daft pipeline. Every method
returns the same output schema - see `HANDS_DTYPE`.
"""

from __future__ import annotations

from . import datasets
from .hands import HAND_DTYPE, HANDS_DTYPE

__all__ = ["HANDS_DTYPE", "HAND_DTYPE", "datasets"]
