"""WiLoR facade tests that need no GPU/torch.

`@daft.cls` is lazy: building the expression and its schema does not run the
class `__init__` (which would import torch and fetch the WiLoR assets). Actually
running WiLoR is GPU-only and is exercised on Modal, not in CI.
"""

from __future__ import annotations

import daft
import pytest

from daft_physical_ai.hands import HANDS_DTYPE, track_hands


def test_wilor_requires_mano_path() -> None:
    df = daft.from_pydict({"image": [1]})
    with pytest.raises(ValueError, match="mano_path"):
        track_hands(df["image"], method="wilor")


def test_wilor_expression_has_shared_schema() -> None:
    # No execution -> no torch, no asset download, no GPU.
    df = daft.from_pydict({"image": [1]})
    df = df.with_column("hands", track_hands(df["image"], method="wilor", mano_path="/fake/MANO_RIGHT.pkl"))
    field = next(f for f in df.schema() if f.name == "hands")
    assert str(field.dtype) == str(HANDS_DTYPE)
