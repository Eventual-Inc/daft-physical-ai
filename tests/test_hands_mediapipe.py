from __future__ import annotations

import daft
import numpy as np
import pytest

from daft_physical_ai.hands import HANDS_DTYPE, track_hands


def test_unknown_method_raises() -> None:
    df = daft.from_pydict({"x": [1]})
    with pytest.raises(ValueError):
        track_hands(df["x"], method="nope")


def test_mediapipe_runs_and_matches_schema() -> None:
    pytest.importorskip("mediapipe")
    img = np.zeros((64, 64, 3), dtype=np.uint8)  # blank frame: pipeline runs, finds 0 hands
    df = daft.from_pydict({"image": [img]})
    df = df.with_column("hands", track_hands(df["image"], method="mediapipe"))

    field = next(f for f in df.schema() if f.name == "hands")
    assert str(field.dtype) == str(HANDS_DTYPE)

    result = df.select("hands").to_pydict()
    assert isinstance(result["hands"][0], list)  # 0 or more detected hands
