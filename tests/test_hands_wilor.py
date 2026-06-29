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
    # track_hands(method="wilor") imports torch in the caller's process (so Daft's
    # worker import is a safe no-op), so this needs torch present - but still no
    # execution, no asset download, no GPU.
    pytest.importorskip("torch")
    df = daft.from_pydict({"image": [1]})
    df = df.with_column("hands", track_hands(df["image"], method="wilor", mano_path="/fake/MANO_RIGHT.pkl"))
    field = next(f for f in df.schema() if f.name == "hands")
    assert str(field.dtype) == str(HANDS_DTYPE)


def test_ensure_assets_places_mano(tmp_path) -> None:
    """No GPU/network: a correctly-provided mano_path is copied where WiLoR expects it."""
    from daft_physical_ai.hands._wilor import ensure_assets

    root = tmp_path / "WiLoR"
    # Pretend the repo + weights are already present so clone/download are skipped.
    (root / "wilor").mkdir(parents=True)
    pretrained = root / "pretrained_models"
    pretrained.mkdir()
    for name in ("detector.pt", "wilor_final.ckpt", "model_config.yaml"):
        (pretrained / name).write_bytes(b"stub")
    (root / "mano_data").mkdir()
    (root / "mano_data" / "mano_mean_params.npz").write_bytes(b"stub")

    mano = tmp_path / "MANO_RIGHT.pkl"
    mano.write_bytes(b"mano-weights")

    returned = ensure_assets(wilor_root=str(root), mano_path=str(mano))

    assert returned == str(root)
    for placed in (root / "mano_data" / "MANO_RIGHT.pkl", root / "mano_data" / "mano" / "MANO_RIGHT.pkl"):
        assert placed.read_bytes() == b"mano-weights"
