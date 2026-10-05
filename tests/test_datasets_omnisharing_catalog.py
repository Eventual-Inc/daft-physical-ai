from __future__ import annotations

from pathlib import Path

import pytest
from daft import DataType, MediaType, col, lit

from daft_physical_ai.datasets import omnisharing
from daft_physical_ai.datasets.omnisharing._catalog import _normalize_dataset_root
from tests.omnisharing_datagen import episode_filename, write_df2_episode


@pytest.fixture
def catalog_root(tmp_path: Path) -> Path:
    for directory, suffix in (("DF-1", None), ("DF-2", "glove"), ("DF-2R", "mano")):
        write_df2_episode(
            tmp_path / directory / episode_filename(7, "212953", 93, 110056, suffix),
            n_frames=3,
        )
    write_df2_episode(tmp_path / "DF-2" / episode_filename(7, "213000", 94, 110057), n_frames=2)
    (tmp_path / "DF-2" / "ignore.hdf5").write_bytes(b"not hdf5")
    return tmp_path


def test_uri_normalization(tmp_path: Path) -> None:
    assert _normalize_dataset_root("paxini/Omnisharing_DB_SampleData") == (
        "hf://datasets/paxini/Omnisharing_DB_SampleData"
    )
    assert _normalize_dataset_root(f"{tmp_path}/") == tmp_path.as_uri()
    assert _normalize_dataset_root("s3://bucket/prefix/") == "s3://bucket/prefix"
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_dataset_root("  ")


def test_raw_has_exact_order_and_types(catalog_root: Path) -> None:
    schema = omnisharing.raw(str(catalog_root)).schema()
    assert [field.name for field in schema] == [
        "episode_key",
        "capture_key",
        "episode_index",
        "capture_time",
        "room_id",
        "personnel_id",
        "stage",
        "hand_model",
        "episode",
        "path",
        "size",
    ]
    assert schema["episode_key"].dtype == DataType.string()
    assert schema["episode_index"].dtype == DataType.int64()
    assert schema["episode"].dtype == DataType.file(MediaType.hdf5())  # type: ignore[attr-defined]
    assert schema["size"].dtype == DataType.int64()


def test_raw_uses_unique_path_identity_and_shared_capture_identity(catalog_root: Path) -> None:
    rows = omnisharing.raw(str(catalog_root)).sort("episode_key").to_pylist()
    assert len(rows) == 4
    assert len({row["episode_key"] for row in rows}) == 4
    assert {row["episode_key"] for row in rows} >= {
        "DF-1/episode_7_212953_93_110056",
        "DF-2/episode_7_212953_93_110056_glove",
        "DF-2R/episode_7_212953_93_110056_mano",
    }
    same_capture = [row for row in rows if row["capture_time"] == "212953"]
    assert {row["capture_key"] for row in same_capture} == {"episode_7_212953_93_110056"}
    assert [(row["stage"], row["hand_model"]) for row in same_capture] == [
        ("DF-1", None),
        ("DF-2", None),
        ("DF-2R", "mano"),
    ]


@pytest.mark.parametrize("stage,count", [("DF-1", 1), ("DF-2", 2), ("DF-2R", 1)])
def test_raw_stage_filter(catalog_root: Path, stage: str, count: int) -> None:
    rows = omnisharing.raw(str(catalog_root), stage=stage).to_pylist()
    assert len(rows) == count
    assert {row["stage"] for row in rows} == {stage}


def test_raw_rejects_unknown_stage(catalog_root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        omnisharing.raw(str(catalog_root), stage="DF-3")


def test_raw_plan_and_collection_do_not_open_hdf5(catalog_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daft.file.hdf5 import Hdf5File

    def forbidden(*args, **kwargs):
        raise AssertionError("catalog opened HDF5 content")

    monkeypatch.setattr(Hdf5File, "open", forbidden)
    result = omnisharing.raw(str(catalog_root)).select("episode_key", "capture_key").collect()
    assert result.count_rows() == 4


def test_describe_is_thin_public_metadata_composition(catalog_root: Path) -> None:
    episodes = omnisharing.raw(str(catalog_root), stage="DF-2").where(col("capture_time") == lit("212953"))
    layout = omnisharing.describe(episodes)
    assert [field.name for field in layout.schema()] == [
        "episode_key",
        "h5path",
        "kind",
        "shape",
        "dtype",
        "chunks",
        "compression",
    ]
    tactile = layout.where(col("h5path").endswith("lefthand/tactile/data")).to_pylist()
    assert tactile[0]["kind"] == "dataset"
    assert tactile[0]["shape"] == [3, 3465]
    assert tactile[0]["dtype"] == "float32"
