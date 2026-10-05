"""Lazy OmniSharing episode discovery and HDF5 layout description."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import daft
from daft.datatype import DataType
from daft.expressions import col, lit
from daft.functions import coalesce, format, hdf5_file, hdf5_metadata, regexp_extract, regexp_replace, unnest, when

from ._common import DF2_SUFFIX, STAGE_DF1, STAGE_DF2, STAGE_DF2R, STAGES, require_episode_column

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.io import IOConfig

_EPISODE_FILENAME_RE = r"episode_(\d+)_(\d+)_(\d+)_(\d+)(?:_([A-Za-z0-9]+))?\.(?:hdf5|h5)$"
_HF_REPO_ID_RE = re.compile(r"[\w.-]+/[\w.-]+")


def _normalize_dataset_root(uri: str) -> str:
    root = uri.strip()
    if not root:
        raise ValueError("dataset_uri must be a non-empty string")
    if "://" not in root and not root.startswith(("/", ".", "~")) and _HF_REPO_ID_RE.fullmatch(root):
        return f"hf://datasets/{root}"
    if "://" not in root:
        return Path(root).expanduser().resolve().as_uri().rstrip("/")
    return root.rstrip("/")


def raw(dataset_uri: str, io_config: IOConfig | None = None, stage: str | None = None) -> DataFrame:
    """Return a lazy catalog of OmniSharing HDF5 episodes without opening them."""
    if stage is not None and stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Expected one of: {', '.join(STAGES)}.")

    root = _normalize_dataset_root(dataset_uri)
    files = daft.from_glob_path([f"{root}/**/*.hdf5", f"{root}/**/*.h5"], io_config=io_config)
    parsed = files.select(
        "path",
        "size",
        regexp_extract(col("path"), _EPISODE_FILENAME_RE, 1).alias("episode_index"),
        regexp_extract(col("path"), _EPISODE_FILENAME_RE, 2).alias("capture_time"),
        regexp_extract(col("path"), _EPISODE_FILENAME_RE, 3).alias("room_id"),
        regexp_extract(col("path"), _EPISODE_FILENAME_RE, 4).alias("personnel_id"),
        regexp_extract(col("path"), _EPISODE_FILENAME_RE, 5).alias("suffix"),
    ).where(col("episode_index").not_null() & (col("episode_index") != lit("")))

    suffix = coalesce(col("suffix"), lit(""))
    is_df2 = suffix == lit(DF2_SUFFIX)
    has_suffix = suffix != lit("")
    relative_path = regexp_replace(col("path"), rf"^{re.escape(root)}/", "")
    episode_key = regexp_replace(relative_path, r"\.(?:hdf5|h5)$", "")
    capture_key = format(
        "episode_{}_{}_{}_{}",
        col("episode_index"),
        col("capture_time"),
        col("room_id"),
        col("personnel_id"),
    )

    parsed = parsed.with_columns(
        {
            "episode_key": episode_key,
            "capture_key": capture_key,
            "stage": when(is_df2, lit(STAGE_DF2)).when(has_suffix, lit(STAGE_DF2R)).otherwise(lit(STAGE_DF1)),
            "hand_model": when(is_df2 | ~has_suffix, lit(None)).otherwise(suffix),
            "episode": hdf5_file(col("path"), io_config=io_config),
        }
    )
    if stage is not None:
        parsed = parsed.where(col("stage") == lit(stage))

    return parsed.select(
        "episode_key",
        "capture_key",
        col("episode_index").cast(DataType.int64()),
        "capture_time",
        col("room_id").cast(DataType.int64()),
        col("personnel_id").cast(DataType.int64()),
        "stage",
        "hand_model",
        "episode",
        "path",
        "size",
    )


def describe(episodes: DataFrame) -> DataFrame:
    """Return Daft's public HDF5 metadata rows for each episode."""
    require_episode_column(episodes)
    objects = episodes.select("episode_key", hdf5_metadata(col("episode")).alias("object")).explode("object")
    return objects.select("episode_key", unnest(col("object")))


__all__ = ["describe", "raw"]
