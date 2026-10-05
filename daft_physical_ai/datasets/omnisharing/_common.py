"""Shared OmniSharing layout, validation, and public HDF5 helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import daft
from daft.datatype import DataType
from daft.expressions import col, lit
from daft.functions import coalesce

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from daft.dataframe import DataFrame
    from daft.file.hdf5 import Hdf5File

ROOT_GROUP = "dataset"
DF2_SUFFIX = "glove"
STAGE_DF1 = "DF-1"
STAGE_DF2 = "DF-2"
STAGE_DF2R = "DF-2R"
STAGES: tuple[str, ...] = (STAGE_DF1, STAGE_DF2, STAGE_DF2R)
SIDES: tuple[str, ...] = ("lefthand", "righthand")
BRANCHES: tuple[str, ...] = ("observation", "action")
HANDPOSE_ORDER: tuple[str, ...] = ("x", "y", "z", "qw", "qx", "qy", "qz")
READ_ERROR_COLUMN = "_omnisharing_read_error"


class OmniSharingReadError(ValueError):
    """An episode could not be read as requested (missing dataset, bad layout, DF-1, ...)."""


def h5path(*parts: str) -> str:
    return "/".join((ROOT_GROUP, *(part.strip("/") for part in parts if part)))


def to_python(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        return to_python(value.item())
    if hasattr(value, "tolist"):
        return to_python(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def attrs(node: Any) -> dict[str, Any]:
    return {str(key): to_python(value) for key, value in node.attrs.items()}


def get_node(h5: Any, path: str) -> Any | None:
    try:
        return h5[path]
    except KeyError:
        return None


@contextmanager
def open_h5(file: Hdf5File) -> Iterator[Any]:
    """Open HDF5 through the public Daft file interface without localization."""
    from daft.dependencies import h5py

    with file.open() as stream, h5py.File(stream, "r") as h5:
        yield h5


def require_episode_column(episodes: DataFrame) -> None:
    if "episode" not in episodes.column_names:
        raise ValueError("Expected an episode DataFrame with an 'episode' column, as produced by omnisharing.raw().")


def require_materializable(stage: str) -> None:
    if stage == STAGE_DF1:
        raise ValueError(
            "OmniSharing DF-1 episodes are catalog-only (their encoder and tactile streams are unparsed). "
            "Filter with raw(..., stage='DF-2') or 'DF-2R' before reading modalities."
        )


def _placeholder(dtype: DataType) -> Any:
    """Return a typed-empty value Daft can build even when every row in a batch is a placeholder."""
    if dtype.is_list() or dtype.is_fixed_size_list():
        return []
    if dtype.is_tensor():
        from daft.dependencies import np

        return np.empty((0,))
    return None


def _describe_error(error: Exception) -> str:
    message = str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)
    return f"{type(error).__name__}: {message}"


def read_episodes(
    episodes: DataFrame,
    fields: dict[str, DataType],
    read: Callable[[Any], dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> DataFrame:
    """Run ``read(h5)`` once per episode and append its ``fields`` as columns.

    When every row of a batch fails, Daft 0.7.20 replaces a UDF's exception with
    ``Need at least 1 series to perform concat`` if the return type holds a list
    or tensor. Failures are therefore caught here, returned as typed-empty
    placeholders plus an error string, and re-raised by a scalar check, whose
    exception Daft does propagate. The returned DataFrame keeps ``columns``
    (default: every input column) followed by ``fields``.
    """
    require_episode_column(episodes)
    return_dtype = DataType.struct({**fields, READ_ERROR_COLUMN: DataType.string()})

    @daft.func(return_dtype=return_dtype, use_process=False, unnest=True)
    def read_episode(file: Hdf5File, stage: str) -> dict[str, Any]:
        try:
            require_materializable(stage)
            with open_h5(file) as h5:
                result = read(h5)
        except Exception as error:  # noqa: BLE001 - re-raised by check_episode_read
            failed: dict[str, Any] = {name: _placeholder(dtype) for name, dtype in fields.items()}
            failed[READ_ERROR_COLUMN] = f"{file.path}: {_describe_error(error)}"
            return failed
        return {**result, READ_ERROR_COLUMN: None}

    @daft.func(return_dtype=DataType.bool(), use_process=False)
    def check_episode_read(error: str) -> bool:
        if error:
            raise OmniSharingReadError(error)
        return True

    kept = episodes.column_names if columns is None else list(columns)
    appended = episodes.select(*kept, read_episode(col("episode"), col("stage")))
    checked = appended.where(check_episode_read(coalesce(col(READ_ERROR_COLUMN), lit(""))))
    return checked.exclude(READ_ERROR_COLUMN)
