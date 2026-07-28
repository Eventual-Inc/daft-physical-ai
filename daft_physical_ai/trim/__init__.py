"""Motion trimming for Daft DataFrames.

`trim_windows(...)` reduces a frame-level activity flag to one contiguous window
per episode: the span from the first motion to the last, padded. Unlike the rest
of the package this takes a DataFrame and returns a DataFrame - a window is an
aggregation across an episode's rows, not a value each row can carry.

The window is for consumers that need a video slice, where `from_ts`/`to_ts`
must stay contiguous. Consumers that sample frames should filter on the
frame-level `is_active` column instead and drop interior pauses too, which the
window has to keep.
"""

from __future__ import annotations

import daft
from daft import DataFrame, col, lit
from daft.functions import when

from .schema import TRIM_DTYPE, TRIM_FIELDS

__all__ = ["TRIM_DTYPE", "TRIM_FIELDS", "trim_windows"]

DEFAULT_PAD_S = 0.25  # cutting flush to the first motion clips the approach


def trim_windows(
    frames: DataFrame,
    *,
    fps: float,
    pad_s: float = DEFAULT_PAD_S,
    episode: str = "episode_index",
    order: str = "frame_index",
    active: str = "is_active",
    from_ts: str | None = None,
) -> DataFrame:
    """Reduce per-frame activity to one trim window per episode.

    Args:
        frames: frame-level DataFrame carrying an ``active`` boolean column
            (see :func:`daft_physical_ai.proprio.is_active`).
        fps: frame rate, used to convert frame indices to timestamps.
        pad_s: seconds kept on each side of the active span.
        episode: column identifying the episode.
        order: per-frame index column, episode-local.
        active: boolean column marking frames with real motion.
        from_ts: optional column holding where the episode starts inside its
            video file (e.g. ``videos/{key}/from_timestamp``). When given,
            ``start_ts``/``end_ts`` are absolute within that file; otherwise
            they are relative to the episode.

    Returns:
        One row per episode: the ``episode`` column plus the fields in
        :data:`~daft_physical_ai.trim.schema.TRIM_FIELDS`. Episodes with no
        motion anywhere keep their full span and are flagged ``never_active``,
        rather than being trimmed to nothing.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if pad_s < 0:
        raise ValueError(f"pad_s must be >= 0, got {pad_s}")

    pad = round(pad_s * fps)
    # Null on inactive frames, so min/max see only the active ones.
    active_index = when(col(active), col(order)).otherwise(lit(None).cast(daft.DataType.int64()))

    aggs = [
        col(order).max().alias("_last_index"),
        active_index.min().alias("_first_active"),
        active_index.max().alias("_last_active"),
    ]
    if from_ts is not None:
        aggs.append(col(from_ts).min().alias("_from_ts"))

    spans = frames.groupby(episode).agg(*aggs)
    offset = col("_from_ts") if from_ts is not None else lit(0.0)

    return (
        spans.with_column("never_active", col("_first_active").is_null())
        .with_column(
            "start_frame",
            when(col("never_active"), lit(0)).otherwise((col("_first_active") - pad).clip(lit(0))),
        )
        .with_column(
            "end_frame",
            when(col("never_active"), col("_last_index")).otherwise(
                (col("_last_active") + pad).clip(max=col("_last_index"))
            ),
        )
        .with_column("kept_frames", col("end_frame") - col("start_frame") + 1)
        .with_column(
            "trim_fraction",
            1.0 - col("kept_frames") / (col("_last_index") + 1),
        )
        .with_column("start_ts", offset + col("start_frame") / fps)
        .with_column("end_ts", offset + col("end_frame") / fps)
        .select(episode, *TRIM_FIELDS)
    )
