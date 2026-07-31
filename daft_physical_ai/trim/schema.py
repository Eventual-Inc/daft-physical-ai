"""Output schema for motion trimming - one row per episode."""

from __future__ import annotations

from daft import DataType

# Columns `trim_windows` adds to an episode-level DataFrame. Frame indices are
# episode-local; the timestamps are absolute within the episode's video file, so
# a decoder can seek straight to them.
TRIM_FIELDS = {
    "start_frame": DataType.int64(),  # first kept frame, episode-local
    "end_frame": DataType.int64(),  # last kept frame, inclusive
    "start_ts": DataType.float64(),  # start_frame as a timestamp in the video file
    "end_ts": DataType.float64(),  # end_frame as a timestamp in the video file
    "kept_frames": DataType.int64(),
    "trim_fraction": DataType.float64(),  # share of the episode dropped, 0-1
    "never_active": DataType.bool(),  # no motion anywhere: an aborted take
}

TRIM_DTYPE = DataType.struct(TRIM_FIELDS)

__all__ = ["TRIM_DTYPE", "TRIM_FIELDS"]
