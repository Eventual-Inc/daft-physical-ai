"""Dataset adapters that normalize raw robot data into canonical episodes.

Daft owns the readers for formats it supports natively (LeRobot, DROID,
parquet, video); adapters live here only where physical-AI data ships in a
format Daft has no reader for. Each adapter yields
`daft_physical_ai.episodes.Episode` objects, which land in the one-row-per-step
parquet contract via the episode writers.
"""

from __future__ import annotations

from .hdf5 import Hdf5Ingestor

__all__ = ["Hdf5Ingestor"]
