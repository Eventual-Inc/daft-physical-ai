"""In-DAG temporal action rates over per-frame pose features.

The distributed counterpart of `daft_physical_ai.pose.features.TemporalFeatureComputer`:
every rate the scenario queries consume becomes a Daft window expression over
``Window().partition_by(*episode_keys).order_by(order_by)`` — next-frame diffs
via ``lead(1)``, point speeds via ``euclidean_distance``, and smoothing via a
centered ``rows_between(-2, 2)`` mean (which shrinks at episode edges). The one
custom UDF, ``forearm_roll``, is fed by the same window — this frame's wrist
rotation plus the next frame's via ``lead(1)`` — and reduces the pair to a roll
rate about the forearm axis.

Because the rates are expressions, the whole computation stays in the query
plan: no collect, and the same code runs on any per-frame table with the
spatial columns — a LeRobot frame table, exploded EgoDex trajectories, or the
canonical step rows. Rate columns are added only for the spatial columns
present, so state-only tables get the state-only rates.

Per-frame spatial inputs per hand tag ``L``/``R`` (produce them with
`state_frame_features` for 48-D state tables):

    curl        -> curl_rate        (grasping)
    wrist_y     -> wrist_vert_vel   (lifting)
    wrist (3,)  -> wrist_speed      (stillness)
    arm_extension -> arm_ext_rate   (reaching; skeleton-derived)
    local_joints  -> articulation   (in-hand; skeleton-derived)
    wrist_rot6d (6,) + forearm_axis (3,) -> roll (twisting; skeleton-derived)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import daft
import numpy as np
from daft import DataType, col
from daft.functions import euclidean_distance
from daft.window import Window

from . import state as state_geometry
from .features import FPS, HANDS, ROLL_SMOOTH_HALF_WIDTH

if TYPE_CHECKING:
    from collections.abc import Sequence

    from daft.dataframe import DataFrame

#: Default episode partitioning for LeRobot-style frame tables.
DEFAULT_EPISODE_KEYS = ("episode_index",)
DEFAULT_ORDER_BY = "frame_index"

_STATE_FRAME_DTYPE = DataType.struct(
    {
        f"{name}_{tag}": dtype
        for tag, _ in HANDS
        for name, dtype in (
            ("curl", DataType.float64()),
            ("pinch", DataType.float64()),
            ("palm_up", DataType.float64()),
            ("wrist_y", DataType.float64()),
            ("wrist", DataType.fixed_size_list(DataType.float64(), 3)),
            ("wrist_rot6d", DataType.fixed_size_list(DataType.float64(), 6)),
        )
    }
)


@daft.func(return_dtype=_STATE_FRAME_DTYPE, use_process=False, unnest=True)
def state_frame_features(state) -> dict[str, object]:
    """Per-frame spatial features from one 48-D hand-state vector.

    The frame-level twin of `daft_physical_ai.pose.state.compute_raw_features`:
    apply it to a frame table's state column (``unnest=True`` spreads the
    per-hand columns), then `add_temporal_features` turns the columns into
    rates — all without leaving the query plan.
    """
    frame = np.asarray(state, dtype=np.float64)[None, :]
    features = state_geometry.compute_raw_features(frame)
    row: dict[str, object] = {}
    for tag, side in HANDS:
        row[f"curl_{tag}"] = float(features[f"curl_{tag}"][0])
        row[f"pinch_{tag}"] = float(features[f"pinch_{tag}"][0])
        row[f"palm_up_{tag}"] = float(features[f"palm_up_{tag}"][0])
        row[f"wrist_y_{tag}"] = float(features[f"wrist_{tag}"][0, 1])
        row[f"wrist_{tag}"] = features[f"wrist_{tag}"][0].tolist()
        row[f"wrist_rot6d_{tag}"] = frame[0, state_geometry.rot6d_slice(side)].tolist()
    return row


@daft.func(return_dtype=DataType.float64(), use_process=False)
def forearm_roll(rot6d, rot6d_next, forearm_axis) -> float:
    """Wrist roll (rad) about the forearm axis from one frame to the next.

    ``rot6d_next`` arrives via ``lead(1)`` over the per-episode window, so it
    is null on each episode's last frame — which maps to a roll of 0 there.
    """
    if rot6d is None or rot6d_next is None:
        return 0.0
    rotations = state_geometry.rotation_from_rot6d(np.asarray([rot6d, rot6d_next], dtype=np.float64))
    delta = rotations[1] @ rotations[0].T
    angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    axis = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ]
    )
    magnitude = np.linalg.norm(axis)
    if magnitude < 1e-9:
        return 0.0
    return float(abs(angle * np.dot(axis / magnitude, np.asarray(forearm_axis, dtype=np.float64))))


def add_temporal_features(
    frames: DataFrame,
    *,
    episode_keys: Sequence[str] = DEFAULT_EPISODE_KEYS,
    order_by: str = DEFAULT_ORDER_BY,
    fps: float = FPS,
) -> DataFrame:
    """Add per-hand action-rate columns to a per-frame feature DataFrame.

    Rates are 0 at each episode's last frame (``lead(1)`` is null there), and
    each rate column is added only when its spatial input columns are present.
    """
    per_episode = Window().partition_by(*episode_keys).order_by(order_by)
    smooth = (
        Window()
        .partition_by(*episode_keys)
        .order_by(order_by)
        .rows_between(-ROLL_SMOOTH_HALF_WIDTH, ROLL_SMOOTH_HALF_WIDTH)
    )
    dt = 1.0 / fps
    columns = set(frames.column_names)

    def rate(name: str) -> daft.Expression:
        return ((col(name).lead(1).over(per_episode) - col(name)) / dt).fill_null(0.0)

    def speed(name: str) -> daft.Expression:
        return (euclidean_distance(col(name), col(name).lead(1).over(per_episode)) / dt).fill_null(0.0)

    df = frames
    for tag, _ in HANDS:
        if f"curl_{tag}" in columns:
            df = df.with_column(f"curl_rate_{tag}", rate(f"curl_{tag}"))
        if f"wrist_y_{tag}" in columns:
            df = df.with_column(f"wrist_vert_vel_{tag}", rate(f"wrist_y_{tag}"))
        if f"arm_extension_{tag}" in columns:
            df = df.with_column(f"arm_ext_rate_{tag}", rate(f"arm_extension_{tag}"))
        if f"wrist_{tag}" in columns:
            df = df.with_column(f"wrist_speed_{tag}", speed(f"wrist_{tag}"))
        if f"local_joints_{tag}" in columns:
            df = df.with_column(f"articulation_{tag}", speed(f"local_joints_{tag}"))
        if f"wrist_rot6d_{tag}" in columns and f"forearm_axis_{tag}" in columns:
            roll_raw = cast(
                "daft.Expression",
                forearm_roll(
                    col(f"wrist_rot6d_{tag}"),
                    col(f"wrist_rot6d_{tag}").lead(1).over(per_episode),
                    col(f"forearm_axis_{tag}"),
                ),
            )
            df = df.with_column(f"roll_raw_{tag}", roll_raw / dt).with_column(
                f"roll_{tag}", col(f"roll_raw_{tag}").mean().over(smooth)
            )
    return df
