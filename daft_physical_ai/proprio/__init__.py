"""Proprioception signals for Daft DataFrames.

Everything here reads the robot's own state columns - joint positions, gripper
position - which LeRobot stores in parquet next to the video. Nothing decodes a
frame, so finding the still parts of an episode costs a columnar scan rather
than a video pass.

`motion_energy(...)` returns a per-frame column: how much the arm moved since
the previous frame. `is_active(...)` thresholds it into a boolean, requiring a
short run so a single noisy frame doesn't count as motion. Both are plain Daft
expressions and compose into any pipeline.
"""

from __future__ import annotations

import daft
from daft import DataFrame, Expression, Window, col
from daft.functions import lag

__all__ = ["is_active", "motion_energy", "motion_scale"]

DEFAULT_THRESHOLD = 0.1  # in units of a typical frame-to-frame step
DEFAULT_RUN = 3  # consecutive frames over the threshold before motion counts


def motion_scale(
    frames: DataFrame,
    state: str,
    *,
    dims: int,
    episode: str = "episode_index",
) -> list[float]:
    """Measure the typical per-dim step size, for normalizing motion deltas.

    One pass over ``frames``. This is the root mean square of the per-frame
    *deltas* - not of the positions, since normalizing by how far a joint ranges
    across the dataset makes ordinary slow motion look like stillness and
    over-trims badly. RMS rather than standard deviation because we want typical
    step magnitude: an arm moving at a constant rate has a large typical step
    but zero variation in it, and standard deviation would call that stillness.

    Kept separate from :func:`motion_energy` so that function stays a pure
    expression builder - it takes the result as ``scale``.

    Args:
        frames: frame-level DataFrame (e.g. from ``lerobot.load_episode_frames``).
        state: name of the state column, a fixed-size list of floats per frame.
        dims: how many dims that list holds.
        episode: column identifying the episode, so deltas don't cross boundaries.

    Returns:
        One positive float per dim, in column order.
    """
    deltas = _deltas(col(state), dims=dims, episode=episode)
    named = {f"_scale_{i}": d * d for i, d in enumerate(deltas)}
    scoped = frames
    for name, squared in named.items():
        scoped = scoped.with_column(name, squared)
    means = scoped.agg(*[col(n).mean().alias(n) for n in named]).to_pydict()
    return [max((means[n][0] or 0.0) ** 0.5, 1e-9) for n in named]


def motion_energy(
    state: Expression,
    *,
    dims: int,
    scale: list[float],
    episode: str = "episode_index",
    order: str = "frame_index",
) -> Expression:
    """How much the arm moved since the previous frame, as one number per frame.

    Each dim of the state list is diffed against its own previous value inside
    the episode, divided by that dim's typical step (see :func:`motion_scale`)
    so a quick wrist joint and a slow shoulder joint count comparably, then
    combined as a Euclidean norm. Idle frames land near zero; real motion sits
    orders of magnitude above.

    The first frame of each episode has no predecessor, so its energy is 0.

    Args:
        state: the state column (a fixed-size list of floats per frame).
        dims: how many dims that list holds.
        scale: per-dim typical step, from :func:`motion_scale`.
        episode: column to partition by, so deltas don't cross episode boundaries.
        order: column to order by within an episode.

    Returns:
        A float expression, one value per frame.
    """
    if len(scale) != dims:
        raise ValueError(f"scale has {len(scale)} entries, expected dims={dims}")

    deltas = _deltas(state, dims=dims, episode=episode, order=order)
    squares = [(d / s) * (d / s) for d, s in zip(deltas, scale)]
    total = squares[0]
    for square in squares[1:]:
        total = total + square
    return total.sqrt().fill_null(0.0)


def is_active(
    energy: Expression,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    run: int = DEFAULT_RUN,
    episode: str = "episode_index",
    order: str = "frame_index",
) -> Expression:
    """Whether a frame belongs to a stretch of real motion.

    A single frame over the threshold is noise - sensor jitter, a rounding
    artifact - so a frame only counts as active when it sits inside a run of
    ``run`` consecutive frames that all clear the threshold.

    Args:
        energy: a motion-energy column, from :func:`motion_energy`.
        threshold: in units of a typical frame-to-frame step.
        run: how many consecutive frames must clear the threshold.
        episode: column to partition by.
        order: column to order by within an episode.

    Returns:
        A boolean expression, one value per frame.
    """
    if run < 1:
        raise ValueError(f"run must be >= 1, got {run}")

    over = (energy > threshold).cast(daft.DataType.int64())  # ty: ignore[unsupported-operator]
    if run == 1:
        return over == 1

    ordered = Window().partition_by(episode).order_by(order)
    # A frame is active if any run-length window covering it is entirely over the
    # threshold. Sliding the window across all `run` offsets covers the interior
    # of every run as well as its two edges.
    covering = [
        over.sum().over(ordered.rows_between(-offset, run - 1 - offset, min_periods=run)) == run
        for offset in range(run)
    ]
    within = covering[0]
    for other in covering[1:]:
        within = within | other
    return within.fill_null(False)


def _deltas(
    state: Expression,
    *,
    dims: int,
    episode: str = "episode_index",
    order: str = "frame_index",
) -> list[Expression]:
    """Per-dim change since the previous frame, within the episode.

    Partitioning by episode matters: without it an episode's first frame diffs
    against the last frame of the previous episode, and every boundary shows a
    spike that isn't there.
    """
    window = Window().partition_by(episode).order_by(order)
    return [state.get(i) - lag(state.get(i), 1).over(window) for i in range(dims)]
