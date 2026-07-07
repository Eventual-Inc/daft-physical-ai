"""Scenario queries over episode-level pose-feature tracks.

Each scenario is a NumPy predicate ``(tracks, thresholds) -> (N,) bool mask``,
where ``tracks`` maps unsuffixed feature names (``closure``, ``flex_nonthumb``,
...) to one hand's arrays from
`daft_physical_ai.pose.features.EpisodeFeatureComputer`. Thresholds come from a
one-time percentile calibration over the corpus (`calibrate` /
`calibrate_arrays`), so query runs are reproducible offline; matching frames
stitch into ``[start, end]`` segments with `segments_of`.

Scenario input requirements: ``grasping`` / ``lifting`` need only the
state-derived tracks; ``writing_grip`` / ``hammer_grip`` / ``reaching`` /
``in_hand`` / ``twisting`` / ``openness`` need the skeleton-derived tracks too.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import daft

# Action thresholds (units per second), surfaced so callers can see/tune them:
# grasping = curl closing fast enough, lifting = wrist rising fast enough.
GRASP_RATE = 0.20
LIFT_VEL = 0.20
TWIST_ROLL_RATE = 2.0

# Segment stitching: contiguous matching frames become one [start, end] segment;
# gaps shorter than SEG_GAP_MERGE are bridged, runs shorter than SEG_MIN_FRAMES
# dropped, and at most MAX_SEGS (longest) are kept per episode.
SEG_GAP_MERGE = 5
SEG_MIN_FRAMES = 1
MAX_SEGS = 12

# Data-driven percentiles for the calibrated thresholds.
REACH_RATE_PERCENTILE = 85
WRIST_STILL_PERCENTILE = 30
ARTICULATION_PERCENTILE = 75


# --- scenarios: (tracks, thresholds) -> (N,) bool mask ------------------------


def writing_grip(t, thr):
    """Tripod: thumb on the index/middle tip, those two not fisted, ring+little more curled."""
    flex = t["flex_nonthumb"]
    return (
        (t["thumb_min_tip"] < thr["thumb_on_tip"])
        & (flex[:, 0] < thr["curled_flexion"])
        & (flex[:, 1] < thr["curled_flexion"])
        & (flex[:, 2] > flex[:, 0] + thr["curl_gap"])
        & (flex[:, 3] > flex[:, 1] + thr["curl_gap"])
    )


def hammer_grip(t, thr):
    """Power: all four fingers curled and the thumb wrapped across a proximal knuckle."""
    return (t["flex_nonthumb"] > thr["curled_flexion"]).all(axis=1) & (
        t["thumb_min_knuckle"] < thr["thumb_on_knuckle"]
    )


def twisting(t, thr):
    """Forearm roll past a fixed rate (action)."""
    return t["roll"] > TWIST_ROLL_RATE


def reaching(t, thr):
    """Arm extending faster than the calibrated rate (action)."""
    return t["arm_ext_rate"] >= thr["reach"]


def in_hand(t, thr):
    """Wrist still while fingers move, both vs calibrated thresholds (action)."""
    return (t["wrist_speed"] < thr["still"]) & (t["articulation"] > thr["articulation"])


def grasping(t, thr):
    """Fingers closing fast enough (action): curl rate <= -GRASP_RATE."""
    return t["curl_rate"] <= -GRASP_RATE


def lifting(t, thr):
    """Wrist rising fast enough (action): vertical velocity >= LIFT_VEL."""
    return t["wrist_vert_vel"] >= LIFT_VEL


def openness(t, thr, open_lo=0.0, open_hi=1.0):
    """Openness band (1 = fully open palm), mapped onto the closure track via the
    calibrated closure spread [closure_lo, closure_hi]: higher openness -> lower closure.
    """
    span = (thr["closure_hi"] - thr["closure_lo"]) or 1.0
    clo_lo = thr["closure_hi"] - open_hi * span
    clo_hi = thr["closure_hi"] - open_lo * span
    return (t["closure"] >= clo_lo) & (t["closure"] <= clo_hi)


SCENARIOS: dict[str, Callable[..., np.ndarray]] = {
    "writing_grip": writing_grip,
    "hammer_grip": hammer_grip,
    "twisting": twisting,
    "reaching": reaching,
    "in_hand": in_hand,
    "grasping": grasping,
    "lifting": lifting,
    "openness": openness,
}


def scenario_mask(pose, tracks_by_tag, thresholds=None, *, hand="either", **kwargs) -> np.ndarray:
    """Evaluate one scenario over one episode's per-hand tracks.

    ``pose`` is a `SCENARIOS` key or a ``(tracks, thresholds) -> mask``
    callable; ``tracks_by_tag`` maps ``'L'``/``'R'`` to unsuffixed track dicts;
    ``hand`` is ``'left'`` / ``'right'`` / ``'either'`` (masks OR together).
    """
    scenario = pose if callable(pose) else SCENARIOS[pose]
    tags = {"left": ("L",), "right": ("R",)}.get(hand, ("L", "R"))
    mask = None
    for tag in tags:
        hand_mask = np.asarray(scenario(tracks_by_tag[tag], thresholds or {}, **kwargs), dtype=bool)
        mask = hand_mask if mask is None else (mask | hand_mask)
    return mask


# --- calibration ---------------------------------------------------------------

_CALIBRATION_TRACKS = (
    "arm_ext_rate",
    "wrist_speed",
    "articulation",
    "flex_nonthumb",
    "thumb_min_tip",
    "thumb_min_knuckle",
    "closure",
)


def _pooled_thresholds(pool: Callable[[str, float, bool], float]) -> dict[str, float]:
    return {
        "reach": pool("arm_ext_rate", REACH_RATE_PERCENTILE, False),
        "still": pool("wrist_speed", WRIST_STILL_PERCENTILE, False),
        "articulation": pool("articulation", ARTICULATION_PERCENTILE, False),
        "curled_flexion": pool("flex_nonthumb", 70, True),
        "thumb_on_tip": pool("thumb_min_tip", 25, False),
        "thumb_on_knuckle": pool("thumb_min_knuckle", 15, False),
        "closure_lo": pool("closure", 2, False),
        "closure_hi": pool("closure", 98, False),
        "curl_gap": math.radians(20),
    }


def calibrate_arrays(episodes: Sequence[dict]) -> dict[str, float]:
    """Global scenario thresholds from per-episode track dicts.

    ``episodes`` is a sequence of `EpisodeFeatureComputer.compute` outputs
    (skeleton tracks included). Each track pools over every episode and both
    hands, then takes ``np.percentile``.
    """

    def pool(name: str, percentile: float, explode: bool) -> float:
        arrays = [np.asarray(ep[f"{name}_{tag}"]) for ep in episodes for tag in ("L", "R")]
        values = np.concatenate([a.ravel() if explode else a for a in arrays])
        return float(np.percentile(values, percentile))

    return _pooled_thresholds(pool)


def calibrate(features: daft.DataFrame) -> dict[str, float]:
    """Global scenario thresholds from an episode-level feature DataFrame.

    The DataFrame carries one row per episode with ``{track}_{L|R}`` tensor
    columns (a Daft-materialized form of `EpisodeFeatureComputer` output).
    Compute once when the features are loaded and pass the dict to the
    scenario predicates.
    """
    columns = [f"{name}_{tag}" for name in _CALIBRATION_TRACKS for tag in ("L", "R")]
    data = features.select(*columns).to_pydict()

    def pool(name: str, percentile: float, explode: bool) -> float:
        arrays = [np.asarray(a) for tag in ("L", "R") for a in data[f"{name}_{tag}"]]
        values = np.concatenate([a.ravel() if explode else a for a in arrays])
        return float(np.percentile(values, percentile))

    return _pooled_thresholds(pool)


# --- segment stitching -----------------------------------------------------------


def segments_of(frames, gap_merge: int = SEG_GAP_MERGE, min_frames: int = SEG_MIN_FRAMES):
    """Contiguous matching runs (merging gaps < gap_merge) as [(start, end), ...]."""
    frames = sorted(frames)
    if not frames:
        return []
    runs, start, prev = [], frames[0], frames[0]
    for frame in frames[1:]:
        if frame - prev > gap_merge:
            runs.append((start, prev))
            start = frame
        prev = frame
    runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= min_frames] or runs


def top_segments(frames, max_segments: int = MAX_SEGS):
    """The longest <= max_segments contiguous runs in `frames`, ordered by start time."""
    runs = segments_of(frames)
    return sorted(sorted(runs, key=lambda run: run[1] - run[0], reverse=True)[:max_segments])


__all__ = [
    "GRASP_RATE",
    "LIFT_VEL",
    "MAX_SEGS",
    "SCENARIOS",
    "SEG_GAP_MERGE",
    "TWIST_ROLL_RATE",
    "calibrate",
    "calibrate_arrays",
    "scenario_mask",
    "segments_of",
    "top_segments",
]
