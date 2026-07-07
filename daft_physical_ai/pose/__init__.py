"""Model-free pose geometry, features, and scenario queries for hand-centric data.

Ported from the daft-examples EgoDex pipeline, keeping its clean seams:

- `state` / `skeleton` - pure-NumPy per-frame geometry over two array
  conventions: the 48-D hand state (wrist xyz + rot6d + 5 fingertips per hand)
  and the 204-D body skeleton (68 named joints x xyz).
- `features` - episode-level track assembly: run the geometry once over a
  whole (N, ...) episode and take plain forward differences for rates. No
  frame explode, no window functions.
- `query` - scenario predicates as ``(tracks, thresholds) -> (N,) bool mask``
  callables (writing grip, hammer grip, grasping, lifting, ...), percentile
  calibration over a corpus, and segment stitching.

Dataset adapters produce the arrays (a LeRobot ``observation.state`` column,
raw HDF5 transforms, a MANO fit); everything here is NumPy in, NumPy out, with
Daft touching only schema types.
"""

from __future__ import annotations

from .features import (
    FPS,
    POSE_FEATURES_DTYPE,
    STATE_TRACKS,
    EpisodeFeatureComputer,
    TemporalFeatureComputer,
)
from .query import (
    SCENARIOS,
    calibrate,
    calibrate_arrays,
    scenario_mask,
    segments_of,
    top_segments,
)
from .skeleton import (
    JOINT_NAMES,
    arm_extension,
    compute_state_features,
    finger_flexion,
    forearm_axis,
    hand_local_joints,
    hand_scale,
    joint_position,
    palm_normal,
)
from .state import (
    compute_raw_features,
    palm_normal_from_rot6d,
    rot6d_slice,
    rotation_from_rot6d,
)

__all__ = [
    "FPS",
    "JOINT_NAMES",
    "POSE_FEATURES_DTYPE",
    "SCENARIOS",
    "STATE_TRACKS",
    "EpisodeFeatureComputer",
    "TemporalFeatureComputer",
    "arm_extension",
    "calibrate",
    "calibrate_arrays",
    "compute_raw_features",
    "compute_state_features",
    "finger_flexion",
    "forearm_axis",
    "hand_local_joints",
    "hand_scale",
    "joint_position",
    "palm_normal",
    "palm_normal_from_rot6d",
    "rot6d_slice",
    "rotation_from_rot6d",
    "scenario_mask",
    "segments_of",
    "top_segments",
]
