"""Episode-level pose-feature tracks, computed in one pass per episode.

The pattern (ported from the daft-examples EgoDex pipeline): instead of
exploding episodes into per-frame rows, running rowwise geometry at N=1, and
differentiating with window functions, run the vectorized geometry libraries
(`daft_physical_ai.pose.state`, `daft_physical_ai.pose.skeleton`) once over the
whole (N, ...) episode and take plain NumPy forward differences for the rates.
Same feature definitions, no explode and no windows.

Feature tracks per hand (tag ``L``/``R``):

    from the 48-D state alone -
    curl             mean fingertip-to-wrist distance (small = curled)
    pinch            thumb-index tip distance (precision grip)
    palm_up          +y component of the palm normal
    curl_rate        d(curl)/dt        (grasping)
    wrist_vert_vel   d(wrist y)/dt     (lifting)
    wrist_speed      |d(wrist)/dt|     (stillness)

    additionally, with the 204-D skeleton -
    closure          mean finger flexion (low = open palm, high = fist)
    flex_nonthumb    (N, 4) per-finger flexion for index..little
    thumb_min_tip    thumb tip -> nearest of index/middle tip (writing grip)
    thumb_min_knuckle thumb tip -> nearest of index/middle knuckle (hammer grip)
    arm_ext_rate     d(arm extension)/dt (reaching)
    articulation     |d(hand-local joints)/dt| (in-hand manipulation)
    roll             wrist roll rate about the forearm axis, smoothed (twisting)

Dataset adapters own array construction (e.g. raw EgoDex HDF5 transforms ->
state/skeleton arrays lives in daft-examples); this module starts at arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from daft import DataType

from . import skeleton as skeleton_geometry
from . import state as state_geometry

FPS = 30.0
HANDS = (("L", "left"), ("R", "right"))

# Rolling-mean half-width for the roll track (matches a
# Window().rows_between(-2, 2) smoothing, including shrunken edge windows).
ROLL_SMOOTH_HALF_WIDTH = 2


@dataclass(frozen=True)
class TemporalFeatureComputer:
    """Compute per-frame rates from episode-length feature tracks."""

    fps: float = FPS
    roll_smooth_half_width: int = ROLL_SMOOTH_HALF_WIDTH

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def forward_rate(self, values: np.ndarray) -> np.ndarray:
        """(next - current) / dt per frame, 0 at the episode's last frame."""
        rates = np.zeros(len(values), dtype=np.float64)
        if len(values) > 1:
            rates[:-1] = np.diff(values, axis=0) / self.dt
        return rates

    def forward_speed(self, points: np.ndarray) -> np.ndarray:
        """|next - current| / dt per frame over (N, d) points, 0 at the last frame."""
        speeds = np.zeros(len(points), dtype=np.float64)
        if len(points) > 1:
            speeds[:-1] = np.linalg.norm(np.diff(points, axis=0), axis=1) / self.dt
        return speeds

    def centered_mean(self, values: np.ndarray) -> np.ndarray:
        """Centered rolling mean with shrinking edge windows."""
        kernel = np.ones(2 * self.roll_smooth_half_width + 1)
        sums = np.convolve(values, kernel, mode="same")
        counts = np.convolve(np.ones_like(values), kernel, mode="same")
        return sums / counts

    def forearm_roll_rates(self, rot6d: np.ndarray, forearm_axis: np.ndarray) -> np.ndarray:
        """Wrist roll rate (rad/s) about the forearm axis, per frame."""
        n = len(rot6d)
        rates = np.zeros(n, dtype=np.float64)
        if n < 2:
            return rates
        rotations = state_geometry.rotation_from_rot6d(np.asarray(rot6d, dtype=np.float64))
        relative = np.einsum("nij,nkj->nik", rotations[1:], rotations[:-1])
        angles = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
        axes = np.stack(
            [
                relative[:, 2, 1] - relative[:, 1, 2],
                relative[:, 0, 2] - relative[:, 2, 0],
                relative[:, 1, 0] - relative[:, 0, 1],
            ],
            axis=1,
        )
        magnitudes = np.linalg.norm(axes, axis=1)
        safe = magnitudes > 1e-9
        projected = np.zeros(n - 1, dtype=np.float64)
        projected[safe] = np.abs(
            angles[safe] * np.einsum("nd,nd->n", axes[safe] / magnitudes[safe, None], forearm_axis[:-1][safe])
        )
        rates[:-1] = projected / self.dt
        return rates


_TRACK = DataType.tensor(DataType.float32())

#: Tracks computable from the 48-D state alone.
STATE_TRACKS = (
    "curl",
    "pinch",
    "palm_up",
    "curl_rate",
    "wrist_vert_vel",
    "wrist_speed",
)

#: Tracks that additionally need the 204-D skeleton.
SKELETON_TRACKS = (
    "closure",
    "thumb_min_tip",
    "thumb_min_knuckle",
    "arm_ext_rate",
    "articulation",
    "roll",
)


def _features_dtype() -> DataType:
    fields: dict[str, DataType] = {"num_frames": DataType.int64()}
    for tag, _ in HANDS:
        for name in STATE_TRACKS + SKELETON_TRACKS:
            fields[f"{name}_{tag}"] = _TRACK  # (N,)
        fields[f"flex_nonthumb_{tag}"] = _TRACK  # (N, 4)
    dtype: DataType = DataType.struct(fields)
    return dtype


#: Struct dtype of the full track set (state + skeleton), for Daft UDF wrappers.
POSE_FEATURES_DTYPE = _features_dtype()


@dataclass(frozen=True)
class EpisodeFeatureComputer:
    """Assemble queryable pose-feature tracks from one episode's arrays.

    ``compute(state=...)`` yields the state-only tracks; adding
    ``skeleton=...`` yields the full set the scenario queries consume.
    """

    temporal: TemporalFeatureComputer = field(default_factory=TemporalFeatureComputer)

    def compute(
        self,
        *,
        state: np.ndarray,
        skeleton: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Per-hand feature tracks for one episode, keyed ``{track}_{L|R}``."""
        state = np.asarray(state, dtype=np.float64)
        state_features = state_geometry.compute_raw_features(state)
        skeleton_features = (
            None
            if skeleton is None
            else skeleton_geometry.compute_state_features(np.asarray(skeleton, dtype=np.float64))
        )

        out: dict[str, object] = {"num_frames": len(state)}
        for tag, side in HANDS:
            wrist = state_features[f"wrist_{tag}"]
            tracks: dict[str, np.ndarray] = {
                "curl": state_features[f"curl_{tag}"],
                "pinch": state_features[f"pinch_{tag}"],
                "palm_up": state_features[f"palm_up_{tag}"],
                "curl_rate": self.temporal.forward_rate(state_features[f"curl_{tag}"]),
                "wrist_vert_vel": self.temporal.forward_rate(wrist[:, 1]),
                "wrist_speed": self.temporal.forward_speed(wrist),
            }
            if skeleton_features is not None:
                thumb_tip = skeleton_features[f"thumb_tip_dist_{tag}"]
                thumb_knuckle = skeleton_features[f"thumb_knuckle_dist_{tag}"]
                local_joints = skeleton_features[f"local_joints_{tag}"].reshape(len(state), -1)
                rot6d = state[:, state_geometry.rot6d_slice(side)]
                tracks.update(
                    {
                        "closure": skeleton_features[f"closure_{tag}"],
                        "flex_nonthumb": skeleton_features[f"flex_nonthumb_{tag}"],
                        "thumb_min_tip": thumb_tip[:, :2].min(axis=1),
                        "thumb_min_knuckle": thumb_knuckle[:, :2].min(axis=1),
                        "arm_ext_rate": self.temporal.forward_rate(skeleton_features[f"arm_extension_{tag}"]),
                        "articulation": self.temporal.forward_speed(local_joints),
                        "roll": self.temporal.centered_mean(
                            self.temporal.forearm_roll_rates(rot6d, skeleton_features[f"forearm_axis_{tag}"])
                        ),
                    }
                )
            out.update({f"{name}_{tag}": values.astype(np.float32) for name, values in tracks.items()})
        return out


__all__ = [
    "FPS",
    "POSE_FEATURES_DTYPE",
    "SKELETON_TRACKS",
    "STATE_TRACKS",
    "EpisodeFeatureComputer",
    "TemporalFeatureComputer",
]
