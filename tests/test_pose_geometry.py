from __future__ import annotations

import numpy as np
import pytest

from daft_physical_ai.pose import (
    JOINT_NAMES,
    EpisodeFeatureComputer,
    TemporalFeatureComputer,
    arm_extension,
    compute_raw_features,
    compute_state_features,
    finger_flexion,
    hand_scale,
    joint_position,
    palm_normal_from_rot6d,
    rotation_from_rot6d,
)
from daft_physical_ai.pose.features import SKELETON_TRACKS, STATE_TRACKS
from daft_physical_ai.pose.skeleton import JOINT_INDEX


def make_state(*, curled: bool = False, pinched: bool = False, wrist_y: float = 1.0, n: int = 4) -> np.ndarray:
    """A synthetic 48-D state: identity wrist rotation, fingertips on a spread or curled."""
    state = np.zeros((n, 48))
    for base, x_sign in ((0, -1.0), (24, 1.0)):
        state[:, base : base + 3] = [x_sign * 0.3, wrist_y, 0.4]  # wrist
        state[:, base + 3 : base + 9] = [1, 0, 0, 0, 1, 0]  # identity rot6d
        for finger in range(5):
            offset = base + 9 + finger * 3
            reach = 0.02 if curled else 0.10
            state[:, offset : offset + 3] = [
                x_sign * 0.3 + reach * (finger + 1) / 5,
                wrist_y,
                0.4 + reach,
            ]
        if pinched:  # thumb tip == index tip
            state[:, base + 12 : base + 15] = state[:, base + 9 : base + 12]
    return state


def test_rot6d_identity_gives_z_palm_normal() -> None:
    rot6d = np.array([[1.0, 0, 0, 0, 1.0, 0]])

    assert np.allclose(palm_normal_from_rot6d(rot6d), [[0, 0, 1]])
    assert np.allclose(rotation_from_rot6d(rot6d)[0], np.eye(3))


def test_rot6d_orthonormalizes_unnormalized_input() -> None:
    rot6d = np.array([[2.0, 0, 0, 1.0, 3.0, 0]])  # scaled + non-orthogonal
    rotation = rotation_from_rot6d(rot6d)[0]

    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)


def test_curl_orders_open_vs_curled_and_pinch() -> None:
    open_features = compute_raw_features(make_state())
    curled_features = compute_raw_features(make_state(curled=True))
    pinched_features = compute_raw_features(make_state(pinched=True))

    for tag in ("L", "R"):
        assert curled_features[f"curl_{tag}"].mean() < open_features[f"curl_{tag}"].mean()
        assert pinched_features[f"pinch_{tag}"].max() < 1e-9
        assert open_features[f"pinch_{tag}"].min() > 0.01
        assert np.allclose(open_features[f"palm_up_{tag}"], 0.0)  # identity rot: normal is +z


def make_skeleton(*, curled: bool = False, arm_straight: bool = True, n: int = 3) -> np.ndarray:
    """A synthetic 204-D skeleton with parameterized finger curl and arm extension."""
    skeleton = np.zeros((n, 204))

    def put(name: str, xyz) -> None:
        start = JOINT_INDEX[name] * 3
        skeleton[:, start : start + 3] = xyz

    for side, x_sign in (("left", -1.0), ("right", 1.0)):
        shoulder_x = x_sign * 0.2
        put(side + "Shoulder", [shoulder_x, 1.4, 0.0])
        if arm_straight:
            put(side + "Arm", [shoulder_x, 1.4, 0.25])
            put(side + "Forearm", [shoulder_x, 1.4, 0.50])
            put(side + "Hand", [shoulder_x, 1.4, 0.75])
        else:
            put(side + "Arm", [shoulder_x, 1.2, 0.15])
            put(side + "Forearm", [shoulder_x, 1.0, 0.05])
            put(side + "Hand", [shoulder_x, 1.15, 0.15])

        hand_start = JOINT_INDEX[side + "Hand"] * 3
        wrist = skeleton[0, hand_start : hand_start + 3].copy()
        for finger_index, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Little")):
            infix = "" if finger == "Thumb" else "Finger"
            parts = (
                ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
                if finger == "Thumb"
                else ["Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
            )
            spread = (finger_index - 2) * 0.02
            for i, part in enumerate(parts):
                if curled and part in ("IntermediateTip", "Tip"):
                    # fold the distal joints back toward the wrist
                    position = wrist + [x_sign * spread, -0.02 * i, 0.04 * (len(parts) - i) / len(parts)]
                else:
                    position = wrist + [x_sign * spread, 0.0, 0.03 * (i + 1)]
                put(f"{side}{finger}{infix}{part}", position)
    return skeleton


def test_joint_catalog_is_204_dims() -> None:
    assert len(JOINT_NAMES) == 68
    skeleton = make_skeleton()
    assert skeleton.shape[1] == 68 * 3
    assert np.allclose(joint_position(skeleton, "leftHand"), joint_position(skeleton, "leftHand"))


def test_finger_flexion_straight_vs_curled() -> None:
    straight = finger_flexion(make_skeleton(), "right", "Index").sum(1)
    curled = finger_flexion(make_skeleton(curled=True), "right", "Index").sum(1)

    assert straight.max() < 1e-6  # collinear bones: no flexion
    assert curled.min() > 0.5


def test_arm_extension_straight_vs_bent() -> None:
    straight = arm_extension(make_skeleton(arm_straight=True), "left")
    bent = arm_extension(make_skeleton(arm_straight=False), "left")

    assert straight.min() > 0.99
    assert bent.max() < 0.9
    assert hand_scale(make_skeleton(), "left").min() > 0


def test_compute_state_features_emits_both_hands() -> None:
    features = compute_state_features(make_skeleton())

    for tag in ("L", "R"):
        assert features[f"flex_nonthumb_{tag}"].shape == (3, 4)
        assert features[f"closure_{tag}"].shape == (3,)
        assert features[f"forearm_axis_{tag}"].shape == (3, 3)


def test_temporal_rates_on_ramps() -> None:
    temporal = TemporalFeatureComputer(fps=10.0)

    ramp = np.arange(5, dtype=np.float64)  # +1 per frame at 10 fps -> rate 10/s
    rates = temporal.forward_rate(ramp)
    assert np.allclose(rates[:-1], 10.0)
    assert rates[-1] == 0.0

    points = np.stack([ramp, np.zeros(5), np.zeros(5)], axis=1)
    speeds = temporal.forward_speed(points)
    assert np.allclose(speeds[:-1], 10.0)

    assert np.allclose(temporal.centered_mean(np.full(7, 3.0)), 3.0)


def test_forearm_roll_rate_detects_twist() -> None:
    temporal = TemporalFeatureComputer(fps=30.0)
    n, per_frame_angle = 6, 0.1
    angles = np.arange(n) * per_frame_angle
    # rotate about the x axis; rot6d columns are the matrix's first two columns
    rot6d = np.stack(
        [np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), np.cos(angles), np.sin(angles)],
        axis=1,
    )
    forearm = np.tile([1.0, 0.0, 0.0], (n, 1))

    rates = temporal.forearm_roll_rates(rot6d, forearm)

    assert np.allclose(rates[:-1], per_frame_angle * 30.0, atol=1e-6)


def test_episode_feature_computer_state_only_and_full() -> None:
    computer = EpisodeFeatureComputer()
    state = make_state(n=5)

    state_only = computer.compute(state=state)
    assert state_only["num_frames"] == 5
    for name in STATE_TRACKS:
        assert f"{name}_L" in state_only and f"{name}_R" in state_only
    assert "closure_L" not in state_only

    full = computer.compute(state=state, skeleton=make_skeleton(n=5))
    for name in STATE_TRACKS + SKELETON_TRACKS:
        assert f"{name}_L" in full
    assert full["flex_nonthumb_R"].shape == (5, 4)


def test_state_shape_is_validated() -> None:
    with pytest.raises((IndexError, ValueError)):
        compute_raw_features(np.zeros((4, 10)))
