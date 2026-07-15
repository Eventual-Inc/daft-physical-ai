"""Tests for daft_physical_ai.pose — state, skeleton, features, query, temporal."""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

RNG = np.random.default_rng(0)
N = 30  # frames per synthetic episode


def _random_state(n: int = N) -> np.ndarray:
    """Random 48-D two-hand state vector (valid enough for geometry tests)."""
    state = RNG.standard_normal((n, 48)).astype(np.float64)
    # Normalise the rot6d columns so rotation_from_rot6d produces sane matrices.
    for base in (0, 24):
        rot6d = state[:, base + 3 : base + 9]
        col0 = rot6d[:, :3]
        col0 /= np.linalg.norm(col0, axis=1, keepdims=True) + 1e-9
        rot6d[:, :3] = col0
    return state


def _random_skeleton(n: int = N) -> np.ndarray:
    """Random 204-D body-skeleton vector (68 joints x xyz)."""
    return RNG.standard_normal((n, 204)).astype(np.float64)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


def test_rot6d_slice():
    from daft_physical_ai.pose.state import rot6d_slice

    assert rot6d_slice("left") == slice(3, 9)
    assert rot6d_slice("right") == slice(27, 33)


def test_rotation_from_rot6d_orthonormal():
    from daft_physical_ai.pose.state import rotation_from_rot6d

    rot6d = _random_state()[:, 3:9]
    R = rotation_from_rot6d(rot6d)
    assert R.shape == (N, 3, 3)
    # Each column should have unit norm
    col_norms = np.linalg.norm(R, axis=1)  # (N, 3)
    np.testing.assert_allclose(col_norms, np.ones((N, 3)), atol=1e-6)
    # Columns should be mutually orthogonal
    dot_01 = (R[:, :, 0] * R[:, :, 1]).sum(1)
    dot_02 = (R[:, :, 0] * R[:, :, 2]).sum(1)
    dot_12 = (R[:, :, 1] * R[:, :, 2]).sum(1)
    np.testing.assert_allclose(dot_01, 0.0, atol=1e-6)
    np.testing.assert_allclose(dot_02, 0.0, atol=1e-6)
    np.testing.assert_allclose(dot_12, 0.0, atol=1e-6)


def test_palm_normal_from_rot6d_unit():
    from daft_physical_ai.pose.state import palm_normal_from_rot6d

    rot6d = _random_state()[:, 3:9]
    normals = palm_normal_from_rot6d(rot6d)
    assert normals.shape == (N, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), np.ones(N), atol=1e-9)


def test_compute_raw_features_keys():
    from daft_physical_ai.pose.state import compute_raw_features

    state = _random_state()
    features = compute_raw_features(state)
    for tag in ("L", "R"):
        assert f"curl_{tag}" in features
        assert f"pinch_{tag}" in features
        assert f"wrist_{tag}" in features
        assert features[f"curl_{tag}"].shape == (N,)
        assert features[f"wrist_{tag}"].shape == (N, 3)


# --------------------------------------------------------------------------- #
# skeleton
# --------------------------------------------------------------------------- #


def test_joint_names_count():
    from daft_physical_ai.pose.skeleton import JOINT_NAMES

    assert len(JOINT_NAMES) == 68


def test_joint_position_shape():
    from daft_physical_ai.pose.skeleton import joint_position

    skel = _random_skeleton()
    pos = joint_position(skel, "leftHand")
    assert pos.shape == (N, 3)


def test_finger_flexion_shape():
    from daft_physical_ai.pose.skeleton import finger_flexion

    skel = _random_skeleton()
    # Thumb has 2 joints, others have 3
    assert finger_flexion(skel, "left", "Thumb").shape == (N, 2)
    assert finger_flexion(skel, "left", "Index").shape == (N, 3)


def test_palm_normal_unit():
    from daft_physical_ai.pose.skeleton import palm_normal

    skel = _random_skeleton()
    n = palm_normal(skel, "left")
    assert n.shape == (N, 3)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), np.ones(N), atol=1e-6)


def test_compute_state_features_keys():
    from daft_physical_ai.pose.skeleton import compute_state_features

    skel = _random_skeleton()
    features = compute_state_features(skel)
    for tag in ("L", "R"):
        assert f"closure_{tag}" in features
        assert f"flex_nonthumb_{tag}" in features
        assert features[f"flex_nonthumb_{tag}"].shape == (N, 4)
        assert f"arm_extension_{tag}" in features
        assert f"local_joints_{tag}" in features


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def test_temporal_forward_rate_zero_at_last():
    from daft_physical_ai.pose.features import TemporalFeatureComputer

    tc = TemporalFeatureComputer()
    values = np.arange(N, dtype=np.float64)
    rates = tc.forward_rate(values)
    assert rates[-1] == 0.0
    np.testing.assert_allclose(rates[:-1], np.full(N - 1, 30.0))  # FPS=30, diff=1


def test_temporal_forward_speed_zero_at_last():
    from daft_physical_ai.pose.features import TemporalFeatureComputer

    tc = TemporalFeatureComputer()
    points = np.zeros((N, 3))
    points[:, 0] = np.arange(N)  # moving along x
    speeds = tc.forward_speed(points)
    assert speeds[-1] == 0.0
    np.testing.assert_allclose(speeds[:-1], np.full(N - 1, 30.0))


def test_episode_feature_computer_state_only():
    from daft_physical_ai.pose.features import STATE_TRACKS, EpisodeFeatureComputer

    state = _random_state()
    computer = EpisodeFeatureComputer()
    tracks = computer.compute(state=state)

    assert tracks["num_frames"] == N
    for tag in ("L", "R"):
        for name in STATE_TRACKS:
            key = f"{name}_{tag}"
            assert key in tracks, f"missing {key}"
            arr = np.asarray(tracks[key])
            assert arr.shape[0] == N, f"{key} shape mismatch"


def test_episode_feature_computer_with_skeleton():
    from daft_physical_ai.pose.features import SKELETON_TRACKS, EpisodeFeatureComputer

    state = _random_state()
    skel = _random_skeleton()
    computer = EpisodeFeatureComputer()
    tracks = computer.compute(state=state, skeleton=skel)

    for tag in ("L", "R"):
        for name in SKELETON_TRACKS:
            assert f"{name}_{tag}" in tracks


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #


def test_segments_of_basic():
    from daft_physical_ai.pose.query import segments_of

    assert segments_of([]) == []
    assert segments_of([1, 2, 3]) == [(1, 3)]
    # Gap of 1 within default gap_merge=5 → merged
    assert segments_of([1, 2, 4, 5]) == [(1, 5)]
    # Gap larger than gap_merge → split
    assert segments_of([1, 2, 20, 21], gap_merge=5) == [(1, 2), (20, 21)]


def test_top_segments_ordering():
    from daft_physical_ai.pose.query import top_segments

    frames = list(range(10)) + list(range(20, 25))
    segs = top_segments(frames, max_segments=2)
    # Longest run first by length, result sorted by start
    assert segs[0][0] < segs[1][0]


def test_grasping_scenario():
    from daft_physical_ai.pose.features import EpisodeFeatureComputer
    from daft_physical_ai.pose.query import GRASP_RATE, scenario_mask

    state = _random_state()
    computer = EpisodeFeatureComputer()
    tracks = computer.compute(state=state)
    tracks_by_tag = {
        tag: {name.rsplit(f"_{tag}", 1)[0]: np.asarray(tracks[name]) for name in tracks if name.endswith(f"_{tag}")}
        for tag in ("L", "R")
    }
    mask = scenario_mask("grasping", tracks_by_tag)
    assert mask.shape == (N,)
    assert mask.dtype == bool
    # Frames with curl_rate <= -GRASP_RATE should match
    expected = (np.asarray(tracks["curl_rate_L"]) <= -GRASP_RATE) | (np.asarray(tracks["curl_rate_R"]) <= -GRASP_RATE)
    np.testing.assert_array_equal(mask, expected)


def test_calibrate_arrays():
    from daft_physical_ai.pose.features import EpisodeFeatureComputer
    from daft_physical_ai.pose.query import calibrate_arrays

    computer = EpisodeFeatureComputer()
    episodes = [computer.compute(state=_random_state(), skeleton=_random_skeleton()) for _ in range(3)]
    thresholds = calibrate_arrays(episodes)
    assert "reach" in thresholds
    assert "still" in thresholds
    assert all(isinstance(v, float) for v in thresholds.values())


# --------------------------------------------------------------------------- #
# temporal (plan-build only — no real data needed)
# --------------------------------------------------------------------------- #


def test_state_frame_features_is_callable():
    """state_frame_features is a @daft.func — verify it's importable and callable as an expression builder."""
    from daft_physical_ai.pose.temporal import state_frame_features

    assert callable(state_frame_features)


def test_add_temporal_features_plan():
    """add_temporal_features builds a valid lazy plan without collecting."""
    import daft
    from daft import DataType, col
    from daft_physical_ai.pose.temporal import add_temporal_features, state_frame_features

    state_col = DataType.fixed_size_list(DataType.float64(), 48)
    df = daft.from_pydict(
        {
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "observation.state": [[0.0] * 48] * 5,
        }
    ).with_column("observation.state", col("observation.state").cast(state_col))

    frames = df.select("episode_index", "frame_index", state_frame_features(col("observation.state")))
    rates = add_temporal_features(frames)
    # Should build without error; verify rate columns present
    assert "curl_rate_L" in rates.column_names
    assert "wrist_vert_vel_R" in rates.column_names
