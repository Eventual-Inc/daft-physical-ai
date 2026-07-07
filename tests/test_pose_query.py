from __future__ import annotations

import numpy as np

from daft_physical_ai.pose import calibrate_arrays, scenario_mask, segments_of, top_segments
from daft_physical_ai.pose.query import GRASP_RATE, LIFT_VEL


def make_tracks(**overrides) -> dict[str, np.ndarray]:
    n = 6
    tracks = {
        "closure": np.full(n, 0.5),
        "flex_nonthumb": np.full((n, 4), 0.5),
        "thumb_min_tip": np.full(n, 1.0),
        "thumb_min_knuckle": np.full(n, 1.0),
        "curl_rate": np.zeros(n),
        "wrist_vert_vel": np.zeros(n),
        "arm_ext_rate": np.zeros(n),
        "wrist_speed": np.zeros(n),
        "articulation": np.zeros(n),
        "roll": np.zeros(n),
    }
    tracks.update(overrides)
    return tracks


def test_grasping_and_lifting_fire_on_rates() -> None:
    grasp_track = np.array([0.0, -GRASP_RATE - 0.1, -GRASP_RATE - 0.1, 0.0, 0.0, 0.0])
    lift_track = np.array([0.0, 0.0, 0.0, LIFT_VEL + 0.1, LIFT_VEL + 0.1, 0.0])
    tracks_by_tag = {
        "L": make_tracks(curl_rate=grasp_track, wrist_vert_vel=lift_track),
        "R": make_tracks(),
    }

    grasp_mask = scenario_mask("grasping", tracks_by_tag)
    lift_mask_left = scenario_mask("lifting", tracks_by_tag, hand="left")
    lift_mask_right = scenario_mask("lifting", tracks_by_tag, hand="right")

    assert grasp_mask.tolist() == [False, True, True, False, False, False]
    assert lift_mask_left.tolist() == [False, False, False, True, True, False]
    assert not lift_mask_right.any()


def test_hammer_grip_needs_curl_and_thumb_wrap() -> None:
    thresholds = {"curled_flexion": 1.0, "thumb_on_knuckle": 0.3}
    wrapped = make_tracks(flex_nonthumb=np.full((6, 4), 1.5), thumb_min_knuckle=np.full(6, 0.1))
    open_hand = make_tracks()
    mask = scenario_mask("hammer_grip", {"L": wrapped, "R": open_hand}, thresholds)

    assert mask.all()
    assert not scenario_mask("hammer_grip", {"L": open_hand, "R": open_hand}, thresholds).any()


def test_writing_grip_tripod_shape() -> None:
    thresholds = {"thumb_on_tip": 0.5, "curled_flexion": 1.0, "curl_gap": 0.3}
    flex = np.tile([0.2, 0.2, 0.9, 0.9], (6, 1))  # index/middle straight, ring/little curled
    tripod = make_tracks(thumb_min_tip=np.full(6, 0.1), flex_nonthumb=flex)

    assert scenario_mask("writing_grip", {"L": tripod, "R": make_tracks()}, thresholds).all()


def test_calibrate_arrays_pools_hands_and_episodes() -> None:
    episodes = [
        {
            f"{name}_{tag}": np.linspace(0, 1, 11)
            for tag in ("L", "R")
            for name in (
                "arm_ext_rate",
                "wrist_speed",
                "articulation",
                "thumb_min_tip",
                "thumb_min_knuckle",
                "closure",
            )
        }
        | {f"flex_nonthumb_{tag}": np.tile(np.linspace(0, 1, 11)[:, None], (1, 4)) for tag in ("L", "R")}
    ]
    thresholds = calibrate_arrays(episodes)

    assert 0.8 <= thresholds["reach"] <= 0.9  # ~85th percentile of the 0..1 ramp
    assert 0.25 <= thresholds["still"] <= 0.35
    assert thresholds["closure_lo"] <= 0.05
    assert thresholds["closure_hi"] >= 0.95
    assert thresholds["curl_gap"] > 0


def test_segments_merge_gaps_and_rank_by_length() -> None:
    frames = [0, 1, 2, 10, 11, 30, 31, 32, 33]

    assert segments_of(frames) == [(0, 2), (10, 11), (30, 33)]
    assert segments_of([]) == []
    assert top_segments(frames, max_segments=2) == [(0, 2), (30, 33)]
