# 04 - Episode operations

Packaged robotics operations that understand trajectory semantics, run over
episode tables.

- [`hand_tracking/`](hand_tracking/) - `track_hands` (MediaPipe CPU / WiLoR
  GPU) on an EgoDex sample, with rendered keypoints and PCK scoring against
  ground truth.

Planned: `motion_trim.py` (drop idle prefixes/suffixes from action/state
deltas), `pose_query_segments.py` (scenario predicates like grasping/lifting
over pose features).
