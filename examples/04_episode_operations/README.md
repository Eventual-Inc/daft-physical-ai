# 04 - Episode operations

Packaged robotics operations that understand trajectory semantics, run over
episode tables.

- [`hand_tracking/`](hand_tracking/) - `track_hands` (MediaPipe CPU / WiLoR
  GPU) on an EgoDex sample, with rendered keypoints and PCK scoring against
  ground truth.
- `motion_trim.py` - `operations.motion_trim` as the "no-noops" audit over
  the committed LIBERO-Spatial demos. Measured verdict: only ~0.2% of steps
  are strict no-ops - this suite barely needed the famous cleaning; point
  the audit at your own teleop data before assuming.
- `pose_query_segments.py` - scenario queries (`daft_physical_ai.pose`) over
  pose-feature tracks: grasping and lifting masks stitched into time
  segments on a public EgoDex sample. The skeleton-gated scenarios (writing
  grip, hammer grip, in-hand, twisting, reaching) additionally need the
  204-D skeleton stream from raw EgoDex HDF5.
