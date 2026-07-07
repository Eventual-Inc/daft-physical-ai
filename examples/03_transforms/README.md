# 03 - Transforms

Deterministic feature computation as Daft expressions and NumPy episode
passes - no models, no services.

- `pose_features_numpy.py` - per-episode pose-feature tracks
  (`daft_physical_ai.pose`) from the 48-D hand state of a public EgoDex
  sample in LeRobot v3 format: curl, pinch, palm orientation, and
  forward-difference rates, one vectorized pass per episode. No frame
  explode, no window functions.

Planned: `quality_checks.py` (jitter, gaps, action norms),
`frame_embeddings.py` (sampled frames to image embeddings).
