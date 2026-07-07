# 03 - Transforms

Deterministic feature computation as Daft expressions and NumPy episode
passes - no models, no services.

- `pose_features_numpy.py` - per-episode pose-feature tracks
  (`daft_physical_ai.pose`) from the 48-D hand state of a public EgoDex
  sample in LeRobot v3 format: curl, pinch, palm orientation, and
  forward-difference rates, one vectorized pass per episode. In-memory
  arrays - easy to test and inspect.
- `pose_rates_in_dag.py` - the distributed twin: the same geometry as a
  `@daft.func` over the state column and every rate as a window expression
  (`lead(1).over(partition_by(episode))`), with scenario thresholds as plain
  column predicates - one lazy plan from the reader to a single collect.
  Both paths are pinned to each other by an equivalence test.

Planned: `quality_checks.py` (jitter, gaps, action norms),
`frame_embeddings.py` (sampled frames to image embeddings).
