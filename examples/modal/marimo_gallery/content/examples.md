# Examples

Each card is a concrete physical-AI workflow: the input data, the operation,
and the result - with the numbers the run actually produced. The full
collection is numbered 01-08 in `examples/README.md`, mirroring the researcher
workflow: reading data, episode data, transforms, episode operations,
inference, writing data, training handoff, policy evals.

## What runs today, on data in the repo

- **Reading data** - DROID episode index (`daft.datasets.droid`), LeRobot v3
  episode/task/frame views (`daft.datasets.lerobot`, Daft >= 0.7.17).
- **Episode data** - the full LIBERO-Spatial demonstration suite normalized
  into canonical step rows (500 demos, 62,250 rows, ~4.7 MB in-repo) and
  LeRobot session merging with re-indexed episodes.
- **Transforms** - pure-NumPy pose-feature tracks per episode (curl, pinch,
  palm orientation, rates) on a public EgoDex sample.
- **Episode operations** - MediaPipe hand tracking scored against EgoDex
  ground truth; the motion-trim/no-noops audit (measured: ~0.2% strict no-ops
  on LIBERO-Spatial); pose scenario queries stitched into time segments.
- **Writing + handoff** - curated SFT views (motion-trimmed successes),
  preference pairs from the policy comparison, and `to_torch_dataloader`
  batches: `(64, 7)` actions, `(64, 8)` states.
- **Policy evals** - OpenVLA 84% vs VLA-JEPA 99% on the same benchmark specs;
  every failure labeled from per-step signals (16 of 17 are slip-then-regrasp
  loops); the acquisition map ranking what to collect next.

Media-heavy artifacts (video clips, rendered keypoints) stay in the repo;
proprietary datasets route to Multibase.
