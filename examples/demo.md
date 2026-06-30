# Hand tracking demo - MediaPipe (local runtime)

This demo reads a [LeRobot](https://docs.daft.ai/en/stable/datasets/lerobot/) dataset, runs hand tracking (MediaPipe) as a Daft UDF with `track_hands`, and shows the keypoints. Every method returns the same schema: a list of `{handedness, confidence, kp2d, kp3d?}` per frame (`kp3d` is null for MediaPipe).

## Setup

Install with `pip install daft-physical-ai[mediapipe]`, then import.

```python
from daft.datasets import lerobot

from daft_physical_ai import track_hands
```

## Configure

The dataset, the camera column to decode, and how many frames to run.

```python
DATASET = "pepijn223/egodex-test"
IMAGE_COLUMN = "observation.image"
LIMIT = 12
```

## Read the dataset

The LeRobot reader gives one row per frame, decoding the camera into an image column.

```python
df = lerobot.read(DATASET, load_video_frames=IMAGE_COLUMN).limit(LIMIT)
```

## Track hands

`track_hands` returns a hand-pose column. It's a lazy, batched Daft UDF, so nothing runs until we materialize below.

```python
df = df.with_column("hands", track_hands(df[IMAGE_COLUMN], method="mediapipe"))
```

## Inspect the results

`.show()` triggers execution and renders the keypoints per frame.

```python
df.select("episode_index", "frame_index", "hands").show()
```

## Evaluate against ground truth

EgoDex ships per-frame GT hand poses, so we can score the predictions: project both GT hands, match the predicted hands to them, and report detect% + PCK. The matching runs as a Daft UDF (`score`); the summary is computed from the collected results.

> EgoDex-specific (GT layout + camera intrinsics). Needs `pip install scipy`.

```python
# --- Evaluation against EgoDex ground truth (2D, wrist + 5 fingertips) ---
# EgoDex-specific: GT hand poses live in observation.state (left = dims 0-23,
# right = 24-47); the camera is observation.extrinsics. Needs scipy + numpy.
import numpy as np
from scipy.optimize import linear_sum_assignment

import daft
from daft import DataType, col

FX = FY = 736.6339          # EgoDex camera intrinsics
CX, CY = 960.0, 540.0
SIX = [0, 4, 8, 12, 16, 20]  # wrist + 5 fingertip keypoints
THRESH = [0.1, 0.2, 0.3]     # PCK thresholds (normalized)


def _hand_pts(state, side):
    b = side * 24            # 24 dims per hand: wrist(3) + joints; we take wrist + 5 tips
    return np.vstack([state[b : b + 3], state[b + 9 : b + 24].reshape(5, 3)])


def _project(points_world, extr):
    cam_from_world = np.linalg.inv(np.asarray(extr, float).reshape(4, 4))
    cam = (cam_from_world @ np.hstack([points_world, np.ones((len(points_world), 1))]).T).T[:, :3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([FX * cam[:, 0] / z + CX, FY * cam[:, 1] / z + CY], axis=1)
    uv[z <= 0] = np.nan
    return uv


def _norm(p):               # translation + scale invariant (hand size)
    p = p - p[0]
    return p / (np.linalg.norm(p[1:], axis=1).mean() + 1e-9)


def _pair_err(gt6, pred6):  # per-keypoint error, fingertips matched by assignment
    g, m = _norm(gt6), _norm(pred6)
    d = np.linalg.norm(g[1:, None] - m[None, 1:], axis=2)
    r, c = linear_sum_assignment(d)
    return np.concatenate([[0.0], d[r, c]])


_ERR = DataType.struct({
    "n_gt": DataType.int64(),
    "n_matched": DataType.int64(),
    "errs": DataType.list(DataType.list(DataType.float64())),
})


@daft.func(return_dtype=_ERR)
def score(hands, state, extr):
    """Match predicted hands to the 2 GT hands (Hungarian on normalized error)."""
    gts = [uv for uv in (_project(_hand_pts(np.asarray(state, float), s), extr) for s in (0, 1)) if np.isfinite(uv).all()]
    preds = [np.asarray(h["kp2d"], float)[SIX] for h in (hands or [])]
    if not gts or not preds:
        return {"n_gt": len(gts), "n_matched": 0, "errs": []}
    pair = [[_pair_err(g, p) for p in preds] for g in gts]
    cost = np.array([[e.mean() for e in row] for row in pair])
    r, c = linear_sum_assignment(cost)   # match predicted hands to GT hands
    return {"n_gt": len(gts), "n_matched": len(r), "errs": [[float(x) for x in pair[i][j]] for i, j in zip(r, c)]}


def report(label, scores):
    n_gt = sum(s["n_gt"] for s in scores)
    matched = sum(s["n_matched"] for s in scores)
    errs = [e for s in scores for hand in s["errs"] for e in hand]
    mean_errs = [float(np.mean(hand)) for s in scores for hand in s["errs"]]
    pck = [100 * np.mean([e < t for e in errs]) if errs else 0.0 for t in THRESH]
    detect = 100 * matched / n_gt if n_gt else 0.0
    mean = float(np.mean(mean_errs)) if mean_errs else float("nan")
    print(f"{label:12} detect={detect:3.0f}%  mean_err={mean:.3f}  "
          f"PCK@.1/.2/.3 = {pck[0]:.0f}/{pck[1]:.0f}/{pck[2]:.0f}")
```

```python
df = df.with_column("score_hands", score(col("hands"), col("observation.state"), col("observation.extrinsics")))
scored = df.select("score_hands").to_pydict()
```

```python
print("EgoDex 2D accuracy:")
report("MediaPipe", scored["score_hands"])
```

```
EgoDex 2D accuracy:
MediaPipe    detect=100%  mean_err=0.105  PCK@.1/.2/.3 = 54/88/97
```
