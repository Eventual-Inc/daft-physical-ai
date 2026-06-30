# Testing

How each hand-tracking method is verified. Two layers: the committed unit suite
(runs in CI, CPU-only) and out-of-band real-data runs (one for MediaPipe locally,
one for WiLoR on a GPU) whose results are recorded here.

## Unit suite (CI)

```bash
uv sync
uv run pytest tests/ -v
```

- `tests/test_hands_mediapipe.py` - facade errors (unknown method -> `ValueError`)
  and, when the `[mediapipe]` extra is installed, a real MediaPipe run asserting
  the output column dtype equals `HANDS_DTYPE`. The MediaPipe run **skips** when
  the extra is absent (e.g. Python 3.13, which has no MediaPipe wheel yet, or CI
  without the extra).
- `tests/test_hands_wilor.py` - `method="wilor"` without `mano_path` -> `ValueError`;
  the wilor expression's dtype equals `HANDS_DTYPE` (built without executing, since
  `@daft.cls` is lazy - no GPU/torch needed); and `ensure_assets` places a provided
  MANO file where WiLoR expects it (no network/GPU). The schema test
  `importorskip`s torch.

CI runs the suite on Python 3.10 and 3.13; ruff + mypy via pre-commit.

## MediaPipe (CPU, 2D) - local

MediaPipe runs on CPU, so it's verified locally end to end on real frames via the
native `daft.datasets.lerobot` reader (`pepijn223/egodex-test`):

- Result: **10 frames -> 9 hands**; a sample hand was
  `handedness=right confidence=0.54 kp2d=21pts kp3d=None`.
- Equivalence (one-off, not committed): the package output was **byte-identical**
  to the verbatim multibase `MPAnnotator` logic on the same 12 frames -
  `hands_compared=24, max_kp2d_abs_diff=0.0, mismatches=0`. The rewrite only
  reshapes the output schema; it doesn't change any values.

## WiLoR (GPU, 3D) - Modal

WiLoR requires CUDA. There is no local NVIDIA GPU here (dev box is Apple Silicon),
so it's verified on **Modal** (an L4). This is a test harness only - the package
has no Modal dependency and the same code runs on any CUDA GPU (local, Ray, ...).

Setup the env needs (the Modal image, or any GPU box):

- a CUDA `torch` build,
- the `[wilor]` extra **plus** `chumpy` from git
  (`pip install 'chumpy @ git+https://github.com/mattloper/chumpy'`),
- `ensure_assets()` to fetch the WiLoR repo + weights and place `MANO_RIGHT.pkl`
  (research-gated, user-supplied).

Then `track_hands(images, method="wilor", mano_path=..., wilor_root=...)`.

- Result: **12 frames -> 24 hands**, each with real 3D `kp3d` (21 points), at the
  default `gpus=0.25, max_concurrency=4`.
- Equivalence (one-off, not committed): the package output was **byte-identical**
  to the multibase WiLoR output on the same 12 frames -
  `hands_compared=24, max_kp2d_diff=0.0, max_kp3d_diff=0.0, mismatches=0`.

### Note on torch import order

WiLoR segfaulted (`SIGSEGV`) at model load whenever `torch`/CUDA was first imported
*inside* the `@daft.cls` worker. `track_hands(method="wilor")` therefore imports
`torch` in the caller's process first, so Daft's worker import is a safe no-op.
This is a Daft+torch interaction, not Modal- or concurrency-specific.
