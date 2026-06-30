# Testing

Two layers: the committed unit suite (CI, CPU-only) and out-of-band real-data
runs whose results are recorded here.

## Unit suite (CI)

```bash
uv sync && uv run pytest tests/ -v
```

- MediaPipe facade + a real run asserting the output dtype equals `HANDS_DTYPE`.
  The real run skips without the `[mediapipe]` extra (e.g. Python 3.13).
- WiLoR facade: `mano_path` required; the expression's dtype equals `HANDS_DTYPE`
  (built without executing - `@daft.cls` is lazy, no GPU needed); `ensure_assets`
  places a provided MANO file. CLI: all method x runtime combos render to valid
  Python + `.ipynb`.

CI runs on Python 3.10 + 3.13; ruff + mypy via pre-commit.

## Real-data runs

**MediaPipe (CPU, 2D)** - verified locally on `pepijn223/egodex-test` via
`daft.datasets.lerobot`: 10 frames -> 9 hands. Byte-identical to the original
reference implementation (`max_kp2d_abs_diff=0.0` over 24 hands).

**WiLoR (GPU, 3D)** - needs CUDA, so verified on Modal (an L4); the package has no
Modal dependency and runs on any CUDA GPU. The env needs a CUDA `torch` build, the
`[wilor]` extra plus `chumpy` from git
(`pip install 'chumpy @ git+https://github.com/mattloper/chumpy'`), and
`ensure_assets()` to fetch the repo + weights and place `MANO_RIGHT.pkl`. Then
`track_hands(images, method="wilor", mano_path=..., wilor_root=...)`: 12 frames ->
24 hands with real 3D `kp3d`, byte-identical to the reference
(`max_kp2d_diff=max_kp3d_diff=0.0`).

> **torch import order:** WiLoR segfaults if `torch`/CUDA is first imported inside
> the `@daft.cls` worker, so `track_hands(method="wilor")` imports it in the
> caller's process first. A Daft+torch interaction, not Modal- or
> concurrency-specific.
