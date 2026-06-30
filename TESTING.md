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

## CLI scaffolder (`daft-physical-ai`)

**Unit tests** (`tests/test_cli.py`, in CI): all 6 method x runtime combos render
a `demo.py` that `compile()`s and a `demo.ipynb` whose every code cell `compile()`s;
the Modal script uses `@app.local_entrypoint()` while the Modal notebook uses
`with app.run():` (no entrypoint); config validation (wilor needs `mano_path`; bad
method/runtime/limit); CLI exit codes (0 ok / 1 file exists / 2 bad args) and
`--force` overwrite.

**Runtime policy:** MediaPipe is CPU-only, so it's always `local` - the CLI skips
the runtime question for it and overrides `--runtime modal` to local. `modal` is
only offered when WiLoR (GPU) is involved.

**Interactive prompts** (driven via tmux): default is interactive; flags pre-fill
answers and an explicitly-passed flag is not re-prompted; `--no-input`/non-tty runs
fully non-interactively. Verified: method/runtime/dataset/image prompts, the MANO
prompt appearing only for wilor/both, the runtime prompt skipped for mediapipe,
invalid-choice re-prompt, `--dataset` suppressing its own prompt, and the Modal
login reminder (`modal setup`) printing for the modal runtime.

**Generated demos executed:**

- **Local MediaPipe** - both `python demo.py` and the notebook (headless via
  `nbconvert --execute`, all cells run, the final `.show()` renders Daft's HTML
  table).
- **With `--with-eval`** - the demo's EgoDex scoring ran end to end on CPU and
  produced sensible metrics (e.g. 12 frames: `detect=100% mean_err=0.105
  PCK@.1/.2/.3 = 54/88/97`). The committed `examples/demo.{py,ipynb,md}` are this
  MediaPipe+eval demo; the notebook is executed and the markdown carries its
  outputs (Daft progress bars disabled via `DAFT_PROGRESS_BAR=0` so they stay
  clean).
- **Three mediums stay equivalent** - notebook and markdown render from one shared
  cell list; a test asserts every notebook code cell appears verbatim in the
  markdown.
- **Modal (MediaPipe pipeline)** - a generated Modal script run end to end with
  `modal run`: image builds, dataset downloads, inference runs on Modal, and 6
  annotated frames return (`got 6 frames back from Modal`). This caught a real bug
  - the image was missing `libGLESv2.so.2` (now `libgles2`/`libegl1` are in the
  apt list) - and exercises the same Modal image + pipeline the `both`/WiLoR modal
  demos use.

**Known limitations (not bugs):**

- Generated demos `pip install daft-physical-ai`, so they need the package
  published to PyPI (for the test above, a locally-built wheel was injected
  instead). Local-runtime demos just need it importable locally.
- The Modal **notebook** path (`app.run()`) serializes the notebook-defined
  function, so the kernel's Python must match the image (3.11).
- The generated **WiLoR** demo isn't run end to end here (GPU cost); the WiLoR
  pipeline itself is verified above, and the Modal run exercises the same
  generated-script structure.
