# Resources

- https://docs.daft.ai for the user-facing API docs for Daft
- https://docs.daft.ai/en/stable/extensions/authoring/ for writing Daft extensions
- - https://docs.daft.ai/en/stable/api/udf/ for `@daft.func`, `@daft.cls`, and `@daft.udaf`


# Dev Workflow

1. Set up Python environment, install dependencies, and build dev package: `uv sync`
2. Activate .venv: `source .venv/bin/activate`
3. Run tests: `uv run pytest tests/ -v`
4. The LeRobot reader (`daft.datasets.lerobot`) ships in stable Daft (>= v0.7.17),
   so `uv sync` is all you need.

# Roadmap

Working implementations to port from: multibase `src/post7_hand_tracking/egodex_daft/`
(`mediapipe_egodex_daft.py`, `wilor_egodex_daft.py`), already on the native
`daft.datasets.lerobot` reader.

## Now

- [ ] **Implement `hands/` - MediaPipe first** (easiest: CPU, permissive license,
  no weights to supply). `_mediapipe.py` `@daft.cls` + the `track_hands(method="mediapipe")`
  facade, returning the shared output schema.
- [ ] **Implement WiLoR** (`method="wilor"`): GPU, 3D MANO keypoints, user-supplied
  `mano_path`. Port the `@daft.cls` from multibase.
- [ ] **Test both thoroughly and document how** (a `TESTING.md`): capture the
  commands + observed hand counts. MediaPipe locally on CPU; WiLoR on Modal (see
  Testing & GPU below).
- [ ] **Add tests + align with the template**: real `tests/` (replace the
  placeholder `greet`), confirm `track_hands` returns a Daft expression, lock the
  output schema.
- [ ] **CLI + demo**: the `daft-physical-ai` console script + demo notebook/script.

## Later

- [x] **Publish to PyPI.** Done: [v0.1.0 on PyPI](https://pypi.org/project/daft-physical-ai/)
  (tag push triggers `.github/workflows/publish-package.yml`, trusted publishing).
  This also unblocks generated Modal demos, which `pip install daft-physical-ai`.
- [ ] Bump the `daft` floor to `>=0.7.18` once it ships stable (brings the
  batched video decode) - tracked with full steps in
  [#17](https://github.com/Eventual-Inc/daft-physical-ai/issues/17).

# Regenerating the hand-tracking demo

`examples/04_episode_operations/hand_tracking/{demo.py,demo.ipynb,demo.md,demo_keypoints.png}`
are **generated** -
don't hand-edit them. They all render from one shared cell list in
`daft_physical_ai/_render.py`, so editing the source keeps the three formats in
sync. To rebuild them:

```bash
python scripts/regen_demo.py          # render -> execute the notebook -> derive md + image
```

The script renders the notebook (empty), executes it headless
(`nbconvert --execute`, `DAFT_PROGRESS_BAR=0`), then derives everything else from
that one executed copy: the figure is written to `demo_keypoints.png`, printed
output is fenced, and the `.show()` HTML table becomes a markdown table. The
executing step needs the full inference env (a Daft with the LeRobot reader +
mediapipe + scipy + opencv + matplotlib + nbconvert; `uv sync` plus the demo
extras covers it). `--skip-exec --source <nb>` reuses an already-executed notebook to rebuild
just the markdown/image (no inference env needed) - handy for tweaking the
conversion.

# Testing & GPU

- **MediaPipe runs on CPU** - testable locally and in CI.
- **WiLoR requires CUDA.** There is no local NVIDIA GPU (dev box is Apple Silicon /
  Metal only), so test WiLoR on **Modal**. CI can't run it - mark WiLoR tests as
  integration / Modal-gated, not part of the default CPU test run.

# PR Conventions

- Titles: Conventional Commits format; enforced by `.github/workflows/pr-labeller.yml`.
- Descriptions: follow `.github/pull_request_template.md`.
