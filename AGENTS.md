# Resources

- https://docs.daft.ai for the user-facing API docs for Daft
- https://docs.daft.ai/en/stable/extensions/authoring/ for writing Daft extensions
- - https://docs.daft.ai/en/stable/api/udf/ for `@daft.func`, `@daft.cls`, and `@daft.udaf`


# Dev Workflow

1. Set up Python environment, install dependencies, and build dev package: `uv sync`
2. Activate .venv: `source .venv/bin/activate`
3. Run tests: `uv run pytest tests/ -v`
4. To use the LeRobot reader (`daft.datasets.lerobot`), install a nightly Daft -
   it is merged ([Daft #7090](https://github.com/Eventual-Inc/Daft/pull/7090))
   but not yet in a released version (latest is v0.7.16):
   `uv pip install --prerelease=allow --extra-index-url https://nightly.daft.ai -U daft`

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

- [ ] **Publish to PyPI.** Tag a release (`v0.1.0`) to trigger
  `.github/workflows/publish-package.yml` (trusted publishing). This also unblocks
  generated Modal demos, which `pip install daft-physical-ai`.
- [ ] Switch off the Daft nightly once `daft.datasets.lerobot` ships in a release
  (> v0.7.16): bump the `daft` floor in `pyproject.toml`, drop the nightly
  install step above, and re-run `uv lock`.

# Testing & GPU

- **MediaPipe runs on CPU** - testable locally and in CI.
- **WiLoR requires CUDA.** There is no local NVIDIA GPU (dev box is Apple Silicon /
  Metal only), so test WiLoR on **Modal**. CI can't run it - mark WiLoR tests as
  integration / Modal-gated, not part of the default CPU test run.

# PR Conventions

- Titles: Conventional Commits format; enforced by `.github/workflows/pr-labeller.yml`.
- Descriptions: follow `.github/pull_request_template.md`.
