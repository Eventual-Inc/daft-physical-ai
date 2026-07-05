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

# Architecture

The repo is a Daft-native physical-AI starter kit, not a pipeline framework -
Daft already owns reads, transforms, writes, and AI functions. The package owns
physical-AI *semantics* and grows only where examples repeat a helper:

- `episodes/` - the canonical one-row-per-step Arrow/parquet contract shared by
  datasets and eval rollouts. `episode_id` names the evaluation spec
  (`suite/task_id/init_state_id/seed`), not the attempt.
- `hands/` - `track_hands` (MediaPipe CPU / WiLoR GPU) behind one output schema.
- `evals/` - the analysis half of the eval loop: `success_rates`,
  `compare_policies`, `failure_counts` (always grouped by policy - shared specs
  must never chimera), `detect_regrasp`, and `validate_run` against the
  published LIBERO protocol.

`examples/` is the product surface: numbered stages 01-08 mirroring the
researcher workflow (read, episode data, transforms, episode operations,
inference, writing, training handoff, policy evals). Index in
`examples/README.md`; planned scripts are named in each stage's README.

Rollout *generation* (LIBERO sim, OpenVLA/VLA-JEPA stacks, Modal GPU apps)
lives in the VLA-JEPA harness repo and lands here as schema-conforming parquet.

# Roadmap

## Now

- [ ] **Port `pose/` from daft-examples egodex** (pure-NumPy hand/skeleton
  geometry, feature UDF wrappers, scenario queries) into
  `daft_physical_ai/pose/` + `examples/03_transforms/`.
- [ ] **`operations.motion_trim`** - first new deterministic episode op
  (`examples/04_episode_operations/motion_trim.py`).
- [x] **Real LIBERO rollout parquet lands in-repo** (OpenVLA + VLA-JEPA,
  libero_spatial, 100 episodes each, ~2 MB compacted;
  `examples/08_policy_evals/data/`) with `success_rates.py` /
  `compare_policies.py` / `validate_protocol.py` running against it offline.
  Follow-up: mirror to HF / Multibase once an org namespace is picked, and add
  the 50-trial canonical sweep when the harness runs it.
- [ ] **LeRobot examples** (`01_reading_data/lerobot_episode_index.py`,
  `02_episode_data/merge_lerobot_datasets.py`) once the reader ships in a
  released Daft.

## Later

- [ ] **Publish to PyPI.** Tag a release (`v0.1.0`) to trigger
  `.github/workflows/publish-package.yml` (trusted publishing). This also unblocks
  generated Modal demos, which `pip install daft-physical-ai`.
- [ ] Switch off the Daft nightly once `daft.datasets.lerobot` ships in a release
  (> v0.7.16): bump the `daft` floor in `pyproject.toml`, drop the nightly
  install step above, and re-run `uv lock`.
- [ ] **Promote the LIBERO rollout runner + `Policy` seam** from the VLA-JEPA
  harness into a `daft-physical-ai[libero]` extra. Feasible - the `hf-libero`
  wheel co-resolves with modern policy stacks in one Python >=3.12 process -
  but gated on a lerobot release carrying the `vla_jepa` policy port (a
  git-SHA pin cannot ship in PyPI metadata). Until then the harness repo owns
  generation; the schema is the contract.

# Regenerating the hand-tracking demo

`examples/04_episode_operations/hand_tracking/{demo.py,demo.ipynb,demo.md,demo_keypoints.png}`
are **generated** - don't hand-edit them. They all render from one shared cell
list in `daft_physical_ai/_render.py`, so editing the source keeps the three
formats in sync. To rebuild them:

```bash
python scripts/regen_demo.py          # render -> execute the notebook -> derive md + image
```

The script renders the notebook (empty), executes it headless
(`nbconvert --execute`, `DAFT_PROGRESS_BAR=0`), then derives everything else from
that one executed copy: the figure is written to `demo_keypoints.png`, printed
output is fenced, and the `.show()` HTML table becomes a markdown table. The
executing step needs the full inference env (a Daft with the LeRobot reader +
mediapipe + scipy + opencv + matplotlib + nbconvert; see the nightly-Daft note
above). `--skip-exec --source <nb>` reuses an already-executed notebook to rebuild
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
