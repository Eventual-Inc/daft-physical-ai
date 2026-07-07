# Resources

- https://docs.daft.ai for the user-facing API docs for Daft
- https://docs.daft.ai/en/stable/extensions/authoring/ for writing Daft extensions
- - https://docs.daft.ai/en/stable/api/udf/ for `@daft.func`, `@daft.cls`, and `@daft.udaf`


# Dev Workflow

1. Set up Python environment, install dependencies, and build dev package: `uv sync`
2. Activate .venv: `source .venv/bin/activate`
3. Run tests: `uv run pytest tests/ -v`

The LeRobot reader (`daft.datasets.lerobot`, [Daft #7090](https://github.com/Eventual-Inc/Daft/pull/7090))
ships in the regular `daft>=0.7.17` dependency - no nightly needed.

# Architecture

The repo is a Daft-native physical-AI starter kit, not a pipeline framework -
Daft already owns reads, transforms, writes, and AI functions. The package owns
physical-AI *semantics* and grows only where examples repeat a helper:

- `episodes/` - the canonical one-row-per-step Arrow/parquet contract shared by
  datasets and eval rollouts. `episode_id` names the evaluation spec
  (`suite/task_id/init_state_id/seed`), not the attempt.
- `ingest/` - adapters for formats Daft has no native reader for (robomimic/
  LIBERO HDF5 today), yielding `Episode` objects into the contract.
- `hands/` - `track_hands` (MediaPipe CPU / WiLoR GPU) behind one output schema.
- `pose/` - model-free pose geometry (48-D hand state + 204-D body skeleton
  conventions), episode-level feature tracks, and scenario queries
  (grasping/lifting from state alone; grips/reaching/in-hand/twisting with
  the skeleton). Dataset adapters produce the arrays; the geometry stays
  NumPy-pure. `pose/temporal.py` is the distributed twin: the same rates as
  in-plan Daft window expressions (`lead(1)` over per-episode windows),
  pinned to the NumPy path by an equivalence test - mirrors the upstream
  daft-examples egodex architecture post-merge.
- `operations/` - deterministic trajectory ops (`motion_trim` / `noop_mask`):
  pure NumPy cores wrapped in Daft groupbys.
- `evals/` - the analysis half of the eval loop: `success_rates`,
  `compare_policies`, `failure_counts` (always grouped by policy - shared specs
  must never chimera), `classify_failure`/`label_failures` (signal-based, the
  production path; `detect_regrasp` needs object poses), and `validate_run`
  against the published LIBERO protocol.
- `curation.py` - the eval->training bridge: `sft_view` (views, not copies),
  `preference_pairs`, `acquisition_map`. Demos join rollouts on
  `(suite, task_name)`, never `task_id`/`episode_id`.

`examples/` is the product surface: numbered stages 01-08 mirroring the
researcher workflow (read, episode data, transforms, episode operations,
inference, writing, training handoff, policy evals). Index in
`examples/README.md`; planned scripts are named in each stage's README. Real
data ships in-repo (500 LIBERO-Spatial demos, 200 benchmark rollouts), so the
loop runs offline up to the training step.

Rollout *generation* (LIBERO sim, OpenVLA/VLA-JEPA stacks, Modal GPU apps)
lives in the VLA-JEPA harness repo and lands here as schema-conforming parquet.
The `TERMINAL_FAILURE_LABELS` tuple is mirrored in the harness's schema - keep
them in sync (the harness still needs `grasp_no_lift` added).

# Roadmap

The loop is operational up to the training step, offline, on committed data:
normalize demos (02) -> audit/trim (04) -> curate views + preference pairs
(06) -> torch handoff (07) -> benchmark analysis, failure labeling, and the
acquisition map (08).

## Now

- [ ] **Close the loop's training step** - fine-tune a small lerobot policy on
  the curated vs naive SFT views and re-roll through the harness. Gated on GPU
  budget authorization (~low hundreds of dollars on Modal); everything up to
  the handoff is already in place.
- [x] **Port `pose/` from daft-examples egodex** - state/skeleton geometry,
  `EpisodeFeatureComputer` + `TemporalFeatureComputer`, scenario predicates,
  calibration, and segment stitching, with examples 03/04 running on the
  public LeRobot v3 EgoDex sample. The EgoDex-specific plumbing (raw-HDF5
  FrameBuilder, SigLIP embeddings, viz, pipeline facade) stays in
  daft-examples; the seam is the state/skeleton arrays.
- [x] **`operations.motion_trim`** - shipped as the no-noops audit
  (`examples/04_episode_operations/motion_trim.py`); measured ~0.2% strict
  no-ops on the LIBERO-Spatial originals.
- [x] **Real LIBERO rollout parquet lands in-repo** (OpenVLA + VLA-JEPA,
  libero_spatial, 100 episodes each, ~2 MB compacted;
  `examples/08_policy_evals/data/`), plus the full 500-demo demonstration
  suite (`examples/02_episode_data/data/`). Follow-up: mirror to HF /
  Multibase once an org namespace is picked, and add the 50-trial canonical
  sweep when the harness runs it.
- [x] **LeRobot examples** (`01_reading_data/lerobot_episode_index.py`,
  `02_episode_data/merge_lerobot_datasets.py`) on the released
  `daft.datasets.lerobot` reader (Daft v0.7.17).

## Later

- [ ] **Publish to PyPI.** Tag a release (`v0.1.0`) to trigger
  `.github/workflows/publish-package.yml` (trusted publishing). This also unblocks
  generated Modal demos, which `pip install daft-physical-ai`.
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
executing step needs the full inference env (mediapipe + scipy + opencv +
matplotlib + nbconvert on top of the regular deps).
`--skip-exec --source <nb>` reuses an already-executed notebook to rebuild
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
