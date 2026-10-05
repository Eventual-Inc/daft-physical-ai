# Resources

- https://docs.daft.ai for the user-facing API docs for Daft
- https://docs.daft.ai/en/stable/extensions/authoring/ for writing Daft extensions
- https://docs.daft.ai/en/stable/api/udf/ for `@daft.func`, `@daft.cls`, and `@daft.udaf`

# Dev Workflow

1. Set up Python environment, install dependencies, and build dev package: `uv sync`
2. Activate .venv: `source .venv/bin/activate`
3. Run tests: `uv run pytest tests/ -v`
4. Lint, format, and type-check: `uv run pre-commit run --all-files` (ruff, ty,
   TOML/YAML formatting, uv-lock). CI runs the same hooks.

# Layout

- `daft_physical_ai/datasets/` - readers for datasets Daft has no native reader
  for (`egodex`, `abc`, `hiw500`, `reassemble`, `omnisharing`). Each starts with a
  lazy `raw()` catalog that opens no files; `_mcap.py` holds the MCAP and Hugging
  Face helpers shared by `abc` and `hiw500`. Prefer Daft's native readers
  (`daft.datasets.lerobot`, `daft.datasets.droid`, `daft.read_mcap`) when they
  cover a dataset. Datasets that aren't common enough for `daft.datasets` belong
  here.
- `daft_physical_ai/datasets/common/ego_centric/` - model-free hand-pose geometry,
  features, and scenario queries.
- `daft_physical_ai/{hands,rewards,proprio,trim}/` - operations: hand tracking,
  reward scoring, motion scoring, and trim windows.
- `daft_physical_ai/cli/` + `_render*.py` + `templates/` - the `daft-physical-ai`
  CLI, which scaffolds a personalized demo per operation. `_render.py`,
  `_render_rewards.py`, and `_render_trim.py` hold each demo's cell list, rendered
  to `.py`, `.ipynb`, and `.md`. `templates/*.tmpl` are the non-Python files the
  CLI writes out: the Modal runtime script for `hands --runtime modal`, and the two
  Robometer server scripts for `rewards`.
- `docs/` - one guide per dataset and operation; `README.md` links them.

# Regenerating the examples demos

`examples/{hands,rewards,trim}/` are **generated** - don't hand-edit them. Each
renders from its `_render*.py` cell list, so the `.py`, `.ipynb`, and `.md`
forms stay in sync. `examples/rewards/{run_robometer_server,modal_eval_server}.py`
are verbatim copies of the templates. To rebuild:

```bash
python scripts/regen_demo.py                                   # hands (default)
ROBOMETER_URL=... python scripts/regen_demo.py --demo rewards  # needs a Robometer server
python scripts/regen_demo.py --demo trim
```

The script renders the notebook, executes it headless (`nbconvert --execute`,
`DAFT_PROGRESS_BAR=0`), then derives the markdown, figure, and script from that one
executed copy. Executing needs each demo's runtime deps (e.g. mediapipe, scipy,
opencv, matplotlib, nbconvert for hands). `--skip-exec --source <nb>` reuses an
already-executed notebook to rebuild just the markdown and image.

# Testing

- The default suite runs on CPU with synthetic fixtures and is what CI runs.
- Tests that read real remote data are marked `integration` and skip without
  credentials. `HF_TOKEN` (needed for gated datasets such as ABC-130k) lives in the
  workspace `.env`, not the shell; REASSEMBLE also needs `REASSEMBLE_ROOT` set to a
  scratch directory. See `TESTING.md` for the pinned episodes and expected counts.
- **WiLoR requires CUDA.** There is no local NVIDIA GPU (dev box is Apple Silicon),
  so test WiLoR on **Modal**. Its tests are Modal-gated, not part of the CPU run.

# PR Conventions

- Titles: Conventional Commits format (`feat(datasets): ...`, `fix: ...`).
- Descriptions: a summary, what changed and why, and validation (commands run and
  their results, including real-data runs where relevant).

# Versioning & Publishing

Versions are derived from git tags via `hatch-vcs`. Tag releases as `v0.1.0`,
`v0.2.0`, etc.

Pushing a `v*` tag triggers `.github/workflows/publish-package.yml`, which
builds a wheel and sdist with `uv build`, uploads both to PyPI via
[trusted publishing](https://docs.pypi.org/trusted-publishers/), and creates a
GitHub release for the tag with auto-generated notes. To cut a release:

```
git tag v0.X.0 <main-sha>
git push origin v0.X.0
```
