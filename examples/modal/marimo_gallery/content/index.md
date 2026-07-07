# Getting Started

A Daft-native starter kit for physical-AI data work: read robot datasets,
normalize episodes, compute pose features, run episode operations, curate
training sets, and analyze policy evals - as plain Daft dataframes, no
pipeline framework.

The loop runs offline on real data committed in the repo: 500 LIBERO-Spatial
demonstrations and 200 OpenVLA / VLA-JEPA benchmark rollouts, with examples
numbered the way a researcher actually works (01 reading data ... 08 policy
evals).

## Quick start

```bash
git clone https://github.com/Eventual-Inc/daft-physical-ai && cd daft-physical-ai
uv sync

uv run python examples/08_policy_evals/success_rates.py     # openvla 84% vs vla_jepa 99%
uv run python examples/08_policy_evals/label_failures.py    # 16/17 failures = re-grasp loops
uv run python examples/04_episode_operations/motion_trim.py # the no-noops audit: ~0.2%
```

Every stage is indexed in `examples/README.md`; each example runs first-try on
a clean environment against public data.

## What the package owns

- `episodes` - the canonical one-row-per-step Arrow/parquet contract shared by
  datasets and eval rollouts.
- `ingest` - adapters for formats Daft has no native reader for (LIBERO HDF5).
- `hands` / `pose` - hand tracking behind one schema; model-free pose geometry,
  feature tracks, and scenario queries.
- `operations` / `evals` / `curation` - motion trim, benchmark comparison,
  failure labeling, and the eval-to-training bridge (SFT views, preference
  pairs, the acquisition map).

Daft owns everything else: readers (`daft.datasets.lerobot`, `daft.datasets.droid`),
expressions, joins, groupbys, media decoding, and the torch handoff.

## This site

The pages here are Markdown; the live DROID notebook is a Marimo app island.
Run it locally while authoring, or host the same app on Modal:

```bash
uvx modal serve examples/modal/marimo_gallery/modal_app.py
```
