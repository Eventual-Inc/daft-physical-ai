# Getting Started

Build data apps for robot datasets without turning the website into a notebook.

This prototype keeps the public surface simple: Markdown pages, demos, examples,
and focused executable notebooks embedded only where they add value. Daft runs in
the backend Python process, so the same app can run locally during authoring and
on Modal when it needs to be hosted.

## What it demonstrates

- A normal website shell with docs-style navigation and readable pages.
- Markdown as the authoring format for docs, usage patterns, and gallery copy.
- A Marimo notebook embedded as an app island for the live DROID workflow.
- A clear "Use Your Own Data" path to Multibase for proprietary workflows.

## Quick start

Install the runtime dependencies and start the local site:

```bash
uv run \
  --with fastapi \
  --with "uvicorn[standard]" \
  --with markdown \
  --with marimo \
  --with daft \
  uvicorn --app-dir examples/modal_marimo_daft web_app:create_app --factory --reload
```

Open the site, edit Markdown, and refresh. Open a demo page when you need live
execution.

## Hosting workflow

```bash
uvx modal serve examples/modal_marimo_daft/modal_app.py
```

The website remains ordinary FastAPI-rendered HTML. Marimo is mounted under an
internal route and embedded with an iframe.
