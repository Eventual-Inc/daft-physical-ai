# Modal + FastAPI + Marimo + Daft

Minimal proof that a normal website can be authored from Markdown, embed a
Marimo notebook as an app island, and run Daft in the backend.

The FastAPI app serves:

- `/` - Getting Started / quick start
- `/demos` - workflow-topic demos
- `/examples` - concrete example gallery
- `/demos/droid-kitchen` - a normal docs page with an embedded notebook iframe
- `/demos/egodex-hands` - a local-first EgoDex hand-tracking demo page
- `/demos/policy-evals` - a CPU-only policy-eval / failure-mining demo page
- `/_marimo/droid-kitchen` - the Marimo notebook backed by Daft

## Local iteration

Write broad docs and gallery copy in Markdown:

```text
examples/modal/marimo_gallery/content/
```

Run the site shell locally:

```bash
uv run \
  --with fastapi \
  --with "uvicorn[standard]" \
  --with markdown \
  --with marimo \
  --with daft \
  uvicorn --app-dir examples/modal/marimo_gallery web_app:create_app --factory --reload
```

Edit the Marimo notebook directly when you want notebook authoring mode:

```bash
uv run --with marimo --with daft marimo edit examples/modal/marimo_gallery/notebooks/droid_kitchen.py --no-token
```

Run the EgoDex hand-tracking demo locally without Modal:

```bash
source .venv/bin/activate
uv pip install --prerelease=allow --extra-index-url https://nightly.daft.ai \
  -U daft av mediapipe scipy opencv-python matplotlib
python examples/04_episode_operations/hand_tracking/demo.py
```

Run the policy-eval failure-mining demo locally without Modal:

```bash
uv run python examples/08_policy_evals/mine_failures.py --no-plot
uv run --with matplotlib python examples/08_policy_evals/mine_failures.py
```

## Pair with Claude Code

This repo has the `marimo-pair` and `retro-marimo-pair` skills installed under
`.claude/skills/`, with project permissions in `.claude/settings.json`.

Start a Marimo notebook with `--no-token`, then invoke the skill from Claude
Code:

```text
/marimo-pair pair with me on examples/modal/marimo_gallery/notebooks/droid_kitchen.py
```

The top-right "Use Your Own Data" CTA links directly to
`https://eventual.ai/multibase`.

## Run on your Modal account

```bash
uvx modal setup
uvx modal serve examples/modal/marimo_gallery/modal_app.py
```

Deploy it as a persistent Modal web app:

```bash
uvx modal deploy examples/modal/marimo_gallery/modal_app.py
```

The browser renders Marimo. Daft runs in the Modal container.
