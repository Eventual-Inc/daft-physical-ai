# The `daft-physical-ai` CLI

Each capability is its own subcommand - `daft-physical-ai hands`,
`daft-physical-ai rewards`, and `daft-physical-ai trim` (`daft-physical-ai`
with no arguments lists what's available). Each scaffolds a personalized,
runnable demo; the `rewards` scaffold also writes the Robometer server
scripts next to the demo, so one directory holds everything: score the
episodes, and serve the model locally or on Modal.

`uvx daft-physical-ai <subcommand>` runs the CLI without installing anything
(scaffolding needs no inference deps). If the
[PyPI package](https://pypi.org/project/daft-physical-ai/) is already installed
(`pip install daft-physical-ai`), the plain command works too; from a clone of
this repo, `uv sync` installs it (`uv run daft-physical-ai`).

To *run* a generated demo you also need its runtime deps (inference libraries,
plotting). `uvx` covers that too - one line, nothing installed:

```bash
uvx --from jupyterlab --with "daft-physical-ai[mediapipe]" --with matplotlib --with scipy \
  jupyter-lab hand-tracking-demo/demo.ipynb
```

(`scipy` is only needed if the demo includes the ground-truth eval.)

In a clone, `uv sync` already brings a Daft with the LeRobot reader; install the
extras into the venv, then run from the activated venv - not `uv run`, which
re-syncs the env and would drop them:

```bash
source .venv/bin/activate
uv pip install -U av mediapipe scipy opencv-python matplotlib jupyterlab
jupyter lab hand-tracking-demo/demo.ipynb
```
