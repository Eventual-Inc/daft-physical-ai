"""Render a personalized reward-scoring demo (script + notebook + markdown) from a config.

Pure string/JSON building - no I/O, no prompts - so it's easy to unit-test, same
shape as :mod:`daft_physical_ai._render` (hands). The CLI
(:mod:`daft_physical_ai.cli.rewards`) collects a :class:`RewardsDemoConfig` and
calls the render functions, and also copies the two server scripts
(``run_robometer_server.py``, ``modal_eval_server.py``) next to the demo.

The generated demo imports :func:`daft_physical_ai.rewards.score_rewards` and
talks to a Robometer eval server behind a URL. The package never serves the
model; serving lives in the copied server scripts, which are user code.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._render import _build_ipynb, _cells_to_markdown, _cells_to_script

# Server scripts copied verbatim next to the demo (no substitution).
SERVER_TEMPLATES = ("run_robometer_server.py", "modal_eval_server.py")


@dataclass
class RewardsDemoConfig:
    """Everything needed to render a rewards demo."""

    dataset: str = "nvidia/LIBERO_LeRobot_v3"
    split: str = "libero_90"
    video_key: str = "observation.images.image"
    episodes: int = 5
    max_frames: int = 8

    def validate(self) -> None:
        if self.episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {self.episodes}")
        if self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")


def _title(config: RewardsDemoConfig) -> str:
    return f"Reward scoring demo - Robometer on {config.dataset}"


def _config_block(config: RewardsDemoConfig) -> str:
    return "\n".join(
        [
            f'DATASET = "{config.dataset}"',
            f'SPLIT = "{config.split}"',
            f'VIDEO_KEY = "{config.video_key}"  # camera whose video the episodes index into',
            f"EPISODES = {config.episodes}",
            f"MAX_FRAMES = {config.max_frames}  # frames sampled per episode (first + last always included)",
        ]
    )


_SERVER_CELL = """import os

# Any running Robometer eval server works here - the pipeline only sees a URL.
#   local GPU:  python run_robometer_server.py         (then http://localhost:8001)
#   Modal:      modal deploy modal_eval_server.py      (prints the https URL)
ROBOMETER_URL = os.environ["ROBOMETER_URL"]
# Modal proxy-auth deployments need these two headers; a local server needs none.
HEADERS = (
    {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]}
    if os.environ.get("MODAL_KEY")
    else None
)"""

_COLLECT_CELL = """episodes = df.to_pylist()
for e in episodes:
    r = e["rewards"]
    print(f"ep{e['episode_index']} ({e['task']}):")
    print(f"  progress = {[round(p, 2) for p in r['reward_score']]}")
    print(f"  success  = {r['robometer_success'][-1]:.2f} (final frame)")"""

_VIZ_CELL = """import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))
for e in episodes:
    r = e["rewards"]
    xs = [f["index"] for f in r["reward_frames"]]
    ax.plot(xs, r["reward_score"], marker="o", label=f"ep{e['episode_index']}: {e['task'][:45]}")
ax.set_xlabel("frame index (within episode)")
ax.set_ylabel("task progress")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Robometer per-frame task progress")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()"""

_FILTER_CELL = """from daft import col

# episodes the model doubts succeeded - review these before training on them
flagged = df.where(col("rewards")["robometer_success"][-1] < 0.5)
flagged.select("episode_index", "task").show()"""


def _demo_cells(config: RewardsDemoConfig) -> list[tuple[str, str]]:
    """Ordered (markdown|code) cells - the shared source for script, notebook, and markdown."""
    intro = (
        f"# {_title(config)}\n\n"
        "This demo scores robot episodes with a reward model "
        "([Robometer-4B](https://huggingface.co/robometer/Robometer-4B)) as a Daft pipeline: "
        "per-frame task progress (0-1) plus success probability, written back as a dataset "
        "column with `score_rewards`. Downstream uses: filter failed or stalled episodes "
        "before BC training, dense reward for RL post-training, and catching mislabeled tasks "
        "(all-zero progress usually means the task text is wrong).\n\n"
        "Scoring is a pure HTTP call - you bring a running Robometer eval server "
        "(`run_robometer_server.py` on any NVIDIA GPU, or `modal deploy modal_eval_server.py`; "
        "both can be found next to this demo) and point `ROBOMETER_URL` at it."
    )
    cells: list[tuple[str, str]] = [
        ("markdown", intro),
        (
            "markdown",
            "## Setup\n\nInstall with `pip install daft-physical-ai huggingface_hub matplotlib`, then import.",
        ),
        (
            "code",
            "import daft\nfrom daft import col, lit\n\nfrom daft_physical_ai.rewards import score_rewards",
        ),
        (
            "markdown",
            "## Configure\n\nThe dataset, which camera's video to decode, how many episodes to "
            "score, and how many frames to sample per episode.",
        ),
        ("code", _config_block(config)),
        (
            "markdown",
            "## Point at your Robometer server\n\nThe pipeline takes a URL and doesn't care "
            "what's behind it - a local GPU ([`run_robometer_server.py`](run_robometer_server.py)), "
            "Modal ([`modal_eval_server.py`](modal_eval_server.py)), or anything else that "
            "serves the eval server's `/evaluate_batch_npy`.",
        ),
        ("code", _SERVER_CELL),
        (
            "markdown",
            "## Fetch the episode metadata and video\n\nLeRobot v3 stores episode metadata as "
            "parquet and concatenates episodes into shared mp4 files. The first metadata and "
            "video files cover the first episodes, which is all this demo scores.",
        ),
        (
            "code",
            "from huggingface_hub import hf_hub_download\n"
            "\n"
            'meta_path = hf_hub_download(DATASET, f"{SPLIT}/meta/episodes/chunk-000/file-000.parquet", '
            'repo_type="dataset")\n'
            'video_path = hf_hub_download(DATASET, f"{SPLIT}/videos/{VIDEO_KEY}/chunk-000/file-000.mp4", '
            'repo_type="dataset")',
        ),
        (
            "markdown",
            "## Build the episode DataFrame\n\nOne row per episode: the task text (from the "
            "episode's own LeRobot metadata), its length, and where its "
            "frames live in the video.",
        ),
        (
            "code",
            "df = (\n"
            "    daft.read_parquet(meta_path)\n"
            '    .sort("episode_index")\n'
            "    .limit(EPISODES)\n"
            "    .select(\n"
            '        "episode_index",\n'
            '        col("tasks").list_join("; ").alias("task"),\n'
            '        "length",\n'
            '        col(f"videos/{VIDEO_KEY}/from_timestamp").alias("from_ts"),\n'
            '        col(f"videos/{VIDEO_KEY}/to_timestamp").alias("to_ts"),\n'
            '        lit(video_path).alias("video_path"),\n'
            "    )\n"
            ")",
        ),
        (
            "markdown",
            "## Score the episodes\n\n`score_rewards` returns a reward column: it samples "
            "`MAX_FRAMES` frames per episode, decodes them from the episode's segment of the "
            "video, and asks the server for per-frame progress + success. It's a lazy async "
            "Daft UDF, so nothing runs until we materialize below - and episodes score "
            "concurrently when they do.",
        ),
        (
            "code",
            "df = df.with_column(\n"
            '    "rewards",\n'
            "    score_rewards(\n"
            '        df["task"], df["length"], df["from_ts"], df["to_ts"], df["video_path"],\n'
            "        url=ROBOMETER_URL, max_frames=MAX_FRAMES, headers=HEADERS,\n"
            "    ),\n"
            ")",
        ),
        (
            "markdown",
            "## Read the curves\n\nA healthy episode climbs toward 1.0. A curve that flatlines "
            "near 0 is a failed or stalled episode - or a mislabeled task - that you almost "
            "trained on.",
        ),
        ("code", _COLLECT_CELL),
        ("markdown", "## Plot the progress curves\n\nNeeds `matplotlib`."),
        ("code", _VIZ_CELL),
        (
            "markdown",
            "## Filter with a Daft query\n\nThe scores are ordinary columns, so quality gates "
            "are one-liners - here, episodes whose final-frame success probability is below 0.5.",
        ),
        ("code", _FILTER_CELL),
    ]
    return cells


def render_script(config: RewardsDemoConfig) -> str:
    """Render the standalone .py demo for this config."""
    config.validate()
    return _cells_to_script(_demo_cells(config))


def render_notebook(config: RewardsDemoConfig) -> str:
    """Render the demo as a Jupyter notebook (.ipynb)."""
    config.validate()
    return _build_ipynb(_demo_cells(config))


def render_markdown(config: RewardsDemoConfig, outputs: list[str] | None = None) -> str:
    """Render the demo as a Markdown tutorial (prose + fenced code), for reading."""
    config.validate()
    return _cells_to_markdown(_demo_cells(config), outputs)


def load_server_script(name: str) -> str:
    """Return the contents of a server script shipped as a package template."""
    from importlib.resources import files

    if name not in SERVER_TEMPLATES:
        raise ValueError(f"unknown server script {name!r}; expected one of {SERVER_TEMPLATES}")
    return (files("daft_physical_ai.templates") / f"{name}.tmpl").read_text(encoding="utf-8")
