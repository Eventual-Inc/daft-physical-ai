"""`daft-physical-ai rewards`: scaffold a personalized reward-scoring demo.

Same shape as `hands`: ask a few questions (dataset, episodes, ...), then
generate a runnable script/notebook/markdown demo plus the two Robometer
server scripts (`run_robometer_server.py` for any NVIDIA GPU,
`modal_eval_server.py` for Modal), and print the commands to serve and run.
Flags pre-fill answers; with `--no-input` it runs non-interactively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .._render_rewards import (
    SERVER_TEMPLATES,
    RewardsDemoConfig,
    load_server_script,
    render_markdown,
    render_notebook,
    render_script,
)

_DEFAULT_OUTPUT_DIR = "reward-scoring-demo"

_INTRO = (
    "daft-physical-ai rewards - scaffold a reward-scoring demo.\n"
    "Answer a few questions to generate a runnable script + notebook (scoring a\n"
    "LeRobot dataset with score_rewards against a Robometer eval server) plus the\n"
    "server scripts to run that server yourself. Press Enter to accept the\n"
    "[default] shown for any question.\n"
)

_DESCRIPTION = "Scaffold a personalized reward-scoring demo (script + notebook + server scripts)."


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach the `rewards` subcommand to the top-level parser."""
    p = subparsers.add_parser("rewards", help=_DESCRIPTION.rstrip(".").lower(), description=_DESCRIPTION)
    p.set_defaults(func=run)
    # Defaults are None so we can tell "user passed this" from "fall back to the default";
    # an explicitly-passed flag is never re-prompted. The real defaults live in RewardsDemoConfig.
    p.add_argument("--dataset", help="LeRobot v3 dataset id (default: nvidia/LIBERO_LeRobot_v3)")
    p.add_argument("--split", help="dataset subdirectory holding meta/ and videos/ (default: libero_90)")
    p.add_argument("--video-key", help="camera video key to decode (default: observation.images.image)")
    p.add_argument("--episodes", type=int, help="number of episodes to score (default: 5)")
    p.add_argument("--max-frames", type=int, help="frames sampled per episode (default: 8)")
    p.add_argument(
        "--format",
        choices=("script", "notebook", "markdown", "all"),
        default="all",
        help="what to generate: script (.py), notebook (.ipynb), markdown (.md), or all (default)",
    )
    p.add_argument(
        "--server-scripts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write run_robometer_server.py + modal_eval_server.py (default on)",
    )
    p.add_argument("--output-dir", help=f"directory to write the demo into (default: {_DEFAULT_OUTPUT_DIR})")
    p.add_argument("--no-input", action="store_true", help="never prompt; use flags/defaults only")
    p.add_argument("-f", "--force", action="store_true", help="overwrite existing files")


def _prompt_text(label: str, default: str | None) -> str | None:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw or default


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  please enter a number")


def _collect_config(args: argparse.Namespace, interactive: bool) -> RewardsDemoConfig:
    """Build the config; values not given as flags are prompted or defaulted."""
    d = RewardsDemoConfig()  # source of the real defaults

    dataset = args.dataset
    if dataset is None:
        dataset = _prompt_text("Dataset", d.dataset) if interactive else d.dataset
    split = args.split
    if split is None:
        split = _prompt_text("Split", d.split) if interactive else d.split
    video_key = args.video_key
    if video_key is None:
        video_key = _prompt_text("Video key", d.video_key) if interactive else d.video_key
    episodes = args.episodes
    if episodes is None:
        episodes = _prompt_int("Episodes to score", d.episodes) if interactive else d.episodes

    return RewardsDemoConfig(
        dataset=dataset or d.dataset,
        split=split or d.split,
        video_key=video_key or d.video_key,
        episodes=episodes,
        max_frames=args.max_frames if args.max_frames is not None else d.max_frames,
    )


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if not args.no_input and not sys.stdin.isatty():
        print(
            "error: no interactive terminal detected. Re-run with --no-input (plus any "
            "flags, e.g. --dataset/--episodes) to generate non-interactively.",
            file=sys.stderr,
        )
        return 2

    interactive = not args.no_input
    if interactive:
        print(_INTRO)

    try:
        config = _collect_config(args, interactive)
        config.validate()
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _prompt_text("Output directory", _DEFAULT_OUTPUT_DIR) if interactive else _DEFAULT_OUTPUT_DIR
    out_dir = Path(output_dir or _DEFAULT_OUTPUT_DIR)

    have_script = args.format in ("script", "all")
    have_nb = args.format in ("notebook", "all")
    have_md = args.format in ("markdown", "all")
    script_path, nb_path, md_path = out_dir / "demo.py", out_dir / "demo.ipynb", out_dir / "demo.md"

    written: list[Path] = []
    try:
        if have_script:
            _write(script_path, render_script(config), args.force)
            written.append(script_path)
        if have_nb:
            _write(nb_path, render_notebook(config), args.force)
            written.append(nb_path)
        if have_md:
            _write(md_path, render_markdown(config), args.force)
            written.append(md_path)
        if args.server_scripts:
            for name in SERVER_TEMPLATES:
                _write(out_dir / name, load_server_script(name), args.force)
                written.append(out_dir / name)
    except FileExistsError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"Created rewards demo ({config.episodes} episodes of {config.dataset}):")
    for path in written:
        print(f"  {path}")

    if args.server_scripts:
        print("\nStart a Robometer eval server (one of):")
        print(f"  python {out_dir / 'run_robometer_server.py'}   # any NVIDIA GPU (A10G/L4 fits the 4B bf16)")
        print(f"  uvx modal deploy {out_dir / 'modal_eval_server.py'}   # Modal (uvx modal setup first)")
    print("\nThen run the demo against it (deps fetched on the fly, nothing to install):")
    withs = "--with daft-physical-ai --with huggingface_hub --with matplotlib"
    if have_script:
        print(f"  ROBOMETER_URL=http://... uv run {withs} {script_path}")
    if have_nb:
        print(f"  ROBOMETER_URL=http://... uvx --from jupyterlab {withs} jupyter-lab {nb_path}")
        print("  (or open demo.ipynb in your code editor, e.g. VS Code)")
    print("  (a Modal proxy-auth deployment also needs MODAL_KEY / MODAL_SECRET set)")
    if have_md:
        print(f"  ({md_path} is a readable walkthrough - not executable)")
    return 0
