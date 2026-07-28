"""`daft-physical-ai trim`: scaffold a personalized motion-trimming demo.

Same shape as `hands` and `rewards`: ask a few questions (dataset, state
column, ...), then generate a runnable script/notebook/markdown demo that finds
dead frames from proprioception and reduces them to trim windows. Flags
pre-fill answers; with `--no-input` it runs non-interactively. No server
scripts here - the pipeline never decodes video, so there is nothing to serve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .._render_trim import TrimDemoConfig, render_markdown, render_notebook, render_script

_DEFAULT_OUTPUT_DIR = "motion-trim-demo"

_INTRO = (
    "daft-physical-ai trim - scaffold a motion-trimming demo.\n"
    "Answer a few questions to generate a runnable script + notebook that finds\n"
    "an episode's dead frames from the robot's joint positions (no video decode)\n"
    "and reduces them to one trim window per episode. Press Enter to accept the\n"
    "[default] shown for any question.\n"
)

_DESCRIPTION = "Scaffold a personalized motion-trimming demo (script + notebook + markdown)."


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach the `trim` subcommand to the top-level parser."""
    p = subparsers.add_parser("trim", help=_DESCRIPTION.rstrip(".").lower(), description=_DESCRIPTION)
    p.set_defaults(func=run)
    # Defaults are None so we can tell "user passed this" from "fall back to the default";
    # an explicitly-passed flag is never re-prompted. The real defaults live in TrimDemoConfig.
    p.add_argument("--dataset", help="LeRobot v3 dataset id (default: lerobot/droid_1.0.1)")
    p.add_argument("--state", help="state column to score (default: observation.state.joint_position)")
    p.add_argument("--dims", type=int, help="how many dims that column holds (default: 7)")
    p.add_argument("--fps", type=float, help="dataset frame rate (default: 15)")
    p.add_argument("--shards", type=int, help="data files to read (default: 1)")
    p.add_argument(
        "--format",
        choices=("script", "notebook", "markdown", "all"),
        default="all",
        help="what to generate: script (.py), notebook (.ipynb), markdown (.md), or all (default)",
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


def _collect_config(args: argparse.Namespace, interactive: bool) -> TrimDemoConfig:
    """Build the config; values not given as flags are prompted or defaulted."""
    d = TrimDemoConfig()  # source of the real defaults

    dataset = args.dataset
    if dataset is None:
        dataset = _prompt_text("Dataset", d.dataset) if interactive else d.dataset
    state = args.state
    if state is None:
        state = _prompt_text("State column", d.state) if interactive else d.state
    dims = args.dims
    if dims is None:
        dims = _prompt_int("Dims in that column", d.dims) if interactive else d.dims
    shards = args.shards
    if shards is None:
        shards = _prompt_int("Data shards to read", d.shards) if interactive else d.shards

    return TrimDemoConfig(
        dataset=dataset or d.dataset,
        state=state or d.state,
        dims=dims,
        fps=args.fps if args.fps is not None else d.fps,
        shards=shards,
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
            "flags, e.g. --dataset/--shards) to generate non-interactively.",
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
    except FileExistsError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"Created trim demo ({config.shards} shard(s) of {config.dataset}):")
    for path in written:
        print(f"  {path}")

    print("\nRun it (deps fetched on the fly, nothing to install):")
    withs = "--with daft-physical-ai --with matplotlib"
    if have_script:
        print(f"  uv run {withs} {script_path}")
    if have_nb:
        print(f"  uvx --from jupyterlab {withs} jupyter-lab {nb_path}")
        print("  (or open demo.ipynb in your code editor, e.g. VS Code)")
    if have_md:
        print(f"  ({md_path} is a readable walkthrough - not executable)")
    return 0
