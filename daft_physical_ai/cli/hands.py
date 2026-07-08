"""`daft-physical-ai hands`: scaffold a personalized hand-tracking demo.

`npm create`-style: ask a few questions (method, runtime, mano_path, ...), then
generate a runnable script and/or notebook tailored to those choices and print the
command to run it. Flags pre-fill answers; with every needed answer supplied (or
`--no-input`) it runs non-interactively, which is also how it's unit-tested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .._render import (
    _VALID_METHODS,
    _VALID_RUNTIMES,
    DemoConfig,
    render_markdown,
    render_notebook,
    render_script,
)

_DEFAULT_OUTPUT_DIR = "hand-tracking-demo"

_INTRO = (
    "daft-physical-ai hands - scaffold a hand-tracking demo.\n"
    "Answer a few questions to generate a runnable script + notebook (using\n"
    "track_hands on a LeRobot dataset) that you can run or edit. Press Enter to accept\n"
    "the [default] shown for any question.\n"
)

_DESCRIPTION = "Scaffold a personalized hand-tracking demo (script + notebook)."


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach the `hands` subcommand to the top-level parser."""
    p = subparsers.add_parser("hands", help=_DESCRIPTION.rstrip(".").lower(), description=_DESCRIPTION)
    p.set_defaults(func=run)
    p.add_argument("--method", choices=_VALID_METHODS, help="which tracker(s): mediapipe, wilor, or both")
    p.add_argument("--runtime", choices=_VALID_RUNTIMES, help="where inference runs: local (GPU/CPU) or modal")
    p.add_argument("--mano-path", help="path to MANO_RIGHT.pkl (required for wilor/both)")
    # Defaults are None so we can tell "user passed this" from "fall back to the default";
    # an explicitly-passed flag is never re-prompted. The real defaults live in DemoConfig.
    p.add_argument("--dataset", help="LeRobot dataset id or path")
    p.add_argument("--image-column", help="camera column to decode")
    p.add_argument("--limit", type=int, help="number of frames to annotate")
    p.add_argument(
        "--with-eval",
        action="store_true",
        default=None,
        help="append EgoDex ground-truth scoring (detect%% + PCK) to the demo (local runtime only)",
    )
    p.add_argument(
        "--format",
        choices=("script", "notebook", "markdown", "all"),
        default="all",
        help="what to generate: script (.py), notebook (.ipynb), markdown (.md), or all (default)",
    )
    p.add_argument("--output-dir", help="directory to write the demo into (default: hand-tracking-demo)")
    p.add_argument("--no-input", action="store_true", help="never prompt; use flags/defaults only")
    p.add_argument("-f", "--force", action="store_true", help="overwrite existing files")


def _prompt_choice(label: str, choices: tuple[str, ...], default: str) -> str:
    opts = "/".join(c if c != default else c.upper() for c in choices)
    while True:
        raw = input(f"{label} [{opts}]: ").strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"  please choose one of: {', '.join(choices)}")


def _prompt_text(label: str, default: str | None) -> str | None:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw or default


def _prompt_yes_no(label: str, default: bool) -> bool:
    opts = "Y/n" if default else "y/N"
    raw = input(f"{label} [{opts}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _collect_config(args: argparse.Namespace, interactive: bool) -> DemoConfig:
    """Build the config; values not given as flags are prompted or defaulted."""
    d = DemoConfig()  # source of the real defaults

    method = args.method
    if method is None:
        method = _prompt_choice("Method", _VALID_METHODS, d.method) if interactive else d.method

    # MediaPipe is CPU-only, so Modal adds nothing - it's always local. Runtime only
    # matters when WiLoR (GPU) is involved.
    runtime = args.runtime
    if method == "mediapipe":
        if runtime == "modal":
            print("note: mediapipe runs on CPU; using local runtime (modal isn't needed).", file=sys.stderr)
        runtime = "local"
    elif runtime is None:
        runtime = _prompt_choice("Runtime", _VALID_RUNTIMES, d.runtime) if interactive else d.runtime

    mano_path = args.mano_path
    if method in ("wilor", "both") and not mano_path and interactive:
        mano_path = _prompt_text("Path to MANO_RIGHT.pkl", None)

    dataset = args.dataset
    if dataset is None:
        dataset = _prompt_text("Dataset", d.dataset) if interactive else d.dataset
    image_column = args.image_column
    if image_column is None:
        image_column = _prompt_text("Image column", d.image_column) if interactive else d.image_column

    # Evaluation is local-only and EgoDex-specific; only offered when not on Modal.
    with_eval = bool(args.with_eval)
    if args.with_eval is None and runtime == "local" and interactive:
        with_eval = _prompt_yes_no("Add EgoDex ground-truth evaluation (detect% + PCK)?", default=False)

    return DemoConfig(
        method=method,
        runtime=runtime,
        mano_path=mano_path,
        dataset=dataset or d.dataset,
        image_column=image_column or d.image_column,
        limit=args.limit if args.limit is not None else d.limit,
        with_eval=with_eval,
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
            "flags, e.g. --method/--runtime/--mano-path) to generate non-interactively.",
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

    print(f"Created {config.method} demo ({config.runtime} runtime):")
    for path in written:
        print(f"  {path}")

    if config.runtime == "modal":
        print("\nModal runs remotely - install Modal and log in first:")
        print("  pip install modal && modal setup")
        print("\nRun it with:")
        if have_script:
            print(f"  modal run {script_path}")
        if have_nb:
            print(f"  jupyter lab {nb_path}   # kernel needs Python 3.11 to match the image")
    else:
        print("\nRun it with:")
        if have_script:
            print(f"  python {script_path}")
        if have_nb:
            print(f"  jupyter lab {nb_path}")
    if have_nb:
        print("  (or open demo.ipynb in your code editor, e.g. VS Code)")
    if have_md:
        print(f"  ({md_path} is a readable walkthrough - not executable)")
    return 0
