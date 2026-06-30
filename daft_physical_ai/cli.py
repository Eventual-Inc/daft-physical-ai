"""`daft-physical-ai` console entry point: scaffold a personalized demo.

`npm create`-style: ask a few questions (method, runtime, mano_path, ...), then
generate a runnable script and/or notebook tailored to those choices and print the
command to run it. Flags pre-fill answers; with every needed answer supplied (or
`--no-input`) it runs non-interactively, which is also how it's unit-tested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._render import _VALID_METHODS, _VALID_RUNTIMES, DemoConfig, render_notebook, render_script


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="daft-physical-ai",
        description="Scaffold a personalized hand-tracking demo (script + notebook).",
    )
    p.add_argument("--method", choices=_VALID_METHODS, help="which tracker(s): mediapipe, wilor, or both")
    p.add_argument("--runtime", choices=_VALID_RUNTIMES, help="where inference runs: local (GPU/CPU) or modal")
    p.add_argument("--mano-path", help="path to MANO_RIGHT.pkl (required for wilor/both)")
    # Defaults are None so we can tell "user passed this" from "fall back to the default";
    # an explicitly-passed flag is never re-prompted. The real defaults live in DemoConfig.
    p.add_argument("--dataset", help="LeRobot dataset id or path")
    p.add_argument("--image-column", help="camera column to decode")
    p.add_argument("--limit", type=int, help="number of frames to annotate")
    p.add_argument("--format", choices=("script", "notebook", "both"), default="both", help="what to generate")
    p.add_argument("--output-dir", default="hand-tracking-demo", help="directory to write the demo into")
    p.add_argument("--no-input", action="store_true", help="never prompt; use flags/defaults only")
    p.add_argument("-f", "--force", action="store_true", help="overwrite existing files")
    return p


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

    return DemoConfig(
        method=method,
        runtime=runtime,
        mano_path=mano_path,
        dataset=dataset or d.dataset,
        image_column=image_column or d.image_column,
        limit=args.limit if args.limit is not None else d.limit,
    )


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    interactive = not args.no_input and sys.stdin.isatty()

    try:
        config = _collect_config(args, interactive)
        config.validate()
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    written: list[Path] = []
    try:
        if args.format in ("script", "both"):
            script = out_dir / "demo.py"
            _write(script, render_script(config), args.force)
            written.append(script)
        if args.format in ("notebook", "both"):
            nb = out_dir / "demo.ipynb"
            _write(nb, render_notebook(config), args.force)
            written.append(nb)
    except FileExistsError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"Created {config.method} demo ({config.runtime} runtime):")
    for path in written:
        print(f"  {path}")

    have_script = args.format in ("script", "both")
    have_nb = args.format in ("notebook", "both")
    script_path = out_dir / "demo.py"
    nb_path = out_dir / "demo.ipynb"

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
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
