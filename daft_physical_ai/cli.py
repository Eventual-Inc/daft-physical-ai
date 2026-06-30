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
    p.add_argument("--dataset", default="pepijn223/egodex-test", help="LeRobot dataset id or path")
    p.add_argument("--image-column", default="observation.image", help="camera column to decode")
    p.add_argument("--limit", type=int, default=12, help="number of frames to annotate")
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
    method = args.method
    runtime = args.runtime
    mano_path = args.mano_path
    dataset = args.dataset
    image_column = args.image_column
    limit = args.limit

    if interactive:
        if method is None:
            method = _prompt_choice("Method", _VALID_METHODS, "mediapipe")
        if runtime is None:
            runtime = _prompt_choice("Runtime", _VALID_RUNTIMES, "local")
        if method in ("wilor", "both") and not mano_path:
            mano_path = _prompt_text("Path to MANO_RIGHT.pkl", None)
        dataset = _prompt_text("Dataset", dataset) or dataset
        image_column = _prompt_text("Image column", image_column) or image_column

    return DemoConfig(
        method=method or "mediapipe",
        runtime=runtime or "local",
        mano_path=mano_path,
        dataset=dataset,
        image_column=image_column,
        limit=limit,
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
    print("\nRun it with:")
    script_path = out_dir / "demo.py"
    if config.runtime == "modal":
        print("  pip install modal && modal setup")
        print(f"  modal run {script_path}")
    elif args.format in ("script", "both"):
        print(f"  python {script_path}")
    else:
        print(f"  jupyter notebook {out_dir / 'demo.ipynb'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
