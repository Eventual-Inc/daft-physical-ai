"""`daft-physical-ai` console entry point: dispatch to subcommands.

Each capability is a subcommand in its own module (e.g. `hands`). To add one,
create `cli/<name>.py` exposing `register(subparsers)` (which adds the parser
and sets `func=run` via `set_defaults`) and register it in `main` below.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from . import hands


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="daft-physical-ai",
        description="Physical-AI data processing on Daft.",
    )
    subparsers = p.add_subparsers(dest="command", metavar="<command>")
    hands.register(subparsers)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
