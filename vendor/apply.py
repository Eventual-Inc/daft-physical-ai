"""Overlay the vendored progress-bar fd-leak fix onto the installed daft.

`vendor/progress_bar.py` is `daft/runners/progress_bar.py` from
Eventual-Inc/Daft#7262 (commit 17aa5558b): SwordfishProgressBar routes all tqdm
writes through a single long-lived Python writer thread instead of writing from
the calling Rust threads. Under ipykernel each writing thread identity costs a
~2-fd zmq pipe that is never reclaimed, so the original leaks fds until the
kernel panics with `Too many open files`
(Eventual-Inc/daft-physical-ai#24, Eventual-Inc/Daft#7253). Until that PR
ships in a release, apply it on top of the installed daft.

    python vendor/apply.py            # back up original, copy fix in
    python vendor/apply.py --status   # which progress bar is installed
    python vendor/apply.py --restore  # put the original back

The overlay lives in site-packages, so reinstalling/upgrading daft removes it.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

VENDORED = Path(__file__).with_name("progress_bar.py")
# The fix funnels writes through one writer thread; the original writes from
# the calling thread. Cheap fingerprint:
FIXED_MARKER = "daft-progress-bar-writer"


def installed_path() -> Path:
    spec = importlib.util.find_spec("daft.runners.progress_bar")
    if spec is None or spec.origin is None:
        raise SystemExit("daft.runners.progress_bar not found - install daft first")
    return Path(spec.origin)


def status(target: Path) -> str:
    return "single-writer (fix applied)" if FIXED_MARKER in target.read_text() else "per-thread (original)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="report which progress bar is installed")
    group.add_argument("--restore", action="store_true", help="restore the original progress bar")
    args = parser.parse_args()

    target = installed_path()
    backup = target.with_suffix(".py.orig")

    if args.status:
        print(f"{target}\n-> {status(target)}")
        return

    if args.restore:
        if not backup.exists():
            raise SystemExit(f"no backup at {backup} - nothing to restore")
        shutil.copy2(backup, target)
        print(f"restored original -> {status(target)}")
        return

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"backed up original to {backup.name}")
    shutil.copy2(VENDORED, target)
    print(f"applied vendored fix -> {status(target)}")


if __name__ == "__main__":
    main()
