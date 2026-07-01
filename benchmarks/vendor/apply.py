"""Overlay the vendored batched-decode LeRobot reader onto the installed daft.

`vendor/lerobot.py` is `daft/datasets/lerobot.py` from Eventual-Inc/Daft#7184
(commit 242b3fdbd): the frame decode is a batch UDF that opens each shard once
per batch instead of re-opening it per frame. Until that PR ships in nightly,
apply it on top of the nightly install this repo's README documents.

    python benchmarks/vendor/apply.py            # back up original, copy fix in
    python benchmarks/vendor/apply.py --status   # which reader is installed
    python benchmarks/vendor/apply.py --restore  # put the original back

The overlay lives in site-packages, so reinstalling/upgrading daft removes it.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

VENDORED = Path(__file__).with_name("lerobot.py")
# The fixed decode is a batch UDF; the original is per-row. Cheap fingerprint:
BATCHED_MARKER = "_decode_one_shard"


def installed_path() -> Path:
    spec = importlib.util.find_spec("daft.datasets.lerobot")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "daft.datasets.lerobot not found - install nightly daft first (see README)"
        )
    return Path(spec.origin)


def status(target: Path) -> str:
    return "batched (fix applied)" if BATCHED_MARKER in target.read_text() else "per-row (original)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="report which reader is installed")
    group.add_argument("--restore", action="store_true", help="restore the original reader")
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
