"""How much of a demonstration dataset is nobody moving?

OpenVLA's LIBERO fine-tunes train on a hand-built "no-noops" variant of the
demonstrations, so idle steps are folklore-famous. `operations.motion_trim`
turns that cleaning into a one-groupby audit you can point at any step-row
table. On the committed LIBERO-Spatial originals the verdict is the
interesting part: only ~0.2% of steps are strict no-ops - this suite barely
needed the cleaning. The audit, not the assumption, is the point; run it on
your own teleop data before deciding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import daft
from daft import col

from daft_physical_ai.operations import motion_trim

DATA_DIR = Path(__file__).resolve().parents[1] / "02_episode_data" / "data" / "libero_spatial_demos"


def main() -> int:
    parser = argparse.ArgumentParser(description="Idle-step statistics for demonstration data.")
    parser.add_argument("--rows", default=str(DATA_DIR / "*.parquet"), help="step-row parquet glob")
    parser.add_argument("--atol", type=float, default=1e-3, help="motion norm below this is idle")
    args = parser.parse_args()

    df = daft.read_parquet(args.rows)
    spans = motion_trim(df, atol=args.atol)

    per_task = (
        spans.groupby("task_name")
        .agg(
            col("episode_id").count().alias("demos"),
            col("noop_fraction").mean().alias("noop_fraction"),
            col("trim_fraction").mean().alias("trim_fraction"),
            col("frames_removed").sum().alias("frames_removed"),
            col("num_steps").sum().alias("steps"),
        )
        .sort("noop_fraction", desc=True)
        .to_pydict()
    )

    print(f"{'task':70s} {'demos':>5s} {'no-op%':>7s} {'trim%':>6s}")
    for i in range(len(per_task["task_name"])):
        print(
            f"{per_task['task_name'][i]:70s} {per_task['demos'][i]:5d} "
            f"{per_task['noop_fraction'][i]:7.1%} {per_task['trim_fraction'][i]:6.1%}"
        )

    total = spans.to_pydict()
    steps = sum(total["num_steps"])
    noops = sum(round(fraction * n) for fraction, n in zip(total["noop_fraction"], total["num_steps"]))
    trimmed = sum(total["frames_removed"])
    print(
        f"\n{len(total['episode_id'])} demos / {steps} steps: "
        f"{noops / steps:.1%} are no-ops ({noops} steps); prefix/suffix trim alone removes "
        f"{trimmed / steps:.1%}. That's what a 'no-noops' filter would drop here - "
        f"the audit that tells you whether your dataset needs the cleaning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
