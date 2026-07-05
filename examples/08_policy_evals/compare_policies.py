"""Pair two policies on the exact same LIBERO specs and find where they differ.

`episode_id` names the evaluation spec (`suite/task_id/init_state_id/seed`),
so both policies' attempts at the same task layout join into one row. That
turns "OpenVLA scored 84%, VLA-JEPA scored 99%" into the question a researcher
actually has: *which* layouts separate them, and what happened there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import daft

from daft_physical_ai.evals import compare_policies

DATA_DIR = Path(__file__).resolve().parent / "data" / "rollouts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-spec comparison of two policies.")
    parser.add_argument("--rollouts", default=str(DATA_DIR / "*.parquet"), help="rollout parquet glob")
    parser.add_argument("--left", default="openvla")
    parser.add_argument("--right", default="vla_jepa")
    args = parser.parse_args()

    df = daft.read_parquet(args.rollouts)
    paired = compare_policies(df, args.left, args.right).sort("episode_id").to_pydict()

    n = len(paired["episode_id"])
    both = sum(1 for a, b in zip(paired["success_left"], paired["success_right"]) if a and b)
    neither = sum(
        1 for a, b in zip(paired["success_left"], paired["success_right"]) if not a and not b
    )
    left_only = sum(1 for a, b in zip(paired["success_left"], paired["success_right"]) if a and not b)
    right_only = sum(1 for a, b in zip(paired["success_left"], paired["success_right"]) if b and not a)

    print(f"{n} shared specs: {args.left} vs {args.right}")
    print(f"  both succeed   {both:3d}")
    print(f"  only {args.left:9s} {left_only:3d}")
    print(f"  only {args.right:9s} {right_only:3d}")
    print(f"  both fail      {neither:3d}")

    print(f"\nSpecs where {args.right} succeeds and {args.left} fails:")
    for episode_id, task_name, ok_l, ok_r, steps_l, steps_r in zip(
        paired["episode_id"],
        paired["task_name"],
        paired["success_left"],
        paired["success_right"],
        paired["num_steps_left"],
        paired["num_steps_right"],
    ):
        if ok_r and not ok_l:
            print(f"  {episode_id:26s} {args.left} ran {steps_l} steps, {args.right} {steps_r}  ({task_name})")

    print("\nLabel these failures from per-step signals: python examples/08_policy_evals/mine_failures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
