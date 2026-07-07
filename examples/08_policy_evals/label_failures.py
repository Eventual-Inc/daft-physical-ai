"""Label the real rollout failures from per-step signals - no simulator, no VLM.

Every failure in the committed LIBERO-Spatial rollouts is `terminal_failure =
"unlabeled"`: the harness records what happened, not why. This script names
each failure from signals the writer already captured - gripper command/state
and end-effector height - via `daft_physical_ai.evals.classify_failure`.

The committed rollout parquet is never rewritten (it stays byte-faithful to
what the harness generated); labels are printed and optionally written to a
sidecar parquet that joins back on `episode_id` + `policy_type`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import daft

from daft_physical_ai.evals import label_failures

DATA_DIR = Path(__file__).resolve().parent / "data" / "rollouts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Label failed rollouts from per-step signals.")
    parser.add_argument("--rollouts", default=str(DATA_DIR / "*.parquet"), help="rollout parquet glob")
    parser.add_argument("--out", help="optional path for a labels sidecar parquet")
    args = parser.parse_args()

    df = daft.read_parquet(args.rollouts)
    labels = label_failures(df).sort(["policy_type", "episode_id"])
    data = labels.to_pydict()

    mix: dict[str, Counter[str]] = {}
    for policy, label in zip(data["policy_type"], data["terminal_failure"]):
        mix.setdefault(policy, Counter())[label] += 1

    print("Failure taxonomy (from gripper/eef signals):")
    for policy, counts in sorted(mix.items()):
        total = sum(counts.values())
        parts = ", ".join(f"{label} {count}" for label, count in counts.most_common())
        print(f"  {policy:9s} {total:2d} failures: {parts}")

    print("\nPer episode:")
    for i, episode_id in enumerate(data["episode_id"]):
        print(
            f"  {data['policy_type'][i]:9s} {episode_id:26s} {data['terminal_failure'][i]:14s} "
            f"close_cycles={data['close_cycles'][i]:2d} held={data['held_frac'][i]:4.0%} "
            f"lift={data['max_lift'][i]:.3f}m steps={data['steps'][i]}"
        )

    regrasp_cycles = [c for c, label in zip(data["close_cycles"], data["terminal_failure"]) if label == "re_grasp"]
    if regrasp_cycles:
        mean_cycles = sum(regrasp_cycles) / len(regrasp_cycles)
        print(
            f"\n{len(regrasp_cycles)}/{len(data['episode_id'])} failures are slip-then-regrasp loops, "
            f"averaging {mean_cycles:.0f} grasp attempts before the step cap."
        )

    if args.out:
        labels.write_parquet(args.out)
        print(f"labels sidecar -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
