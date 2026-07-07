"""Success rates from real LIBERO-Spatial rollouts - overall and per task.

Reads the in-repo rollout parquet (OpenVLA vs VLA-JEPA, 100 episodes each; see
data/README.md) and answers the scoreboard question with two Daft groupbys.
Runs offline on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import daft

from daft_physical_ai.evals import success_rates

DATA_DIR = Path(__file__).resolve().parent / "data" / "rollouts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Success rates per policy and per task.")
    parser.add_argument("--rollouts", default=str(DATA_DIR / "*.parquet"), help="rollout parquet glob")
    args = parser.parse_args()

    df = daft.read_parquet(args.rollouts)

    overall = success_rates(df).sort("policy_type").to_pydict()
    print("Overall success rate (libero_spatial, seed 7, 10 trials/task):")
    for policy, rate, episodes in zip(
        overall["policy_type"], overall["success_rate"], overall["episodes"]
    ):
        print(f"  {policy:9s} {rate:6.1%}  ({episodes} episodes)")

    per_task = success_rates(df, by=("task_id", "task_name", "policy_type")).sort(
        ["task_id", "policy_type"]
    ).to_pydict()

    print("\nPer task:")
    print(f"  {'task':4s} {'openvla':>8s} {'vla_jepa':>9s}  name")
    by_task: dict[int, dict[str, float]] = {}
    names: dict[int, str] = {}
    for task_id, name, policy, rate in zip(
        per_task["task_id"], per_task["task_name"], per_task["policy_type"], per_task["success_rate"]
    ):
        by_task.setdefault(task_id, {})[policy] = rate
        names[task_id] = name
    for task_id in sorted(by_task):
        rates = by_task[task_id]
        print(
            f"  {task_id:<4d} {rates.get('openvla', 0):>8.0%} {rates.get('vla_jepa', 0):>9.0%}  {names[task_id]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
