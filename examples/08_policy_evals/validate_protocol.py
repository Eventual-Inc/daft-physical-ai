"""Check that rollout parquet faithfully reproduces the LIBERO eval protocol.

"We reproduced the benchmark" becomes a check instead of a claim: task
coverage, trials per task, the eval seed, per-suite step caps, and step-count
consistency, straight off the parquet - no simulator required. This dataset
uses the 10-trials/task quick variant of the protocol; the published canonical
run uses 50 (`daft_physical_ai.evals.TRIALS_PER_TASK`).

Exits non-zero if any policy's run deviates, so it drops into CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import daft

from daft_physical_ai.evals import validate_run

DATA_DIR = Path(__file__).resolve().parent / "data" / "rollouts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate rollout parquet against the LIBERO protocol.")
    parser.add_argument("--rollouts", default=str(DATA_DIR / "*.parquet"), help="rollout parquet glob")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--policies", default="openvla,vla_jepa", help="comma-separated policy_type values")
    parser.add_argument("--trials-per-task", type=int, default=10, help="50 = the published canonical protocol")
    args = parser.parse_args()

    df = daft.read_parquet(args.rollouts)

    all_ok = True
    for policy_type in args.policies.split(","):
        report = validate_run(
            df,
            suite=args.suite,
            policy_type=policy_type.strip(),
            trials_per_task=args.trials_per_task,
        )
        status = "OK" if report.ok else "DEVIATES"
        print(
            f"{report.policy_type:9s} {report.suite}: {status} - "
            f"{report.n_attempts}/{report.expected_attempts} attempts across {report.n_tasks} tasks"
        )
        for issue in report.issues:
            print(f"  [{issue.code}] {issue.message}")
        all_ok = all_ok and report.ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
