"""Where should the next data-collection budget go?

Failures are the acquisition function: they mark the (task, init states) where
the policy needs corrective data, and the failure label says what kind. This
example labels the real rollout failures from per-step signals and ranks where
they concentrate - the "collect these next" table that aims the next teleop
session, sim-generation pass, or demonstration mining query.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import daft

from daft_physical_ai.curation import acquisition_map
from daft_physical_ai.evals import label_failures

DATA_DIR = Path(__file__).resolve().parent / "data" / "rollouts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank failures into a collection plan.")
    parser.add_argument("--rollouts", default=str(DATA_DIR / "*.parquet"), help="rollout parquet glob")
    args = parser.parse_args()

    df = daft.read_parquet(args.rollouts)
    ranked = acquisition_map(label_failures(df)).to_pydict()

    print(f"{'policy':9s} {'task':4s} {'failure':14s} {'n':>2s}  init states / task name")
    for i in range(len(ranked["task_id"])):
        init_states = ",".join(str(s) for s in ranked["init_state_ids"][i])
        print(
            f"{ranked['policy_type'][i]:9s} {ranked['task_id'][i]:<4d} "
            f"{ranked['terminal_failure'][i]:14s} {ranked['failures'][i]:2d}  "
            f"[{init_states}] {ranked['task_name'][i]}"
        )

    top = 0
    print(
        f"\nRead as a collection plan: the top row says collect corrective demonstrations for "
        f"task {ranked['task_id'][top]} ({ranked['terminal_failure'][top]} on init states "
        f"{ranked['init_state_ids'][top]}) before anything else."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
