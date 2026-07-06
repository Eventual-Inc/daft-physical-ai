"""Write a curated training set: filtered step rows + the manifest describing them.

Curation output is a *view*, not a copy: the manifest names which episodes made
the cut (and why); the step rows carry the signals; media stays wherever the
source dataset keeps it. Three artifacts from the committed data:

1. ``sft_steps/`` - demonstration step rows worth imitating (successes,
   motion-trimmed to each episode's active window), the input to stage 07.
2. ``sft_manifest.parquet`` - one row per kept episode with its trim window.
3. ``preference_pairs.parquet`` - (chosen, rejected) specs from the rollout
   comparison, ready for contrastive/DPO-style consumers.

Outputs are deterministic from committed inputs, so they are gitignored -
regenerate them anytime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import daft

from daft_physical_ai.curation import preference_pairs, sft_view
from daft_physical_ai.operations import motion_trim

EXAMPLES = Path(__file__).resolve().parents[1]
DEMOS = EXAMPLES / "02_episode_data" / "data" / "libero_spatial_demos"
ROLLOUTS = EXAMPLES / "08_policy_evals" / "data" / "rollouts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write curated training artifacts.")
    parser.add_argument("--demos", default=str(DEMOS / "*.parquet"))
    parser.add_argument("--rollouts", default=str(ROLLOUTS / "*.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("curated-training-set"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    demos = daft.read_parquet(args.demos)
    spans = motion_trim(demos)
    curated = sft_view(demos, trim_spans=spans)

    steps_dir = args.out_dir / "sft_steps"
    curated.write_parquet(str(steps_dir))
    n_steps = daft.read_parquet(str(steps_dir) + "/*.parquet").count_rows()
    n_source = demos.count_rows()
    print(f"sft_steps: {n_steps} step rows from {n_source} (motion-trimmed successes) -> {steps_dir}/")

    manifest_path = args.out_dir / "sft_manifest.parquet"
    spans.write_parquet(str(manifest_path))
    print(f"sft_manifest: one row per kept episode with its [start_step, end_step] window -> {manifest_path}")

    pairs = preference_pairs(daft.read_parquet(args.rollouts), "openvla", "vla_jepa")
    pairs_path = args.out_dir / "preference_pairs.parquet"
    pairs.write_parquet(str(pairs_path))
    pairs_data = pairs.to_pydict()
    print(
        f"preference_pairs: {len(pairs_data['episode_id'])} (chosen, rejected) specs "
        f"from the policy comparison -> {pairs_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
