"""Mine policy failures from canonical rollout parquet with Daft.

Writes synthetic OpenVLA / VLA-JEPA rollout failures into the one-row-per-step
schema, scans the parquet glob with Daft, and labels slip-then-regrasp loops
from object height and gripper state. Synthetic on purpose: it proves the
analysis surface end-to-end on CPU; hosted rollouts from a real benchmark run
drop in via the same schema.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from daft_physical_ai.episodes import Episode, Step, write_episode
from daft_physical_ai.evals import detect_regrasp

TARGET_OBJECT = "akita_black_bowl"
SCENARIOS = {
    "openvla": ("regrasp2", "regrasp1", "regrasp2", "regrasp1", "drop"),
    "vla_jepa": ("drop", "drop", "regrasp1", "drop", "nograsp"),
}


def build_signals(scenario: str, *, rng: np.random.Generator) -> tuple[list[bool], list[float]]:
    gripper_closed: list[bool] = []
    object_z: list[float] = []

    def add(n_steps: int, closed: bool, z0: float, z1: float | None = None) -> None:
        end = z0 if z1 is None else z1
        values = np.linspace(z0, end, n_steps) + rng.normal(0, 0.002, n_steps)
        gripper_closed.extend([closed] * n_steps)
        object_z.extend(np.clip(values, 0.0, None).tolist())

    if scenario in ("regrasp1", "regrasp2"):
        add(6, False, 0.0)
        add(1, True, 0.0)
        add(6, True, 0.0, 0.15)
        add(3, True, 0.15, 0.02)
        add(2, False, 0.02)
        add(1, True, 0.02)
        add(7, True, 0.02, 0.16)
        add(3, True, 0.16, 0.02)
        if scenario == "regrasp2":
            add(2, False, 0.02)
            add(1, True, 0.02)
            add(6, True, 0.02, 0.15)
            add(3, True, 0.15, 0.02)
        add(3, False, 0.02)
    elif scenario == "drop":
        add(6, False, 0.0)
        add(1, True, 0.0)
        add(7, True, 0.0, 0.15)
        add(3, True, 0.15, 0.01)
        add(6, False, 0.01)
    elif scenario == "nograsp":
        add(20, False, 0.0)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return gripper_closed, object_z


def make_episode(
    episode_id: str,
    policy_type: str,
    scenario: str,
    *,
    rng: np.random.Generator,
) -> Episode:
    gripper_closed, object_z = build_signals(scenario, rng=rng)
    steps = []
    for step_idx, (closed, z_value) in enumerate(zip(gripper_closed, object_z)):
        action = np.zeros(7, np.float32)
        action[-1] = 1.0 if closed else -1.0
        steps.append(
            Step(
                timestep=step_idx,
                action=action,
                reward=0.0,
                done=step_idx == len(gripper_closed) - 1,
                is_terminal=step_idx == len(gripper_closed) - 1,
                eef_pos=np.array([0.0, 0.0, 0.2 + z_value], np.float32),
                gripper_state=float(0.01 if closed else 0.08),
                object_poses={
                    TARGET_OBJECT: [0.0, 0.0, float(z_value), 0.0, 0.0, 0.0, 1.0]
                },
            )
        )
    return Episode(
        episode_id=episode_id,
        source="rollout",
        instruction="put the bowl on the plate",
        steps=tuple(steps),
        success=False,
        terminal_failure="unlabeled",
        model=f"{policy_type}-demo",
        policy_type=policy_type,
        suite="libero_spatial",
        task_id=0,
        task_name="put_bowl",
    )


def write_synthetic_rollouts(out_dir: Path) -> Path:
    rng = np.random.default_rng(0)
    parquet_dir = out_dir / "rollouts"
    for policy_type, scenarios in SCENARIOS.items():
        for index, scenario in enumerate(scenarios):
            episode = make_episode(
                f"libero_spatial/0/{index}/{policy_type}",
                policy_type,
                scenario,
                rng=rng,
            )
            write_episode(episode, parquet_dir, run_id=f"demo-{policy_type}")
    return parquet_dir


def mine_regrasp_failures(parquet_dir: Path) -> list[dict[str, object]]:
    import daft

    df = daft.read_parquet(str(parquet_dir / "*.parquet"))
    failures = df.where(df["success"] == False).sort(["episode_id", "step_idx"])
    data = failures.to_pydict()

    episode_rows: OrderedDict[str, list[int]] = OrderedDict()
    for row_idx, episode_id in enumerate(data["episode_id"]):
        episode_rows.setdefault(episode_id, []).append(row_idx)

    results: list[dict[str, object]] = []
    for episode_id, indexes in episode_rows.items():
        object_z = [json.loads(data["object_poses"][i])[TARGET_OBJECT][2] for i in indexes]
        gripper_closed = [data["gripper_state"][i] < 0.04 for i in indexes]
        detection = detect_regrasp(object_z, gripper_closed)
        results.append(
            {
                "episode_id": episode_id,
                "policy_type": data["policy_type"][indexes[0]],
                "label": detection.label,
                "object_z": object_z,
                "gripper_closed": gripper_closed,
                "events": detection.events,
            }
        )
    return results


def summarize(results: list[dict[str, object]]) -> tuple[dict[str, Counter], dict[str, float]]:
    by_policy: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        by_policy[str(result["policy_type"])][str(result["label"])] += 1

    rates = {}
    for policy_type, counts in by_policy.items():
        total = sum(counts.values())
        rates[policy_type] = counts["re_grasp"] / total if total else 0.0
    return by_policy, rates


def plot_results(results: list[dict[str, object]], rates: dict[str, float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hero = next(
        result
        for result in results
        if result["policy_type"] == "openvla" and result["label"] == "re_grasp"
    )
    object_z = list(hero["object_z"])
    gripper_closed = list(hero["gripper_closed"])
    events = list(hero["events"])

    fig, (trace_ax, rate_ax) = plt.subplots(
        1,
        2,
        figsize=(13, 4.6),
        gridspec_kw={"width_ratios": [2.3, 1]},
    )

    colors = {"grasp": "#2ca02c", "lift": "#1f77b4", "drop": "#d62728", "re-grasp": "#9467bd"}
    trace_ax.plot(range(len(object_z)), object_z, lw=2.4, color="#1f77b4", label="object height")
    trace_ax.axhline(0.0, ls="--", lw=1, color="gray", label="table")

    in_closed_span = False
    span_start = 0
    for step_idx, closed in enumerate(gripper_closed + [False]):
        if closed and not in_closed_span:
            in_closed_span = True
            span_start = step_idx
        elif not closed and in_closed_span:
            in_closed_span = False
            trace_ax.axvspan(span_start, step_idx, color="#ffce6b", alpha=0.25)
    trace_ax.axvspan(0, 0, color="#ffce6b", alpha=0.25, label="gripper closed")

    for event in events:
        trace_ax.scatter(
            [event.step_idx],
            [object_z[event.step_idx]],
            s=55,
            zorder=5,
            color=colors[event.kind],
            edgecolor="white",
            lw=0.8,
        )
        trace_ax.annotate(
            event.kind,
            (event.step_idx, object_z[event.step_idx]),
            textcoords="offset points",
            xytext=(0, -16 if event.kind == "drop" else 13),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=colors[event.kind],
        )

    trace_ax.set_title("Automatic re-grasp detection", fontweight="bold", fontsize=12)
    trace_ax.set_xlabel("rollout step")
    trace_ax.set_ylabel("object height (m)")
    trace_ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    trace_ax.margins(x=0.02)

    policies = ["openvla", "vla_jepa"]
    ratio = rates["openvla"] / max(rates["vla_jepa"], 1e-9)
    rate_ax.bar(policies, [rates[policy] for policy in policies], color=["#d62728", "#2ca02c"])
    for index, policy in enumerate(policies):
        rate_ax.text(index, rates[policy] + 0.03, f"{rates[policy]:.0%}", ha="center", fontweight="bold")
    rate_ax.set_ylim(0, 1.05)
    rate_ax.set_ylabel("share of failures")
    rate_ax.set_title(f"OpenVLA re-grasps {ratio:.0f}x more often", fontweight="bold", fontsize=12)

    fig.suptitle("A success rate says what happened. Step rows explain why.", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.02,
        "synthetic demo data - proves detection and visualization; real numbers come from real rollouts",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine synthetic re-grasp failures with Daft.")
    parser.add_argument("--out-dir", type=Path, default=Path("failure-mode-demo"))
    parser.add_argument("--no-plot", action="store_true", help="skip matplotlib plot generation")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parquet_dir = write_synthetic_rollouts(args.out_dir)
    results = mine_regrasp_failures(parquet_dir)
    by_policy, rates = summarize(results)

    print(f"{len(results)} failure episodes mined from {parquet_dir}")
    for policy_type in ("openvla", "vla_jepa"):
        counts = by_policy[policy_type]
        total = sum(counts.values())
        print(
            f"{policy_type:9s} re-grasp rate {counts['re_grasp']}/{total} "
            f"= {rates[policy_type]:.0%} mix={dict(counts)}"
        )

    ratio = rates["openvla"] / max(rates["vla_jepa"], 1e-9)
    print(f"OpenVLA failures are {ratio:.0f}x more often slip-then-regrasp loops.")

    if not args.no_plot:
        plot_path = args.out_dir / "regrasp_demo.png"
        plot_results(results, rates, plot_path)
        print(f"hero screenshot -> {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
