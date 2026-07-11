"""Pose rates without leaving the query plan — window expressions end to end.

The distributed twin of ``pose_features_numpy.py``: instead of collecting each
episode into NumPy, the per-frame geometry runs as a ``@daft.func`` over the
state column and every temporal rate is a Daft window expression
(``lead(1).over(partition_by(episode).order_by(frame))``). Scenario thresholds
then become plain column predicates, so "which frames are grasping/lifting"
is a ``where`` — lazy from the reader to the single terminal collect, on any
per-frame table with the same columns.
"""

from __future__ import annotations

import argparse

from daft import col
from daft.datasets import lerobot

from daft_physical_ai.pose import add_temporal_features, state_frame_features
from daft_physical_ai.pose.query import GRASP_RATE, LIFT_VEL


def main(dataset: str = "pepijn223/egodex-test") -> None:
    frames = lerobot.read(dataset).select(
        "episode_index", "frame_index", state_frame_features(col("observation.state"))
    )
    rates = add_temporal_features(frames, episode_keys=("episode_index",))

    summary = (
        rates.groupby("episode_index")
        .agg(
            col("frame_index").count().alias("frames"),
            (col("curl_rate_L") <= -GRASP_RATE).cast("int64").sum().alias("grasping_L"),
            (col("curl_rate_R") <= -GRASP_RATE).cast("int64").sum().alias("grasping_R"),
            (col("wrist_vert_vel_L") >= LIFT_VEL).cast("int64").sum().alias("lifting_L"),
            (col("wrist_vert_vel_R") >= LIFT_VEL).cast("int64").sum().alias("lifting_R"),
        )
        .sort("episode_index")
    )

    data = summary.to_pydict()  # the one and only collect
    print(f"{'ep':3s} {'frames':>6s} {'grasping L/R':>13s} {'lifting L/R':>12s}")
    for i, episode in enumerate(data["episode_index"]):
        print(
            f"{episode:<3d} {data['frames'][i]:>6d} "
            f"{data['grasping_L'][i]:6d}/{data['grasping_R'][i]:<6d} "
            f"{data['lifting_L'][i]:5d}/{data['lifting_R'][i]:<6d}"
        )

    print(
        "\nEverything above — geometry UDF, lead() rates, scenario thresholds, groupby —"
        "\nstayed in one lazy plan until the final collect."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="In-DAG pose rates + scenario predicates.")
    parser.add_argument("--dataset", default="pepijn223/egodex-test", help="LeRobot v3 repo id or path")
    args = parser.parse_args()
    main(args.dataset)
