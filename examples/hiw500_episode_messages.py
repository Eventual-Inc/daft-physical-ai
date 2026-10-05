"""Read HIW-500 humanoid episodes (ROS 2 MCAP) with Daft: catalog, info, subtasks, calibration, states, frames.

HIW-500 is public on Hugging Face. HF_TOKEN (or `hf auth login`) is optional
and only raises the Hub rate limits. Run:

    uv run --extra hiw500 python examples/hiw500_episode_messages.py --task Hang-Hanger

Each stage reads only what it needs: the catalog lists one task's files, info
and calibration download a few small json/yaml files, metadata range-reads the
MCAP summary, and the joint-state / camera reads are bounded to --seconds from
the start of the first annotated episode among the --sample smallest ones.
"""

from __future__ import annotations

from daft import col

from daft_physical_ai.datasets import hiw500


def main(
    path: str = hiw500.HF_DATASET,
    *,
    task: str = "Hang-Hanger",
    sample: int = 20,
    seconds: float = 2.0,
    frames: bool = False,
) -> None:
    episodes = hiw500.raw(path, tasks=task)
    # Some episodes are short or unannotated; pick the smallest one with subtask labels.
    candidates = hiw500.info(episodes.sort("mcap_size").limit(sample)).where(col("subtask_count") > 0)
    one = candidates.sort("mcap_size").limit(1).collect()
    one.select("episode_id", "mcap_size", "task_name", "scene", "duration_seconds", "subtask_count").show()

    hiw500.subtasks(one).select("subtask_index", "label", "start_offset_seconds", "end_offset_seconds").show()
    hiw500.calibration(one).select("camera", "stream", "width", "height", "fx", "fy", "cx", "cy").show()

    summary = hiw500.metadata(one).collect()
    summary.select("ros_distro", "message_count", "chunk_count", "duration_seconds").show()
    start = summary.to_pydict()["message_start_time"][0]
    end = start + int(seconds * 1e9)

    states = hiw500.joint_states(one, start_time=start, end_time=end)
    states.select(
        "log_time",
        "stamp_ns",
        col("q")[hiw500.JOINT_NAMES.index("left_elbow")].alias("left_elbow_q"),
        col("q")[hiw500.JOINT_NAMES.index("right_elbow")].alias("right_elbow_q"),
        "imu_rpy",
    ).sort("log_time").show(5)

    wbc = hiw500.wbc_states(one, start_time=start, end_time=end)
    wbc.select("log_time", "pivot", "left_trigger", "right_trigger").sort("log_time").show(5)

    counts = hiw500.messages(one, start_time=start, end_time=end)
    counts.groupby("topic").agg(col("log_time").count().alias("messages")).sort("topic").show()

    if frames:
        decoded = hiw500.camera_frames(one, start_time=start, end_time=end, width=224, height=224)
        decoded.groupby("camera").agg(col("log_time").count().alias("frames")).sort("camera").show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read HIW-500 MCAP episodes with Daft.")
    parser.add_argument("--path", default=hiw500.HF_DATASET, help="Dataset root (default: the HF dataset)")
    parser.add_argument("--task", default="Hang-Hanger", help="Task folder, e.g. Hang-Hanger (narrows the listing)")
    parser.add_argument("--sample", type=int, default=20, help="How many of the smallest episodes to consider")
    parser.add_argument("--seconds", type=float, default=2.0, help="Window length from the episode start")
    parser.add_argument("--frames", action="store_true", help="Also decode head and wrist camera frames")
    args = parser.parse_args()
    main(args.path, task=args.task, sample=args.sample, seconds=args.seconds, frames=args.frames)
