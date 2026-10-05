"""Read ABC-130k episodes (MCAP) with Daft: catalog, summaries, messages, annotations, frames.

ABC-130k is gated on Hugging Face. Accept the conditions on
https://huggingface.co/datasets/XDOF/ABC-130k (approval is automatic), set
HF_TOKEN (or `hf auth login`), then run:

    uv run --extra abc python examples/abc_episode_messages.py --task clip_the_socks_to_the_hanger

Pass --path to read a local mirror instead. Each stage reads only what it
needs: the catalog lists objects, metadata range-reads MCAP summaries, and the
message/frame reads are bounded to the first --seconds of the smallest episode.
"""

from __future__ import annotations

from daft import col

from daft_physical_ai.datasets import abc


def main(
    path: str = abc.HF_DATASET,
    *,
    split: str | None = "train",
    task: str | None = None,
    seconds: float = 2.0,
    frames: bool = False,
) -> None:
    episodes = abc.raw(path, split=split, tasks=task)  # type: ignore[arg-type]
    one = episodes.sort("episode_size").limit(1).collect()
    one.select("split", "task_slug", "episode_id", "episode_size", "annotated").show()

    summary = abc.metadata(one).collect()
    summary.select("task_name", "duration_seconds", "message_count", "chunk_count", "video_topics").show()
    start = summary.to_pydict()["message_start_time"][0]
    end = start + int(seconds * 1e9)

    states = abc.messages(one, start_time=start, end_time=end)
    states.groupby("topic").agg(col("log_time").count().alias("messages")).sort("topic").show()

    if one.to_pydict()["annotated"][0]:
        abc.annotations(one).select("log_time", "timestamp_ns", "label").show()

    if frames:
        decoded = abc.camera_frames(one, cameras="left_wrist", start_time=start, end_time=end, width=224, height=224)
        decoded.select("camera", "format", "log_time", "is_key_frame", "width", "height").show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read ABC-130k MCAP episodes with Daft.")
    parser.add_argument("--path", default=abc.HF_DATASET, help="Dataset root (default: the gated HF dataset)")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--task", help="Task slug, e.g. clip_the_socks_to_the_hanger (narrows the listing)")
    parser.add_argument("--seconds", type=float, default=2.0, help="Window length from the episode start")
    parser.add_argument("--frames", action="store_true", help="Also decode left-wrist camera frames")
    args = parser.parse_args()
    main(args.path, split=args.split, task=args.task, seconds=args.seconds, frames=args.frames)
