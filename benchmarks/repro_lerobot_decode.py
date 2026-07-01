"""Reproduce slow per-frame video decode in `daft.datasets.lerobot` (upstream Daft).

The decode UDF re-opens the remote MP4 shard once per frame, and each `av.open()`
re-fetches the shard's index over the network - so cost scales ~linearly at
~3s/frame. Parsing and decoding themselves are cheap (see raw_av_decode.py); the
cost is the repeated network fetch. Fix: open each shard once, decode its frames
in a single pass.

    python benchmarks/repro_lerobot_decode.py              # 1 frame  (~6s incl. startup)
    python benchmarks/repro_lerobot_decode.py --rows 8     # 8 frames (~25s)
    python benchmarks/repro_lerobot_decode.py --rows 8 --profile
"""

from __future__ import annotations

import argparse
import os
import time

os.environ["DAFT_PROGRESS_BAR"] = "0"  # avoid the Jupyter/zmq progress-bar overhead

DATASET = "pepijn223/egodex-test"
IMAGE_COLUMN = "observation.image"


def decode_rows(n: int) -> None:
    from daft.datasets import lerobot

    df = lerobot.read(DATASET, load_video_frames=IMAGE_COLUMN).limit(n)
    df.select("episode_index", "frame_index", IMAGE_COLUMN).collect()  # materialize -> decode


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce slow per-frame LeRobot video decode.")
    parser.add_argument("--rows", type=int, default=1, help="frames to decode (default: 1)")
    parser.add_argument("--profile", action="store_true", help="cProfile the run (top cumulative calls)")
    args = parser.parse_args()

    import daft

    if args.profile:
        import cProfile
        import io
        import pstats

        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.runcall(decode_rows, args.rows)
        elapsed = time.perf_counter() - start
        report_buffer = io.StringIO()
        pstats.Stats(profiler, stream=report_buffer).sort_stats("cumulative").print_stats(30)
        print(report_buffer.getvalue())
    else:
        start = time.perf_counter()
        decode_rows(args.rows)
        elapsed = time.perf_counter() - start

    print(
        f"daft {daft.__version__}: decoded {args.rows} frame(s) in {elapsed:.2f}s  ({elapsed / args.rows:.2f}s/frame)"
    )


if __name__ == "__main__":
    main()
