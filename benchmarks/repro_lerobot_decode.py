"""Reproduce slow per-frame video decode in `daft.datasets.lerobot` (upstream Daft).

The decode UDF re-opens and re-parses the MP4 container once per frame, so cost
scales ~linearly at ~3s/frame (mostly `av.open()`). Fix: open each shard once and
decode its frames in a single pass.

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
    df.select("episode_index", "frame_index", IMAGE_COLUMN).to_pydict()  # materialize -> decode


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce slow per-frame LeRobot video decode.")
    ap.add_argument("--rows", type=int, default=1, help="frames to decode (default: 1)")
    ap.add_argument("--profile", action="store_true", help="cProfile the run (top cumulative calls)")
    args = ap.parse_args()

    import daft

    if args.profile:
        import cProfile
        import io
        import pstats

        pr = cProfile.Profile()
        t = time.perf_counter()
        pr.runcall(decode_rows, args.rows)
        dt = time.perf_counter() - t
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
        print(s.getvalue())
    else:
        t = time.perf_counter()
        decode_rows(args.rows)
        dt = time.perf_counter() - t

    print(f"daft {daft.__version__}: decoded {args.rows} frame(s) in {dt:.2f}s  ({dt / args.rows:.2f}s/frame)")


if __name__ == "__main__":
    main()
