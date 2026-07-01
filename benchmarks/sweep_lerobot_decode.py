"""Sweep the LeRobot decode over row counts, profile each, and chart the growth.

Shows that decode time grows ~linearly with frame count (no amortization) because
the reader re-opens the remote shard once per frame, and each `av.open()` re-fetches
its index over the network (see raw_av_decode.py - parsing/decoding are cheap).
Writes two charts next to this script: total time vs frames, per-function self-time.

    python benchmarks/sweep_lerobot_decode.py            # rows 1..10 (~35s)
"""

from __future__ import annotations

import cProfile
import json
import os
import pstats
import time
from pathlib import Path

os.environ["DAFT_PROGRESS_BAR"] = "0"

DATASET = "pepijn223/egodex-test"
IMAGE_COLUMN = "observation.image"
ROWS = list(range(1, 11))
OUT = Path(__file__).parent


def decode_rows(n: int) -> None:
    from daft.datasets import lerobot

    df = lerobot.read(DATASET, load_video_frames=IMAGE_COLUMN).limit(n)
    df.select("episode_index", "frame_index", IMAGE_COLUMN).collect()


def tottime_for(stats: pstats.Stats, needle: str) -> float:
    """Sum self-time (tottime) across profiled functions whose name/file matches ``needle``."""
    total = 0.0
    for (fn, _ln, func), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        if needle in func or needle in fn:
            total += tt
    return total


def main() -> None:
    import daft

    print(f"daft {daft.__version__}, sweeping rows {ROWS}", flush=True)
    results = []
    for n in ROWS:
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.runcall(decode_rows, n)
        wall = time.perf_counter() - start
        stats = pstats.Stats(profiler)
        row = {
            "rows": n,
            "wall": wall,
            "av_open": tottime_for(stats, "av.container.core.open"),
            "decode_loop": tottime_for(stats, "_decode_lerobot_video_timestamp"),
            "file_read": tottime_for(stats, "_from_file_reference"),
        }
        results.append(row)
        print(
            f"rows={n:2d}  wall={wall:6.2f}s  av_open={row['av_open']:5.2f}  "
            f"decode={row['decode_loop']:5.2f}  file_read={row['file_read']:5.2f}",
            flush=True,
        )
        (OUT / "sweep_results.json").write_text(json.dumps(results, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["rows"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [r["wall"] for r in results], "o-", color="#d62728")
    ax.set_title("Total decode time vs frames (LeRobot reader)")
    ax.set_xlabel("frames decoded")
    ax.set_ylabel("wall time (s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "chart_total.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [r["av_open"] for r in results], "o-", label="av.open (MP4 open/parse)")
    ax.plot(xs, [r["decode_loop"] for r in results], "s-", label="decode loop (self)")
    ax.plot(xs, [r["file_read"] for r in results], "^-", label="file read")
    ax.set_title("Per-function self-time vs frames")
    ax.set_xlabel("frames decoded")
    ax.set_ylabel("tottime (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "chart_functions.png", dpi=130)
    print("charts written", flush=True)


if __name__ == "__main__":
    main()
