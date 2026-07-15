"""Reward scoring for Daft DataFrames.

`score_rewards(...)` takes episode-metadata columns (Daft expressions) and
returns a reward column (also an expression), so it composes with any Daft
pipeline and runs lazily. Scoring is a pure HTTP call to a Robometer eval
server the user runs - no torch, CUDA, or weights in this package. The output
schema is fixed - see `schema.REWARD_DTYPE`.
"""

from __future__ import annotations

import asyncio

import daft
from daft import Expression

from ._robometer import build_request, decode_frames, parse_response, post_request, sample_indexes
from .schema import REWARD_DTYPE, REWARD_FRAME_DTYPE

__all__ = ["REWARD_DTYPE", "REWARD_FRAME_DTYPE", "score_rewards"]


def score_rewards(
    task: Expression,
    length: Expression,
    from_ts: Expression,
    to_ts: Expression,
    video_path: Expression,
    *,
    url: str,
    max_frames: int = 8,
    headers: dict[str, str] | None = None,
    timeout_s: float = 600.0,
) -> Expression:
    """Score episodes with a reward model: per-frame task progress plus success probability.

    Each row is one episode. Frames are sampled uniformly across the episode
    (first + last always included), decoded from the episode's segment of the
    LeRobot mp4, and sent to a Robometer eval server; the decoded response
    comes back as dataset columns. Downstream uses: filter failed or stalled
    episodes before BC training, dense reward for RL post-training, catching
    mislabeled tasks.

    Args:
        task: task-text column (from the episode's LeRobot task metadata).
        length: episode length column (frame count).
        from_ts: episode start timestamp column (seconds, in the video).
        to_ts: episode end timestamp column (seconds, in the video).
        video_path: path column for the video file holding the episode.
        url: base URL of a running Robometer eval server (local or remote);
            the pipeline doesn't care what's behind it.
        max_frames: how many frames to sample per episode (default 8, matching
            Macrodata refiner; fewer may come back when rounding collapses
            neighbors).
        headers: extra HTTP headers, passed through untouched (e.g. Modal
            proxy-auth tokens).
        timeout_s: per-request timeout in seconds.

    Returns:
        An expression yielding ``struct{reward_score, robometer_success,
        reward_frames}`` per episode - see ``REWARD_DTYPE``.
    """

    @daft.func(return_dtype=REWARD_DTYPE)
    async def _score(task: str, length: int, from_ts: float, to_ts: float, video_path: str) -> dict:
        idxs = sample_indexes(int(length), max_frames)
        # av decode is blocking; keep it off the event loop so requests overlap.
        frames, refs = await asyncio.to_thread(decode_frames, video_path, float(from_ts), float(to_ts), idxs)
        npy, sample_json = build_request(frames, task)
        out = await post_request(url, npy, sample_json, headers=headers, timeout_s=timeout_s)
        progress, success = parse_response(out)
        return {"reward_score": progress, "robometer_success": success, "reward_frames": refs}

    return _score(task, length, from_ts, to_ts, video_path)
