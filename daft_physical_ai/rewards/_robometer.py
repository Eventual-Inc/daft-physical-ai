"""Robometer eval-server client - pure HTTP, no torch/CUDA/weights.

Talks to a running Robometer eval server's ``/evaluate_batch_npy`` endpoint.
The frame sampling and wire format mirror Macrodata refiner's reward pipeline
(``_sample_indexes`` / ``build_payload``), validated frame-exact against its
baseline, so the scores are directly comparable. The server itself is not part
of this package - see the post-2 example for a pinned launcher and a Modal
wrapper.
"""

from __future__ import annotations

import io
import json

import numpy as np


def sample_indexes(length: int, max_frames: int = 8) -> list[int]:
    """Uniformly sample frame indexes across an episode, first + last always included.

    Identical to Macrodata refiner's ``_sample_indexes`` so results diff cleanly
    against its baseline. May return fewer than ``max_frames`` indexes when
    rounding collapses neighbors.
    """
    if length <= 0:
        return []
    if length <= max_frames:
        return list(range(length))
    if max_frames == 1:
        return [length - 1]
    return sorted({round(i * float(length - 1) / float(max_frames - 1)) for i in range(max_frames)})


def decode_frames(video_path: str, from_ts: float, to_ts: float, want: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Decode an episode's segment of a concatenated LeRobot mp4 and pick the wanted frames.

    ``want`` holds frame indexes relative to the episode start (``from_ts``).
    Returns ``(frames [N, H, W, 3] uint8, refs)`` where each ref records the
    frame's relative index and absolute timestamp in seconds.
    """
    import av

    want_set = set(want)
    frames, refs = [], []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            raise ValueError(f"video stream in {video_path} has no time base")
        container.seek(max(0, int((from_ts - 1.0) / time_base)), stream=stream)
        rel = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * time_base)
            if t < from_ts - 1e-6:
                continue
            if t >= to_ts - 1e-6:
                break
            rel = 0 if rel is None else rel + 1
            if rel in want_set:
                frames.append(frame.to_ndarray(format="rgb24"))
                refs.append({"index": rel, "timestamp_s": round(t, 6)})
            if len(frames) == len(want_set):
                break
    if len(frames) != len(want_set):
        raise ValueError(f"decoded {len(frames)}/{len(want_set)} wanted frames in [{from_ts},{to_ts})")
    return np.stack(frames), refs


def build_request(frames: np.ndarray, task: str) -> tuple[bytes, str]:
    """Build one ``/evaluate_batch_npy`` request: the frames as npy bytes plus the sample JSON.

    Mirrors robometer's ``eval_utils.raw_dict_to_sample`` + ``build_payload``
    without importing robometer (no torch client-side). The server's linspace
    subsampling is a no-op because the frames are pre-sampled to the requested
    count.
    """
    buf = io.BytesIO()
    np.save(buf, frames)
    sample = {
        "sample_type": "progress",
        "data_gen_strategy": None,
        "resample_attempts": 1,
        "trajectory": {
            "frames": {"__numpy_file__": "sample_0_trajectory_frames"},
            "frames_shape": list(frames.shape),
            "task": task,
            "lang_vector": None,
            "metadata": {"subsequence_length": int(frames.shape[0])},
            "video_embeddings": None,
            "text_embedding": None,
        },
    }
    return buf.getvalue(), json.dumps(sample)


def parse_response(out: dict) -> tuple[list[float], list[float]]:
    """Extract (per-frame progress, per-frame success probability) from a server response."""
    progress = out["outputs_progress"]["progress_pred"][0]
    success = out["outputs_success"]["success_probs"][0]
    return progress, success


async def post_request(
    url: str,
    npy: bytes,
    sample_json: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 600.0,
) -> dict:
    """POST one request to the eval server's ``/evaluate_batch_npy`` and return the JSON body.

    ``headers`` passes auth through untouched (e.g. ``Modal-Key`` /
    ``Modal-Secret`` for a Modal proxy-auth deployment); the client doesn't
    care what's behind the URL.
    """
    import aiohttp

    form = aiohttp.FormData()
    form.add_field(
        "sample_0_trajectory_frames",
        npy,
        filename="sample_0_trajectory_frames.npy",
        content_type="application/octet-stream",
    )
    form.add_field("sample_0", sample_json)
    form.add_field("use_frame_steps", "false")
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(url.rstrip("/") + "/evaluate_batch_npy", data=form, headers=headers or {}) as resp,
    ):
        resp.raise_for_status()
        return await resp.json()
