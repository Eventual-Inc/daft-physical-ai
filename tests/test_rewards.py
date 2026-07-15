from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import daft
import numpy as np
import pytest

from daft_physical_ai.rewards import REWARD_DTYPE, score_rewards
from daft_physical_ai.rewards._robometer import build_request, parse_response, sample_indexes

FPS = 20
N_FRAMES = 10


def test_sample_indexes_short_episode_returns_all() -> None:
    assert sample_indexes(5, max_frames=8) == [0, 1, 2, 3, 4]


def test_sample_indexes_uniform_with_endpoints() -> None:
    idxs = sample_indexes(100, max_frames=8)
    assert len(idxs) == 8
    assert idxs[0] == 0
    assert idxs[-1] == 99
    assert idxs == sorted(idxs)


def test_sample_indexes_max_frames_param() -> None:
    assert len(sample_indexes(100, max_frames=4)) == 4
    assert sample_indexes(100, max_frames=1) == [99]
    assert sample_indexes(0) == []


def test_build_request_round_trips_frames() -> None:
    frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    npy, sample_json = build_request(frames, "pick up the mug")
    assert np.array_equal(np.load(io.BytesIO(npy)), frames)
    sample = json.loads(sample_json)
    assert sample["trajectory"]["task"] == "pick up the mug"
    assert sample["trajectory"]["frames_shape"] == [3, 4, 4, 3]
    assert sample["trajectory"]["metadata"]["subsequence_length"] == 3


def test_parse_response() -> None:
    out = {
        "outputs_progress": {"progress_pred": [[0.1, 0.9]]},
        "outputs_success": {"success_probs": [[0.2, 0.8]]},
    }
    assert parse_response(out) == ([0.1, 0.9], [0.2, 0.8])


def _write_video(path: Path, n_frames: int = N_FRAMES) -> None:
    """Write a small mp4 whose frames have known content and timestamps."""
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            img = np.full((32, 32, 3), i * 10, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class _FakeRobometer(BaseHTTPRequestHandler):
    """Responds to any POST with a canned Robometer-shaped payload."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        n = sample_indexes(N_FRAMES, max_frames=8)
        body = json.dumps(
            {
                "outputs_progress": {"progress_pred": [[i / max(len(n) - 1, 1) for i in range(len(n))]]},
                "outputs_success": {"success_probs": [[0.5] * len(n)]},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # keep test output clean
        pass


@pytest.fixture
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeRobometer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join()


def test_score_rewards_end_to_end(tmp_path: Path, fake_server: str) -> None:
    video = tmp_path / "episode.mp4"
    _write_video(video)

    df = daft.from_pydict(
        {
            "task": ["pick up the mug"],
            "length": [N_FRAMES],
            "from_ts": [0.0],
            "to_ts": [N_FRAMES / FPS],
            "video_path": [str(video)],
        }
    )
    df = df.with_column(
        "rewards",
        score_rewards(df["task"], df["length"], df["from_ts"], df["to_ts"], df["video_path"], url=fake_server),
    )

    field = next(f for f in df.schema() if f.name == "rewards")
    assert str(field.dtype) == str(REWARD_DTYPE)

    rewards = df.to_pydict()["rewards"][0]
    expected_idxs = sample_indexes(N_FRAMES, max_frames=8)
    assert [f["index"] for f in rewards["reward_frames"]] == expected_idxs
    assert len(rewards["reward_score"]) == len(expected_idxs)
    assert rewards["reward_score"][0] == 0.0
    assert rewards["reward_score"][-1] == 1.0
    assert all(s == 0.5 for s in rewards["robometer_success"])
    # timestamps line up with the video's frame clock
    assert rewards["reward_frames"][0]["timestamp_s"] == pytest.approx(0.0)
    assert rewards["reward_frames"][-1]["timestamp_s"] == pytest.approx((N_FRAMES - 1) / FPS)


def test_score_rewards_max_frames(tmp_path: Path, fake_server: str) -> None:
    video = tmp_path / "episode.mp4"
    _write_video(video)

    df = daft.from_pydict(
        {
            "task": ["pick up the mug"],
            "length": [N_FRAMES],
            "from_ts": [0.0],
            "to_ts": [N_FRAMES / FPS],
            "video_path": [str(video)],
        }
    )
    df = df.with_column(
        "rewards",
        score_rewards(
            df["task"],
            df["length"],
            df["from_ts"],
            df["to_ts"],
            df["video_path"],
            url=fake_server,
            max_frames=3,
        ),
    )
    frames = df.to_pydict()["rewards"][0]["reward_frames"]
    assert [f["index"] for f in frames] == sample_indexes(N_FRAMES, max_frames=3)
