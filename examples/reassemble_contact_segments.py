"""Read the original REASSEMBLE HDF5 release with Daft: segments, 1 kHz force/torque, audio, events, frames.

REASSEMBLE ships as one 58.9 GB zip on TU Wien research data (CC-BY-4.0). This
example pulls just the smallest recording out of it (24 MB over HTTP range
requests) unless the directory already holds extracted .h5 files, then runs:

    uv run python examples/reassemble_contact_segments.py /data/reassemble

Each stage reads only what it needs from the HDF5 file: the segment
annotations, the force/torque stream at its native ~1 kHz, decoded audio, the
event stream inside one action, and hand-camera frames inside the same window.
Pass --compare-port to count how many frames the Hugging Face LeRobot port keeps
for the same recording.
"""

from __future__ import annotations

from pathlib import Path

import daft
import numpy as np
from daft import DataType, col

from daft_physical_ai.datasets import reassemble

SMALLEST_RECORDING = "2025-01-11-14-43-37"


@daft.func(return_dtype=DataType.struct({"ft_samples": DataType.int64(), "peak_force_n": DataType.float64()}), unnest=True)
def force_in_window(force: np.ndarray, timestamps: np.ndarray, start: float, end: float) -> dict[str, object]:
    inside = (timestamps >= start) & (timestamps < end)
    magnitude = np.linalg.norm(force[inside], axis=1) if inside.any() else np.zeros(0)
    return {"ft_samples": int(inside.sum()), "peak_force_n": float(magnitude.max()) if magnitude.size else None}


def main(root: str, recording: str | None = None, *, frames: bool = True, compare_port: bool = False) -> None:
    if not any(Path(root).glob("**/*.h5")):
        print(f"No .h5 files under {root}; extracting {SMALLEST_RECORDING} from {reassemble.DATA_ZIP_URL}")
        reassemble.download(SMALLEST_RECORDING, root)

    episodes = reassemble.raw(root, recordings=recording).limit(1).collect()
    name = episodes.to_pydict()["recording"][0]
    print(f"Recording {name}")

    # Hierarchical labels: high-level actions with their low-level skills.
    actions = reassemble.segments(episodes).collect()
    actions.select("segment_index", "text", "success", "duration").show()
    reassemble.segments(episodes, level="low").select("segment_text", "text", "success", "duration").show()

    # Force/torque at the sensor's native rate, summarized per action with a Daft join.
    force = reassemble.robot_state(episodes, fields=["measured_force"])
    per_action = actions.join(force, on="recording").select(
        "segment_index",
        "text",
        force_in_window(col("measured_force"), col("measured_force_timestamps"), col("start"), col("end")),
    )
    per_action.sort("segment_index").show()

    # Three microphones, decoded from the embedded MP3 bitstreams.
    reassemble.audio(episodes).select("microphone", "sample_rate", "channels", "num_samples").show()

    # Event-camera events and hand-camera frames inside the first manipulation action.
    action = actions.where(col("text") != "No action.").sort("segment_index").limit(1).to_pydict()
    if action["start"]:
        start, end = action["start"][0], action["end"][0]
        print(f"Window: {action['text'][0]!r} ({end - start:.1f} s)")
        reassemble.events(episodes, start_time=start, end_time=end).select("num_events").show()
        if frames:
            reassemble.camera_frames(
                episodes, "hand", start_time=start, end_time=min(end, start + 1.0), width=224, height=168
            ).select("camera", "frame_index", "timestamp", "image").show(4)

    if compare_port:
        import os

        from daft.datasets import lerobot
        from daft.io import HuggingFaceConfig, IOConfig

        token = os.environ.get("HF_TOKEN")
        io_config = IOConfig(hf=HuggingFaceConfig(token=token)) if token else None
        port = reassemble.lerobot_sidecars(io_config=io_config).where(col("recording") == name)
        episode = port.select("episode_index").to_pydict()["episode_index"]
        if not episode:
            print(f"{name} is not in the LeRobot port")
            return
        episode_index = episode[0]
        lengths = lerobot.read_episodes(reassemble.HF_LEROBOT_PORT, io_config=io_config)
        port_frames = lengths.where(col("episode_index") == episode_index).to_pydict()["length"][0]
        ft_samples = force.select(col("measured_force_timestamps")).to_pydict()["measured_force_timestamps"][0].size
        print(
            f"LeRobot port episode {episode_index}: {port_frames} frames at 30 fps; "
            f"the HDF5 holds {ft_samples} force/torque samples ({ft_samples / port_frames:.0f}x more)"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read original REASSEMBLE HDF5 recordings with Daft.")
    parser.add_argument("root", help="Directory of extracted .h5 recordings (filled with one if empty)")
    parser.add_argument("--recording", help="Recording to inspect, e.g. 2025-01-11-14-43-37")
    parser.add_argument("--no-frames", action="store_true", help="Skip decoding camera frames")
    parser.add_argument("--compare-port", action="store_true", help="Compare against the Hugging Face LeRobot port")
    args = parser.parse_args()
    main(args.root, args.recording, frames=not args.no_frames, compare_port=args.compare_port)
