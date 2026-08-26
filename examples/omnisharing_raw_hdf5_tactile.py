"""Read a PX OmniSharing DF-2 release with Daft, tactile and cameras included.

The published data is CC-BY-NC-SA 4.0, so nothing is downloaded for you. Point
this at the Hugging Face repo (streamed over `hf://`) or at your own copy:

    uv run python examples/omnisharing_raw_hdf5_tactile.py paxini/Omnisharing_DB_SampleData
    uv run python examples/omnisharing_raw_hdf5_tactile.py /data/omnisharing --episodes 4

Episodes are 0.4-3.2 GB each, so the episode catalog is built from filenames
alone and only the modalities selected below are streamed - one hand's tactile
pads, its joint angles, and a single decoded frame per camera.
"""

from __future__ import annotations

import daft

from daft_physical_ai.datasets import omnisharing

CAMERAS = ("RGB_Camera0", ("RGBD_0", "color"))


def main(
    dataset_uri: str,
    *,
    limit: int = 1,
    side: str = "lefthand",
    stage: str | None = "DF-2",
) -> daft.DataFrame:
    episodes = omnisharing.raw(dataset_uri, stage=stage).limit(limit)
    labelled = omnisharing.episode_metadata(episodes)
    tactile = omnisharing.tactile(labelled, sides=side, split_by_sensor=True)
    poses = omnisharing.trajectory(tactile, fields=[f"observation.{side}.joints"], include_attrs=False)
    frames = omnisharing.camera_frames(poses, list(CAMERAS), max_frames=1, width=320, height=200)
    return frames.select(
        "episode_key",
        "instruction",
        "n_frames",
        f"observation.{side}.joints",
        f"{side}.tactile.sensor_names",
        f"{side}.tactile.palm_sensor1",
        "RGB_Camera0.frames",
        "RGBD_0.color.frames",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read PX OmniSharing DF-2 HDF5 episodes with Daft.")
    parser.add_argument("dataset_uri", help="HF repo id (org/name), or a local / remote directory")
    parser.add_argument("--episodes", type=int, default=1, help="Maximum episodes to read")
    parser.add_argument("--side", default="lefthand", choices=list(omnisharing.SIDES), help="Hand to read")
    parser.add_argument("--stage", default="DF-2", help="Pipeline stage to keep; pass 'all' for every stage")
    parser.add_argument("--layout", action="store_true", help="Print the HDF5 layout instead of reading modalities")
    args = parser.parse_args()

    stage = None if args.stage == "all" else args.stage
    if args.layout:
        episodes = omnisharing.raw(args.dataset_uri, stage=stage).limit(args.episodes)
        omnisharing.describe(episodes).select("h5path", "kind", "shape", "dtype").show(60)
    else:
        main(args.dataset_uri, limit=args.episodes, side=args.side, stage=stage).show()
