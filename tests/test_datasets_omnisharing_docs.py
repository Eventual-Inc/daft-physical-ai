from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from daft import col, lit

from daft_physical_ai.datasets import omnisharing
from tests.omnisharing_datagen import episode_filename, write_df2_episode


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "omnisharing_raw_hdf5_tactile.py"
    spec = importlib.util.spec_from_file_location("omnisharing_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_documented_quickstart_and_example_are_runnable(tmp_path: Path) -> None:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    example = _load_example()
    result = example.build(str(tmp_path))
    assert result.column_names == [
        "episode_key",
        "capture_key",
        "instruction",
        "frame_count",
        "observation/lefthand/joints",
        "observation/lefthand/tactile_sensors",
    ]
    assert result.count_rows() == 1


def test_documented_camera_depth_and_alignment_calls_plan(tmp_path: Path) -> None:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    one = omnisharing.raw(str(tmp_path)).limit(1)
    assert omnisharing.cameras(one).where(col("camera") == lit("RGB_Camera0")).count_rows() == 1
    assert "RGB_Camera0/payload" in omnisharing.camera_payloads(one, "RGB_Camera0").column_names
    assert "RGB_Camera99/frames" in omnisharing.camera_frames(one, "RGB_Camera99").column_names
    assert "RGBD_0/depth" in omnisharing.depth_frames(one, "RGBD_0", frame_indices=[0, 1]).column_names
    assert "RGBD_0/left_to_color" in omnisharing.stereo_extrinsics(one, "RGBD_0").column_names
    aligned = omnisharing.frames(
        one,
        "observation/lefthand/joints",
        align_cameras=["RGB_Camera0", ("RGBD_0", "left")],
        include_columns=["episode_key", "capture_key"],
    )
    assert aligned.count_rows() == 3


GUIDE = Path(__file__).parents[1] / "docs" / "omnisharing.md"
PUBLIC_DATASET = "paxini/Omnisharing_DB_SampleData"


def _guide_python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", GUIDE.read_text(), flags=re.DOTALL)


def test_every_guide_snippet_runs_against_a_generated_release(tmp_path: Path) -> None:
    part = tmp_path / "data" / "part_01"
    write_df2_episode(part / episode_filename(1203, "213135", 115, 110092), n_frames=12)
    # The duplicated episode_index the guide warns about.
    write_df2_episode(part / episode_filename(1217, "212953", 93, 110056), n_frames=12)
    write_df2_episode(part / episode_filename(1217, "213630", 115, 110092), n_frames=12)

    blocks = _guide_python_blocks()
    assert len(blocks) >= 8
    namespace: dict = {}
    for block in blocks:
        exec(compile(block.replace(PUBLIC_DATASET, str(tmp_path)), str(GUIDE), "exec"), namespace)  # noqa: S102

    # Snippets that only build plans are executed too, except decoding, which
    # needs encoders that the generated payloads do not carry.
    for name in ("flat", "split", "clip", "tracked", "inventory", "payloads", "depth", "stereo", "per_frame"):
        assert namespace[name].count_rows() > 0, name
    assert "RGB_Camera0/frames" in namespace["decoded"].column_names
    assert namespace["quat_xyzw"].shape == (12, 4)
    keys = namespace["episodes"].select("episode_key", "episode_index").to_pylist()
    assert len({row["episode_key"] for row in keys}) == 3
    assert len({row["episode_index"] for row in keys}) == 2


def test_guide_states_the_measured_sizes() -> None:
    text = GUIDE.read_text()
    assert "292 MB to 4.12 GB" in text
    assert "1.4-3.5 MB per stream" in text
    assert "0.4" not in text
