from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from daft import DataType
from daft.exceptions import DaftCoreException

from daft_physical_ai.datasets import omnisharing
from tests.omnisharing_datagen import SENSOR_LENGTHS, SENSOR_NAMES, episode_filename, write_df2_episode


@pytest.fixture
def heterogeneous_root(tmp_path: Path) -> Path:
    write_df2_episode(
        tmp_path / "DF-2" / episode_filename(1, "212953", 93, 110056),
        n_frames=4,
        n_objects=2,
    )
    write_df2_episode(
        tmp_path / "DF-2R" / episode_filename(2, "213000", 93, 110056, "mano"),
        n_frames=5,
        n_objects=4,
        n_joints=17,
        tactile_width=3750,
    )
    return tmp_path


def test_tactile_flat_mode_has_tensor_and_fixed_layout(heterogeneous_root: Path) -> None:
    result = omnisharing.tactile(
        omnisharing.raw(str(heterogeneous_root), stage="DF-2"),
        "lefthand",
    )
    assert result.column_names[-2:] == [
        "observation/lefthand/tactile",
        "observation/lefthand/tactile_layout",
    ]
    row = result.to_pylist()[0]
    values = np.asarray(row["observation/lefthand/tactile"])
    layout = row["observation/lefthand/tactile_layout"]
    assert values.shape == (4, 3465)
    assert [item["name"] for item in layout] == SENSOR_NAMES
    assert [item["width"] for item in layout] == SENSOR_LENGTHS
    assert [item["offset"] for item in layout] == list(np.cumsum([0, *SENSOR_LENGTHS[:-1]]))


def test_tactile_split_mode_is_stable_for_heterogeneous_widths(heterogeneous_root: Path) -> None:
    episodes = omnisharing.raw(str(heterogeneous_root))
    result = omnisharing.tactile(episodes, ["lefthand", "lefthand"], split_by_sensor=True)
    assert result.column_names[-1] == "observation/lefthand/tactile_sensors"
    rows = sorted(result.to_pylist(), key=lambda row: row["stage"])
    assert [sum(item["width"] for item in row["observation/lefthand/tactile_sensors"]) for row in rows] == [
        3465,
        3750,
    ]
    for row in rows:
        for sensor in row["observation/lefthand/tactile_sensors"]:
            assert np.asarray(sensor["values"]).shape[1] == sensor["width"]


@pytest.mark.parametrize("sides", [[], ["middlehand"]])
def test_tactile_validates_selector(heterogeneous_root: Path, sides: list[str]) -> None:
    with pytest.raises(ValueError):
        omnisharing.tactile(omnisharing.raw(str(heterogeneous_root)), sides)


@pytest.mark.parametrize("problem", ["duplicate", "width"])
def test_tactile_rejects_invalid_sensor_layout(tmp_path: Path, problem: str) -> None:
    path = write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3)
    import h5py

    with h5py.File(path, "a") as h5:
        node = h5["dataset/observation/lefthand/tactile/data"]
        if problem == "duplicate":
            node.attrs["sensor_names"] = np.array([SENSOR_NAMES[0], *SENSOR_NAMES[:-1]], dtype=object)
        else:
            node.attrs["sensor_lengths"] = np.array([*SENSOR_LENGTHS[:-1], SENSOR_LENGTHS[-1] - 1])
    with pytest.raises(DaftCoreException, match="observation/lefthand/tactile/data"):
        omnisharing.tactile(omnisharing.raw(str(tmp_path)), "lefthand").collect()


def test_audio_is_namespaced_and_bounded_at_read_time(heterogeneous_root: Path) -> None:
    result = omnisharing.audio(
        omnisharing.raw(str(heterogeneous_root), stage="DF-2"),
        mono=True,
        max_seconds=0.01,
    )
    assert result.column_names[-3:] == ["audio/waveform", "audio/sample_rate", "audio/sample_count"]
    assert result.schema()["audio/waveform"].dtype == DataType.tensor(DataType.float64())
    row = result.to_pylist()[0]
    assert row["audio/sample_rate"] == 8000
    assert row["audio/sample_count"] == 80
    assert np.asarray(row["audio/waveform"]).shape == (80,)


def test_audio_missing_is_typed_empty(tmp_path: Path) -> None:
    write_df2_episode(
        tmp_path / episode_filename(1, "212953", 93, 110056),
        n_frames=3,
        include_audio=False,
    )
    row = omnisharing.audio(omnisharing.raw(str(tmp_path))).to_pylist()[0]
    assert np.asarray(row["audio/waveform"]).shape == (0,)
    assert row["audio/sample_rate"] is None
    assert row["audio/sample_count"] is None


def test_audio_validates_limit(heterogeneous_root: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        omnisharing.audio(omnisharing.raw(str(heterogeneous_root)), max_seconds=0)


def test_objects_use_stable_nested_schema_for_variable_counts(heterogeneous_root: Path) -> None:
    result = omnisharing.objects(omnisharing.raw(str(heterogeneous_root))).sort("stage")
    assert result.column_names[-2:] == ["n_objects", "objects"]
    rows = result.to_pylist()
    assert [row["n_objects"] for row in rows] == [2, 4]
    assert [[item["index"] for item in row["objects"]] for row in rows] == [[1, 2], [1, 2, 3, 4]]
    assert rows[0]["objects"][0]["name"] == "object_1"
    assert rows[0]["objects"][0]["id"] == 101
    assert np.asarray(rows[0]["objects"][0]["pose"]).shape == (4, omnisharing.OBJECT_POSE_WIDTH)


def test_objects_missing_are_empty_not_dynamic_columns(tmp_path: Path) -> None:
    write_df2_episode(
        tmp_path / episode_filename(1, "212953", 93, 110056),
        n_frames=3,
        n_objects=0,
    )
    row = omnisharing.objects(omnisharing.raw(str(tmp_path))).to_pylist()[0]
    assert row["n_objects"] == 0
    assert row["objects"] == []


def _replace_audio(path: Path, waveform: np.ndarray, sample_rate: int | None = 8000) -> None:
    import h5py

    with h5py.File(path, "a") as h5:
        del h5["dataset/observation/audio"]
        node = h5["dataset/observation"].create_dataset("audio", data=waveform)
        if sample_rate is not None:
            node.attrs["samplerate"] = np.int64(sample_rate)


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    return write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=4)


def _read_audio(path: Path, **kwargs) -> dict:
    return omnisharing.audio(omnisharing.raw(str(path.parent)), **kwargs).to_pylist()[0]


def test_audio_full_read_keeps_the_channel_axis(audio_path: Path) -> None:
    row = _read_audio(audio_path)
    waveform = np.asarray(row["audio/waveform"])
    assert waveform.shape == (160, 1)
    assert waveform.dtype == np.float64
    assert row["audio/sample_count"] == 160


def test_audio_mono_downmix_is_exact(audio_path: Path) -> None:
    _replace_audio(audio_path, np.stack([np.ones(50), np.full(50, 3.0)], axis=1))
    waveform = np.asarray(_read_audio(audio_path, mono=True)["audio/waveform"])
    assert waveform.shape == (50,)
    np.testing.assert_allclose(waveform, 2.0)


def test_audio_mono_handles_waveform_without_channel_axis(audio_path: Path) -> None:
    flat = np.linspace(-1, 1, 100, dtype="float64")
    _replace_audio(audio_path, flat)
    row = _read_audio(audio_path, mono=True)
    np.testing.assert_array_equal(row["audio/waveform"], flat)
    assert row["audio/sample_count"] == 100


def test_audio_max_seconds_truncates_a_prefix_without_rescaling(audio_path: Path) -> None:
    full = _read_audio(audio_path)
    clipped = _read_audio(audio_path, max_seconds=0.005)
    assert clipped["audio/sample_count"] == 40
    assert clipped["audio/sample_rate"] == full["audio/sample_rate"] == 8000
    np.testing.assert_array_equal(clipped["audio/waveform"], np.asarray(full["audio/waveform"])[:40])


def test_audio_max_seconds_beyond_duration_returns_everything(audio_path: Path) -> None:
    assert _read_audio(audio_path, max_seconds=1e6)["audio/sample_count"] == 160


def test_audio_max_seconds_clamps_to_at_least_one_sample(audio_path: Path) -> None:
    assert _read_audio(audio_path, max_seconds=1e-9)["audio/sample_count"] == 1


def test_audio_zero_length_dataset_reports_zero_samples(audio_path: Path) -> None:
    _replace_audio(audio_path, np.empty((0, 1), dtype="float64"))
    row = _read_audio(audio_path)
    assert row["audio/sample_count"] == 0
    assert row["audio/sample_rate"] == 8000


def test_audio_without_sample_rate_ignores_max_seconds(audio_path: Path) -> None:
    _replace_audio(audio_path, np.zeros((160, 1)), sample_rate=None)
    row = _read_audio(audio_path, max_seconds=0.001)
    assert row["audio/sample_rate"] is None
    assert row["audio/sample_count"] == 160


def test_audio_mixed_release_reads_present_and_missing(tmp_path: Path) -> None:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=4)
    write_df2_episode(tmp_path / episode_filename(2, "213630", 115, 110092), n_frames=4, include_audio=False)
    rows = omnisharing.audio(omnisharing.raw(str(tmp_path))).sort("episode_index").to_pylist()
    assert [np.asarray(row["audio/waveform"]).size for row in rows] == [160, 0]
    assert [row["audio/sample_rate"] for row in rows] == [8000, None]


def test_objects_pad_variable_counts_without_dynamic_columns(tmp_path: Path) -> None:
    for index, count in ((1, 2), (2, 0), (3, 4)):
        write_df2_episode(tmp_path / episode_filename(index, "212953", 93, 110056), n_frames=4, n_objects=count)
    result = omnisharing.objects(omnisharing.raw(str(tmp_path))).sort("episode_index")
    assert result.column_names[-2:] == ["n_objects", "objects"]
    assert not any(re.fullmatch(r"obj\d+.*", name) for name in result.column_names)
    two, none, four = result.to_pylist()
    assert [two["n_objects"], none["n_objects"], four["n_objects"]] == [2, 0, 4]
    assert none["objects"] == []
    assert [item["id"] for item in four["objects"]] == [101, 102, 103, 104]
    assert all(np.asarray(item["pose"]).shape == (4, omnisharing.OBJECT_POSE_WIDTH) for item in four["objects"])


def test_objects_read_cleanly_when_no_episode_has_objects(tmp_path: Path) -> None:
    # Every row empty is the case where an all-null nested column would break Daft.
    for index in (1, 2):
        write_df2_episode(tmp_path / episode_filename(index, "212953", 93, 110056), n_frames=3, n_objects=0)
    rows = omnisharing.objects(omnisharing.raw(str(tmp_path))).to_pylist()
    assert [(row["n_objects"], row["objects"]) for row in rows] == [(0, []), (0, [])]


def test_objects_order_by_numeric_group_index(tmp_path: Path) -> None:
    write_df2_episode(tmp_path / episode_filename(1, "212953", 93, 110056), n_frames=3, n_objects=11)
    row = omnisharing.objects(omnisharing.raw(str(tmp_path))).to_pylist()[0]
    assert [item["index"] for item in row["objects"]] == list(range(1, 12))


def test_modalities_construct_lazy_schemas_without_reads(
    heterogeneous_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daft.file.hdf5 import Hdf5File

    def forbidden(*args, **kwargs):
        raise AssertionError("planning opened HDF5")

    monkeypatch.setattr(Hdf5File, "open", forbidden)
    episodes = omnisharing.raw(str(heterogeneous_root))
    assert (
        "observation/lefthand/tactile_sensors"
        in omnisharing.tactile(episodes, "lefthand", split_by_sensor=True).column_names
    )
    assert "audio/waveform" in omnisharing.audio(episodes).column_names
    assert omnisharing.objects(episodes).column_names[-2:] == ["n_objects", "objects"]
