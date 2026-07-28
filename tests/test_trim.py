from __future__ import annotations

from typing import Any

import daft
import pytest

from daft_physical_ai.trim import TRIM_FIELDS, trim_windows

FPS = 10.0


def _frames(active: dict[int, list[bool]], from_ts: dict[int, float] | None = None) -> daft.DataFrame:
    """Build a frame-level DataFrame from {episode_index: [is_active per frame]}."""
    rows: dict[str, Any] = {"episode_index": [], "frame_index": [], "is_active": []}
    if from_ts is not None:
        rows["from_ts"] = []
    for episode, flags in active.items():
        for i, flag in enumerate(flags):
            rows["episode_index"].append(episode)
            rows["frame_index"].append(i)
            rows["is_active"].append(flag)
            if from_ts is not None:
                rows["from_ts"].append(from_ts[episode])
    return daft.from_pydict(rows)


def _one(df: daft.DataFrame, episode: int = 0) -> dict:
    rows = df.where(df["episode_index"] == episode).to_pylist()
    assert len(rows) == 1
    return rows[0]


def test_window_spans_first_to_last_active_frame() -> None:
    # active on frames 3..6 of a 10-frame episode, no padding
    df = _frames({0: [False] * 3 + [True] * 4 + [False] * 3})
    row = _one(trim_windows(df, fps=FPS, pad_s=0.0))
    assert (row["start_frame"], row["end_frame"]) == (3, 6)
    assert row["kept_frames"] == 4
    assert row["trim_fraction"] == pytest.approx(0.6)
    assert row["never_active"] is False


def test_padding_extends_the_window_on_both_sides() -> None:
    df = _frames({0: [False] * 3 + [True] * 4 + [False] * 3})
    row = _one(trim_windows(df, fps=FPS, pad_s=0.2))  # 0.2s * 10fps = 2 frames
    assert (row["start_frame"], row["end_frame"]) == (1, 8)


def test_padding_is_clamped_to_the_episode() -> None:
    df = _frames({0: [True] * 4})
    row = _one(trim_windows(df, fps=FPS, pad_s=1.0))  # 10 frames of padding
    assert (row["start_frame"], row["end_frame"]) == (0, 3)
    assert row["trim_fraction"] == pytest.approx(0.0)


def test_interior_gaps_stay_inside_the_window() -> None:
    # The window is contiguous by construction: the pause at 3..5 is kept.
    df = _frames({0: [False, True, True, False, False, False, True, True, False]})
    row = _one(trim_windows(df, fps=FPS, pad_s=0.0))
    assert (row["start_frame"], row["end_frame"]) == (1, 7)
    assert row["kept_frames"] == 7


def test_never_active_episode_keeps_its_full_span_and_is_flagged() -> None:
    df = _frames({0: [False] * 5})
    row = _one(trim_windows(df, fps=FPS, pad_s=0.0))
    assert row["never_active"] is True
    assert (row["start_frame"], row["end_frame"]) == (0, 4)
    assert row["trim_fraction"] == pytest.approx(0.0)


def test_timestamps_are_episode_relative_without_from_ts() -> None:
    df = _frames({0: [False] * 2 + [True] * 3})
    row = _one(trim_windows(df, fps=FPS, pad_s=0.0))
    assert row["start_ts"] == pytest.approx(0.2)
    assert row["end_ts"] == pytest.approx(0.4)


def test_from_ts_makes_timestamps_absolute_in_the_video_file() -> None:
    df = _frames({0: [False] * 2 + [True] * 3}, from_ts={0: 100.0})
    row = _one(trim_windows(df, fps=FPS, pad_s=0.0, from_ts="from_ts"))
    assert row["start_ts"] == pytest.approx(100.2)
    assert row["end_ts"] == pytest.approx(100.4)


def test_one_row_per_episode_with_independent_windows() -> None:
    df = _frames(
        {
            0: [False, True, True, False],
            1: [True, True, False, False, False, False],
        }
    )
    out = trim_windows(df, fps=FPS, pad_s=0.0)
    assert out.count_rows() == 2
    assert (_one(out, 0)["start_frame"], _one(out, 0)["end_frame"]) == (1, 2)
    assert (_one(out, 1)["start_frame"], _one(out, 1)["end_frame"]) == (0, 1)


def test_output_carries_exactly_the_schema_fields() -> None:
    df = _frames({0: [True, True]})
    out = trim_windows(df, fps=FPS)
    assert out.column_names == ["episode_index", *TRIM_FIELDS]


def test_rejects_a_nonpositive_fps() -> None:
    df = _frames({0: [True]})
    with pytest.raises(ValueError, match="fps must be > 0"):
        trim_windows(df, fps=0)


def test_rejects_negative_padding() -> None:
    df = _frames({0: [True]})
    with pytest.raises(ValueError, match="pad_s must be >= 0"):
        trim_windows(df, fps=FPS, pad_s=-1.0)
