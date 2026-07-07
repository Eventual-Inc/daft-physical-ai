from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from daft_physical_ai.episodes import (
    ACTION_DIM,
    COLUMNS,
    ROLLOUT_SCHEMA,
    SCHEMA_VERSION,
    Episode,
    Step,
    assert_emits_schema,
    empty_step_row,
    validate_rows,
    write_rows,
)


def _toy_episode() -> Episode:
    steps = tuple(
        Step(
            timestep=timestep,
            state=np.zeros(8, dtype=np.float32),
            action=np.full(ACTION_DIM, 0.1 * timestep, dtype=np.float32),
            reward=float(timestep == 2),
            done=timestep == 2,
            is_terminal=timestep == 2,
            eef_pos=np.array([0.1, 0.2, 0.3 + 0.01 * timestep], dtype=np.float32),
            gripper_state=0.04 - 0.01 * timestep,
            object_poses={"akita_black_bowl": [0.0, 0.1, 0.2, 0, 0, 0, 1]},
        )
        for timestep in range(3)
    )
    return Episode(
        episode_id="libero_goal/0/7/0",
        source="rollout",
        instruction="put the bowl on the plate",
        steps=steps,
        success=False,
        terminal_failure="re_grasp",
        model="openvla/openvla-7b-finetuned-libero-goal",
        policy_type="openvla",
        suite="libero_goal",
        task_id=0,
        task_name="put_the_bowl_on_the_plate",
        metadata={"control_mode": "relative"},
    )


def test_empty_row_has_all_columns() -> None:
    row = empty_step_row()
    assert set(row) == set(COLUMNS)
    assert all(value is None for value in row.values())


def test_to_step_rows_matches_schema_columns() -> None:
    rows = _toy_episode().to_step_rows(run_id="test")
    assert len(rows) == 3
    for row in rows:
        assert set(row) == set(COLUMNS)
        assert row["schema_version"] == SCHEMA_VERSION

    table = validate_rows(rows)
    assert table.schema.equals(ROLLOUT_SCHEMA, check_metadata=False)


def test_roundtrip_through_parquet(tmp_path) -> None:
    rows = _toy_episode().to_step_rows(run_id="test")
    out = write_rows(rows, tmp_path / "episode.parquet")

    assert out.exists()
    assert_emits_schema(out)

    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("episode_id")[0].as_py() == "libero_goal/0/7/0"
    assert table.column("terminal_failure")[0].as_py() == "re_grasp"
    assert table.column("success")[0].as_py() is False
    np.testing.assert_allclose(
        table.column("action")[1].as_py(),
        [0.1] * ACTION_DIM,
        rtol=1e-6,
    )
    np.testing.assert_allclose(table.column("gripper_action")[1].as_py(), 0.1, rtol=1e-6)
    assert table.column("step_idx").to_pylist() == [0, 1, 2]


def test_failure_filter_query(tmp_path) -> None:
    out = write_rows(_toy_episode().to_step_rows(run_id="test"), tmp_path / "episode.parquet")
    table = pq.read_table(out)
    mask = pa.array([not value for value in table.column("success").to_pylist()])
    failures = table.filter(mask)
    assert failures.num_rows == 3
    assert set(failures.column("terminal_failure").to_pylist()) == {"re_grasp"}
