from __future__ import annotations

import pytest

from daft_physical_ai.failure_modes import detect_regrasp


def test_detect_regrasp_loop() -> None:
    detection = detect_regrasp(
        object_z=[0.0, 0.0, 0.08, 0.11, 0.02, 0.02, 0.09],
        gripper_closed=[False, True, True, True, True, False, True],
    )

    assert detection.label == "re_grasp"
    assert detection.regrasp_count == 1
    assert [event.kind for event in detection.events] == ["grasp", "lift", "drop", "re-grasp", "lift"]


def test_detect_drop_without_recovery() -> None:
    detection = detect_regrasp(
        object_z=[0.0, 0.0, 0.08, 0.02, 0.01],
        gripper_closed=[False, True, True, True, False],
    )

    assert detection.label == "drop_no_recover"
    assert detection.regrasp_count == 0


def test_detect_no_grasp() -> None:
    detection = detect_regrasp(
        object_z=[0.0, 0.01, 0.0],
        gripper_closed=[False, False, False],
    )

    assert detection.label == "no_grasp"


def test_detect_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        detect_regrasp(object_z=[0.0], gripper_closed=[])
