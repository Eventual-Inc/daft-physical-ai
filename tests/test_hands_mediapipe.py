from __future__ import annotations

import daft
import numpy as np
import pytest

from daft_physical_ai.hands import HANDS_DTYPE, track_hands


def test_unknown_method_raises() -> None:
    df = daft.from_pydict({"x": [1]})
    with pytest.raises(ValueError):
        track_hands(df["x"], method="nope")


def test_close_skips_dead_dispatcher() -> None:
    """A close() with no dispatcher worker to run it would deadlock; it must be skipped.

    When GC finalizes the detector (e.g. pytest's forced collection at session end),
    MediaPipe's SerialDispatcher worker thread can already be dead, and close() then
    blocks forever on a job nothing will run. Simulate that state and assert the C
    close is never dispatched.
    """
    import threading

    pytest.importorskip("mediapipe")
    from daft_physical_ai.hands._mediapipe import MediaPipeHands

    # @daft.cls wraps the class; the undecorated original is its generic base arg.
    inst = MediaPipeHands.__orig_bases__[0].__args__[0]()  # ty: ignore[unresolved-attribute]

    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    executor = inst.det._lib._executor
    executor._threads.clear()
    executor._threads.add(dead)

    dispatched = []
    inst.det._lib.MpHandLandmarkerClose = lambda *args: dispatched.append(args)

    inst.det.close()  # must return instead of blocking on the dead worker

    assert dispatched == []


def test_close_abandons_a_blocked_close() -> None:
    """A blocked close must not hang the process.

    The worker can die between the aliveness check and the submit; close runs on
    a helper thread and is abandoned on timeout.
    """
    import threading
    import time

    pytest.importorskip("mediapipe")
    from daft_physical_ai.hands._mediapipe import MediaPipeHands

    inst = MediaPipeHands.__orig_bases__[0].__args__[0]()  # ty: ignore[unresolved-attribute]

    release = threading.Event()
    inst.det._lib.MpHandLandmarkerClose = lambda *args: release.wait()

    start = time.monotonic()
    inst.det.close()  # underlying close blocks until released; must return anyway
    elapsed = time.monotonic() - start
    release.set()

    assert elapsed < 30  # returned via the timeout instead of hanging forever


def test_mediapipe_runs_and_matches_schema() -> None:
    pytest.importorskip("mediapipe")
    img = np.zeros((64, 64, 3), dtype=np.uint8)  # blank frame: pipeline runs, finds 0 hands
    df = daft.from_pydict({"image": [img]})
    df = df.with_column("hands", track_hands(df["image"], method="mediapipe"))

    field = next(f for f in df.schema() if f.name == "hands")
    assert str(field.dtype) == str(HANDS_DTYPE)

    result = df.select("hands").to_pydict()
    assert isinstance(result["hands"][0], list)  # 0 or more detected hands
