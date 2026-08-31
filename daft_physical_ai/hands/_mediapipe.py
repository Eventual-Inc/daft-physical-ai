"""MediaPipe hand tracking as a Daft class-UDF (CPU, 2D keypoints).

Permissive license, no user-supplied weights: the hand_landmarker model is
downloaded once to a local cache on first use.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request

import daft

logger = logging.getLogger(__name__)

from .schema import HANDS_DTYPE

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _default_model_path() -> str:
    cache = os.environ.get("DAFT_PHYSICAL_AI_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "daft_physical_ai"
    )
    return os.path.join(cache, "hand_landmarker.task")


def _ensure_model(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


@daft.cls(max_concurrency=2)
class MediaPipeHands:
    """Detect up to `num_hands` hands per frame; returns the shared hands schema."""

    def __init__(self, model_path: str | None = None, num_hands: int = 2, min_confidence: float = 0.3):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as err:
            raise ImportError(
                "method='mediapipe' requires MediaPipe. Install with `pip install \"daft-physical-ai[mediapipe]\"`."
            ) from err

        self.mp = mp
        path = _ensure_model(model_path or _default_model_path())
        opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
        )
        self.det = vision.HandLandmarker.create_from_options(opts)
        # MediaPipe's __del__ closes the task at interpreter shutdown, after the
        # dispatcher's thread pool and C bindings are gone, spraying an "Exception
        # ignored" traceback (google-ai-edge/mediapipe: shutdown-order bug). There is
        # no earlier hook to close from (Daft UDFs have no teardown; atexit already
        # runs too late), so swallow errors from close instead.
        #
        # Worse, when __del__ runs at GC time before interpreter shutdown (e.g.
        # pytest's teardown forces a collection), the SerialDispatcher's single
        # worker thread can be dead or dying: GC clears the executor's weakref, the
        # worker exits, and close() then submits the C teardown to an executor that
        # will never run it and blocks forever on the result (ThreadPoolExecutor
        # never respawns a dead worker). Skip the close when the worker is already
        # dead, and bound the wait otherwise - the worker can also die between the
        # check and the submit, so aliveness alone can't be trusted. An abandoned
        # close only leaks task resources into a process that is tearing down.
        _close = self.det.close
        _det = self.det

        def _dispatcher_dead() -> bool:
            try:
                threads = _det._lib._executor._threads  # _lib is the SerialDispatcher
                return bool(threads) and not any(t.is_alive() for t in threads)
            except AttributeError:  # private API moved; assume alive and let close try
                return False

        def _run_close():
            try:
                _close()
            except Exception:
                logger.debug("MediaPipe detector close failed (known shutdown-order bug)", exc_info=True)

        def _quiet_close():
            if _dispatcher_dead():
                logger.debug("MediaPipe dispatcher thread already dead; skipping close to avoid a deadlock")
                return
            try:
                closer = threading.Thread(target=_run_close, name="mediapipe-close", daemon=True)
                closer.start()
            except RuntimeError:  # interpreter shutting down; can't spawn threads
                return
            closer.join(timeout=5.0)
            if closer.is_alive():
                logger.debug("MediaPipe close did not finish in 5s; abandoning (dispatcher worker died)")

        self.det.close = _quiet_close

    @daft.method.batch(return_dtype=HANDS_DTYPE, batch_size=16)
    def track(self, images):
        import numpy as np

        out = []
        for arr in images.to_pylist():
            rgb = np.ascontiguousarray(np.asarray(arr), dtype=np.uint8)
            h, w = rgb.shape[:2]
            res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
            landmarks = res.hand_landmarks or []
            handed = res.handedness or []
            frame = []
            for i, lm in enumerate(landmarks):
                kp2d = [[float(p.x * w), float(p.y * h)] for p in lm]
                cat = handed[i][0] if i < len(handed) and handed[i] else None
                frame.append(
                    {
                        # MediaPipe handedness is from the camera's POV and unreliable on
                        # egocentric video; kept for completeness.
                        "handedness": cat.category_name.lower() if cat else "unknown",
                        "confidence": float(cat.score) if cat else 0.0,
                        "kp2d": kp2d,
                        "kp3d": None,  # MediaPipe HandLandmarker is 2D-only here
                    }
                )
            out.append(frame)
        return out


__all__ = ["MediaPipeHands"]
