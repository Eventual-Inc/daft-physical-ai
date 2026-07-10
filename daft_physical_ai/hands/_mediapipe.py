"""MediaPipe hand tracking as a Daft class-UDF (CPU, 2D keypoints).

Permissive license, no user-supplied weights: the hand_landmarker model is
downloaded once to a local cache on first use.
"""

from __future__ import annotations

import os
import urllib.request
from contextlib import suppress

import daft

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
        _close = self.det.close

        def _quiet_close():
            with suppress(Exception):
                _close()

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
