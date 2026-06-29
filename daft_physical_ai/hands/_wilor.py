"""WiLoR hand tracking as a Daft class-UDF (GPU, 3D MANO keypoints).

WiLoR is a research repo (not pip-installable) and MANO is research-gated, so this
needs two things set up in the environment:

  - the WiLoR repo + pretrained models under ``wilor_root`` (default ``/WiLoR`` or
    ``$DAFT_PHYSICAL_AI_WILOR_ROOT``). Call :func:`ensure_assets` once to fetch them.
  - ``MANO_RIGHT.pkl`` supplied via ``mano_path`` (you must accept the MANO license).

It runs on CUDA. There is no local-GPU path on Apple Silicon, so this is exercised
on Modal (see the project roadmap / TESTING notes).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request

import daft

from .schema import HANDS_DTYPE

_HF = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main"
_WILOR_REPO = "https://github.com/rolpotamias/WiLoR.git"


def _default_wilor_root() -> str:
    return os.environ.get("DAFT_PHYSICAL_AI_WILOR_ROOT", "/WiLoR")


def ensure_assets(wilor_root: str | None = None, mano_path: str | None = None) -> str:
    """Idempotently fetch the WiLoR repo + weights and place MANO. Returns the root.

    Safe to call repeatedly: each step is skipped if its output already exists.
    """
    root = wilor_root or _default_wilor_root()
    pretrained = os.path.join(root, "pretrained_models")
    mano_dir = os.path.join(root, "mano_data")

    if not os.path.isdir(os.path.join(root, "wilor")):
        subprocess.run(["git", "clone", "--depth", "1", _WILOR_REPO, root], check=True)
    os.makedirs(os.path.join(mano_dir, "mano"), exist_ok=True)
    os.makedirs(pretrained, exist_ok=True)

    downloads = {
        f"{pretrained}/detector.pt": f"{_HF}/pretrained_models/detector.pt",
        f"{pretrained}/wilor_final.ckpt": f"{_HF}/pretrained_models/wilor_final.ckpt",
        f"{pretrained}/model_config.yaml": f"{_HF}/pretrained_models/model_config.yaml",
        f"{mano_dir}/mano_mean_params.npz": f"{_HF}/mano_data/mano_mean_params.npz",
    }
    for dst, url in downloads.items():
        if not os.path.exists(dst):
            urllib.request.urlretrieve(url, dst)

    if mano_path:
        for dst in (f"{mano_dir}/MANO_RIGHT.pkl", f"{mano_dir}/mano/MANO_RIGHT.pkl"):
            if not os.path.exists(dst):
                shutil.copy(mano_path, dst)
    return root


@daft.cls(gpus=0.25, max_concurrency=4)
class WiLoRHands:
    """Detect hands per frame with WiLoR; returns the shared hands schema (with kp3d)."""

    def __init__(self, mano_path: str, wilor_root: str | None = None, device: str = "cuda"):
        import sys

        try:
            import torch
        except ImportError as err:
            raise ImportError(
                "method='wilor' requires torch (a CUDA build for GPU). Install the WiLoR "
                "extras with `pip install daft-physical-ai[wilor]`."
            ) from err

        root = ensure_assets(wilor_root, mano_path)
        os.chdir(root)
        sys.path.insert(0, root)

        # WiLoR imports pyrender (for mesh overlays we don't use); stub it so the
        # headless import doesn't require an EGL/OSMesa GL stack.
        import types

        class _AnyMeta(type):
            def __getattr__(cls, n):
                return _Any

            def __call__(cls, *a, **k):
                return super().__call__()

        class _Any(metaclass=_AnyMeta):
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, n):
                return _Any

            def __call__(self, *a, **k):
                return self

        st = types.ModuleType("pyrender")
        st.__file__ = "/tmp/p.py"
        st.__getattr__ = lambda n: (_ for _ in ()).throw(AttributeError(n)) if n.startswith("__") else _Any  # type: ignore[method-assign]
        sys.modules["pyrender"] = st

        try:
            from ultralytics import YOLO
            from wilor.datasets.vitdet_dataset import ViTDetDataset
            from wilor.models import load_wilor
            from wilor.utils import recursive_to
        except ImportError as err:
            raise ImportError(
                f"Could not import the WiLoR stack ({err}). It needs the WiLoR repo on the path "
                f"(fetched by ensure_assets into {root!r}) plus the [wilor] extras AND chumpy from git "
                "(`pip install 'chumpy @ git+https://github.com/mattloper/chumpy'`), which is omitted "
                "from the extra because PyPI metadata can't carry direct references."
            ) from err

        self.torch = torch
        self.recursive_to = recursive_to
        self.ViTDetDataset = ViTDetDataset
        self.device = torch.device(device)
        self.model, self.cfg = load_wilor(
            checkpoint_path="./pretrained_models/wilor_final.ckpt",
            cfg_path="./pretrained_models/model_config.yaml",
        )
        self.model = self.model.to(self.device).eval()
        self.detector = YOLO("./pretrained_models/detector.pt")

    def _cam_full(self, cam_bbox, bc, bs, isz, focal):
        torch = self.torch
        img_w, img_h = isz[:, 0], isz[:, 1]
        cx, cy = bc[:, 0], bc[:, 1]
        s = bs * cam_bbox[:, 0] + 1e-9
        return torch.stack(
            [2 * (cx - img_w / 2) / s + cam_bbox[:, 1], 2 * (cy - img_h / 2) / s + cam_bbox[:, 2], 2 * focal / s],
            dim=-1,
        )

    def _project(self, pts, cam_t, focal, isz):
        torch = self.torch
        K = torch.eye(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = isz[0] / 2.0
        K[1, 2] = isz[1] / 2.0
        p = torch.from_numpy(pts).float() + torch.from_numpy(cam_t).float()
        pr = (K @ p.T).T
        return (pr[:, :2] / pr[:, 2:3]).numpy()

    @daft.method.batch(return_dtype=HANDS_DTYPE, batch_size=8)
    def track(self, images):
        import numpy as np

        torch = self.torch
        out = []
        for arr in images.to_pylist():
            rgb = np.asarray(arr)
            img = rgb[:, :, ::-1].copy()  # RGB (from Daft) -> BGR (WiLoR)
            det = self.detector(img, conf=0.3, verbose=False)[0]
            if len(det.boxes) == 0:
                out.append([])
                continue
            confs = det.boxes.conf.cpu().numpy()
            boxes = det.boxes.xyxy.cpu().numpy()[:, :4]
            isrs = det.boxes.cls.cpu().numpy().astype(np.float32)  # 0=left, 1=right
            ds = self.ViTDetDataset(self.cfg, img, boxes, isrs, rescale_factor=2.0)
            batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(boxes), shuffle=False)))
            batch = self.recursive_to(batch, self.device)
            with torch.no_grad():
                o = self.model(batch)
            mult = 2 * batch["right"] - 1
            pc = o["pred_cam"]
            pc[:, 1] = mult * pc[:, 1]
            isz = batch["img_size"].float()
            focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * isz.max()
            ct = (
                self._cam_full(pc, batch["box_center"].float(), batch["box_size"].float(), isz, focal)
                .detach()
                .cpu()
                .numpy()
            )
            hands = []
            for h in range(len(boxes)):
                isr = float(isrs[h])
                j3 = o["pred_keypoints_3d"][h].detach().cpu().numpy()
                j3[:, 0] = (2 * isr - 1) * j3[:, 0]
                kp2 = self._project(j3, ct[h], focal.item(), isz[h].cpu().numpy())
                hands.append(
                    {
                        "handedness": "right" if isr == 1.0 else "left",
                        "confidence": float(confs[h]),
                        "kp2d": kp2.astype(float).tolist(),  # [[x, y], ...]
                        "kp3d": j3.astype(float).tolist(),  # [[x, y, z], ...]
                    }
                )
            out.append(hands)
        return out


__all__ = ["WiLoRHands", "ensure_assets"]
