"""Robometer eval server on Modal - the cloud wrapper around the launcher.

All the substance (pins, install, model download, dtype patches, launch)
lives in run_robometer_server.py, which also runs standalone on any local
NVIDIA GPU. This file only adds the Modal shape: bake the install + model
into the image at build time (--install-only), patch + launch at container
start.

Two serving shapes, same image:

- `web` function: the blog-facing "bring a URL" shape. @modal.web_server
  exposes the eval server at a public HTTPS URL behind Modal proxy auth
  (create a token under Settings > Proxy Auth Tokens, send it as
  Modal-Key / Modal-Secret headers).
- `Robometer` class: no public endpoint. The server listens on localhost
  inside the container and a Modal class method relays requests to it, so
  calls go through the Modal SDK and auth rides on ~/.modal.toml.

Deploy:  modal deploy modal_eval_server.py
"""

from pathlib import Path

import modal

LAUNCHER = Path(__file__).parent / "run_robometer_server.py"
ROOT = "/opt/robometer"
MODEL_DIR = "/model"
PORT = 8001

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(LAUNCHER, "/root/run_robometer_server.py", copy=True)
    .run_commands(
        f"python /root/run_robometer_server.py --install-only --root {ROOT} --model-dir {MODEL_DIR}"
    )
)

app = modal.App("robometer-eval-server")


def start_server(host: str):
    """Patch + launch via the baked-in launcher; returns the process."""
    import run_robometer_server as launcher

    launcher.apply_patches(Path(ROOT))
    return launcher.launch(Path(ROOT), Path(MODEL_DIR), host, PORT)


@app.function(image=image, gpu="A10G", timeout=3600, scaledown_window=600)
@modal.web_server(port=PORT, startup_timeout=900, requires_proxy_auth=True)
def web():
    # Bind 0.0.0.0 so Modal's proxy can reach the port; the decorator waits
    # for it to accept connections (up to startup_timeout, covers model load).
    start_server("0.0.0.0")


@app.cls(image=image, gpu="A10G", timeout=3600, scaledown_window=600)
class Robometer:
    @modal.enter()
    def start(self):
        import time

        import requests

        self.proc = start_server("127.0.0.1")
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                    return
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"eval server exited with {self.proc.returncode}")
            time.sleep(3)
        raise TimeoutError("eval server did not become healthy in 15 min")

    @modal.method()
    def score(self, frames_npy: bytes, sample_json: str) -> dict:
        """Relay one /evaluate_batch_npy request to the local eval server.

        Same wire format as the HTTP path so client code stays identical.
        """
        import requests

        resp = requests.post(
            f"http://127.0.0.1:{PORT}/evaluate_batch_npy",
            files={
                "sample_0_trajectory_frames": (
                    "sample_0_trajectory_frames.npy",
                    frames_npy,
                    "application/octet-stream",
                )
            },
            data={"sample_0": sample_json, "use_frame_steps": "false"},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()
