from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "daft-physical-ai-marimo"
REMOTE_ROOT = Path("/app")


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "daft>=0.7.16",
        "fastapi[standard]>=0.115",
        "markdown>=3.6",
        "marimo>=0.14",
    )
    .add_local_file(
        Path(__file__).parent / "web_app.py",
        remote_path=str(REMOTE_ROOT / "web_app.py"),
    )
    .add_local_dir(
        Path(__file__).parent / "notebooks",
        remote_path=str(REMOTE_ROOT / "notebooks"),
    )
    .add_local_dir(
        Path(__file__).parent / "content",
        remote_path=str(REMOTE_ROOT / "content"),
    )
    .add_local_dir(
        Path(__file__).parent.parent / "egodex_handtracking_lite",
        remote_path=str(REMOTE_ROOT / "example_assets"),
    )
)


@app.function(image=image, timeout=3600)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    from web_app import create_app

    return create_app(
        content_dir=REMOTE_ROOT / "content",
        notebooks_dir=REMOTE_ROOT / "notebooks",
        example_assets_dir=REMOTE_ROOT / "example_assets",
    )
