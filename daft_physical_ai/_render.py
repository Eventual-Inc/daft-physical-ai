"""Render a personalized hand-tracking demo (script + notebook) from a config.

Pure string/JSON building - no I/O, no prompts - so it's easy to unit-test. The
CLI (:mod:`daft_physical_ai.cli`) collects a :class:`DemoConfig` and calls
:func:`render_script` / :func:`render_notebook`.

The generated demo imports :func:`daft_physical_ai.track_hands`. The package never
imports Modal; the Modal runtime only appears in the *generated* user code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from string import Template

_TEMPLATES = {
    "local": "script_local.py.tmpl",
    "modal": "script_modal.py.tmpl",
}

_VALID_METHODS = ("mediapipe", "wilor", "both")
_VALID_RUNTIMES = ("local", "modal")


@dataclass
class DemoConfig:
    """Everything needed to render a demo."""

    method: str = "mediapipe"
    runtime: str = "local"
    mano_path: str | None = None
    dataset: str = "pepijn223/egodex-test"
    image_column: str = "observation.image"
    limit: int = 12

    def validate(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {self.method!r}")
        if self.runtime not in _VALID_RUNTIMES:
            raise ValueError(f"runtime must be one of {_VALID_RUNTIMES}, got {self.runtime!r}")
        if self.method in ("wilor", "both") and not self.mano_path:
            raise ValueError(f"method={self.method!r} needs a mano_path (MANO_RIGHT.pkl; research-gated)")
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {self.limit}")


def _uses(method: str, name: str) -> bool:
    return method == name or method == "both"


def _indent_cont(text: str, n: int) -> str:
    """Indent every line except the first by n spaces.

    `string.Template` only indents a placeholder's first line (from the template
    text before it); continuation lines of a multi-line value land at column 0.
    """
    lines = text.split("\n")
    pad = " " * n
    return "\n".join([lines[0]] + [pad + line if line else line for line in lines[1:]])


def _track_lines(method: str, wilor_extra: str = "") -> str:
    """The df.with_column(...) call(s) for the chosen method(s).

    wilor_extra is appended to the WiLoR call (e.g. wilor_root/device for Modal).
    """
    lines = []
    if _uses(method, "mediapipe"):
        col = "hands_mediapipe" if method == "both" else "hands"
        lines.append(f'df = df.with_column("{col}", track_hands(df[IMAGE_COLUMN], method="mediapipe"))')
    if _uses(method, "wilor"):
        col = "hands_wilor" if method == "both" else "hands"
        lines.append(
            f'df = df.with_column("{col}", '
            f'track_hands(df[IMAGE_COLUMN], method="wilor", mano_path=MANO_PATH{wilor_extra}))'
        )
    return "\n".join(lines)


def _modal_image_block(config: DemoConfig) -> str:
    """The `image = (...)` Modal recipe, installing only what the chosen method needs.

    Grounded in the recipe verified in TESTING.md. WiLoR assets (repo + weights) and
    MANO are set up at build time so replicas don't race to clone.
    """
    apt = ['"git"', '"ffmpeg"', '"libgl1"', '"libglib2.0-0"']
    pip = ['"daft"', '"av"', '"numpy<2"', '"pillow"', '"hf-transfer"', '"huggingface_hub"']
    if _uses(config.method, "mediapipe"):
        pip.append('"mediapipe"')

    parts = [
        "image = (",
        '    modal.Image.debian_slim(python_version="3.11")',
        f"    .apt_install({', '.join(apt)})",
        '    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})',
    ]
    if _uses(config.method, "wilor"):
        parts += [
            '    .pip_install("torch==2.1.2", "torchvision==0.16.2",',
            '                 index_url="https://download.pytorch.org/whl/cu121")',
        ]
    parts.append('    .pip_install("daft", extra_index_url="https://nightly.daft.ai", pre=True)')
    if _uses(config.method, "wilor"):
        wilor_pip = (
            '"opencv-python-headless", "pytorch-lightning==2.1.3", "scikit-image", '
            '"smplx==0.1.28", "yacs", "timm", "einops", "xtcocotools", "pandas", '
            '"hydra-core", "hydra-colorlog", "pyrootutils", "rich", "webdataset", '
            '"ultralytics==8.1.34", "pyrender", '
            '"chumpy @ git+https://github.com/mattloper/chumpy"'
        )
        parts.append(f"    .pip_install({', '.join(pip)}, {wilor_pip})")
    else:
        parts.append(f"    .pip_install({', '.join(pip)})")
    parts.append('    .pip_install("daft-physical-ai")')
    if _uses(config.method, "wilor"):
        # Mount MANO, then set up the WiLoR repo + weights at build time (idempotent),
        # so runtime replicas don't race to clone. See _modal_setup_fn.
        parts += [
            f'    .add_local_file("{config.mano_path}", "/mano/MANO_RIGHT.pkl", copy=True)',
            "    .run_function(_setup_wilor)",
        ]
    parts.append(")")
    return "\n".join(parts)


def _modal_setup_fn(config: DemoConfig) -> str:
    """A build-time setup function for WiLoR assets (empty for MediaPipe-only)."""
    if not _uses(config.method, "wilor"):
        return ""
    return (
        "def _setup_wilor():\n"
        "    from daft_physical_ai.hands._wilor import ensure_assets\n"
        '    ensure_assets("/WiLoR", "/mano/MANO_RIGHT.pkl")\n'
    )


def _result_columns(method: str) -> str:
    """Python list literal of the columns to display."""
    base = ["episode_index", "frame_index"]
    hands = ["hands_mediapipe", "hands_wilor"] if method == "both" else ["hands"]
    return repr(base + hands)


def _config_block(config: DemoConfig) -> str:
    lines = [
        f'DATASET = "{config.dataset}"',
        f'IMAGE_COLUMN = "{config.image_column}"',
        f"LIMIT = {config.limit}",
    ]
    if config.mano_path:
        # On Modal the weights are mounted into the image, so point at the container path.
        mano = "/mano/MANO_RIGHT.pkl" if config.runtime == "modal" else config.mano_path
        lines.append(f'MANO_PATH = "{mano}"')
    return "\n".join(lines)


def _load_template(name: str) -> str:
    from importlib.resources import files

    return (files("daft_physical_ai.templates") / name).read_text(encoding="utf-8")


def _title(config: DemoConfig) -> str:
    method = {"mediapipe": "MediaPipe", "wilor": "WiLoR", "both": "MediaPipe + WiLoR"}[config.method]
    return f"Hand tracking demo - {method} ({config.runtime} runtime)"


def render_script(config: DemoConfig) -> str:
    """Render the standalone .py demo for this config."""
    config.validate()
    tmpl = Template(_load_template(_TEMPLATES[config.runtime]))
    if config.runtime == "modal":
        gpu = ', gpu="L4"' if _uses(config.method, "wilor") else ""
        track = _track_lines(config.method, wilor_extra=', wilor_root="/WiLoR", device="cuda"')
        return tmpl.substitute(
            title=_title(config),
            app_name=f"hand-tracking-{config.method}",
            modal_setup_fn=_modal_setup_fn(config),
            modal_image=_modal_image_block(config),
            modal_gpu=gpu,
            config_block=_config_block(config),
            track_lines=_indent_cont(track, 4),  # $track_lines sits in the function body
            result_columns=_result_columns(config.method),
        )
    return tmpl.substitute(
        title=_title(config),
        config_block=_config_block(config),
        track_lines=_track_lines(config.method),
        result_columns=_result_columns(config.method),
    )


def render_notebook(config: DemoConfig) -> str:
    """Render the demo as a Jupyter notebook (.ipynb JSON) with the same steps as cells."""
    config.validate()
    intro = (
        f"# {_title(config)}\n\n"
        "Generated by `daft-physical-ai`. Reads a LeRobot dataset, runs hand tracking "
        "as a Daft UDF, and shows the annotated frames."
    )
    cells: list[tuple[str, str]] = [("markdown", intro)]
    if config.runtime == "modal":
        # The whole Modal script is one code cell (it must run as a module via `modal run`).
        cells.append(("code", render_script(config)))
    else:
        cells.append(
            ("code", "import daft\nfrom daft.datasets import lerobot\n\nfrom daft_physical_ai import track_hands")
        )
        cells.append(("code", _config_block(config)))
        cells.append(("code", "df = lerobot.read(DATASET, load_video_frames=IMAGE_COLUMN).limit(LIMIT)"))
        cells.append(("code", _track_lines(config.method)))
        cells.append(("code", f"df.select(*{_result_columns(config.method)}).show()"))
    return _build_ipynb(cells)


def _build_ipynb(cells: list[tuple[str, str]]) -> str:
    def _src(text: str) -> list[str]:
        # nbformat stores source as a list of lines, each keeping its trailing newline.
        lines = text.splitlines(keepends=True)
        return lines or [""]

    nb_cells = []
    for kind, text in cells:
        cell: dict = {"cell_type": kind, "metadata": {}, "source": _src(text)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1) + "\n"
