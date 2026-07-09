"""Render a personalized hand-tracking demo (script + notebook) from a config.

Pure string/JSON building - no I/O, no prompts - so it's easy to unit-test. The
CLI (:mod:`daft_physical_ai.cli`) collects a :class:`DemoConfig` and calls
:func:`render_script` / :func:`render_notebook`.

The generated demo imports :func:`daft_physical_ai.hands.track_hands`. The package never
imports Modal; the Modal runtime only appears in the *generated* user code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from string import Template

# Only the Modal script uses a template; local demos render from the shared cells.
_TEMPLATES = {"modal": "script_modal.py.tmpl"}

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
    with_eval: bool = False  # append EgoDex GT scoring (detect% + PCK) to the demo

    def validate(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {self.method!r}")
        if self.runtime not in _VALID_RUNTIMES:
            raise ValueError(f"runtime must be one of {_VALID_RUNTIMES}, got {self.runtime!r}")
        if self.method in ("wilor", "both") and not self.mano_path:
            raise ValueError(f"method={self.method!r} needs a mano_path (MANO_RIGHT.pkl; research-gated)")
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {self.limit}")
        if self.with_eval and self.runtime != "local":
            raise ValueError("with_eval is only supported for the local runtime")


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
    # libgl1/libglib2.0-0 + the GLES/EGL libs MediaPipe loads at runtime.
    apt = ['"git"', '"ffmpeg"', '"libgl1"', '"libglib2.0-0"', '"libgles2"', '"libegl1"']
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
    parts.append('    .pip_install("daft")')
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
    """Comma-separated quoted column names for `df.select(...)`."""
    cols = ["episode_index", "frame_index"]
    cols += ["hands_mediapipe", "hands_wilor"] if method == "both" else ["hands"]
    return ", ".join(f'"{c}"' for c in cols)


# EgoDex ground-truth scoring, appended to a demo when with_eval is set. It's an
# example concern (dataset-specific), never part of the package. The scoring runs as
# a Daft @daft.func; metrics are summarized from the collected results.
_EVAL_HELPERS = '''# --- Evaluation against EgoDex ground truth (2D, wrist + 5 fingertips) ---
# EgoDex-specific: GT hand poses live in observation.state (left = dims 0-23,
# right = 24-47); the camera is observation.extrinsics. Needs scipy + numpy.
import numpy as np
from scipy.optimize import linear_sum_assignment

import daft
from daft import DataType, col

FX = FY = 736.6339          # EgoDex camera intrinsics
CX, CY = 960.0, 540.0
SIX = [0, 4, 8, 12, 16, 20]  # wrist + 5 fingertip keypoints
THRESH = [0.1, 0.2, 0.3]     # PCK thresholds (normalized)


def _hand_pts(state, side):
    b = side * 24            # 24 dims per hand: wrist(3) + joints; we take wrist + 5 tips
    return np.vstack([state[b : b + 3], state[b + 9 : b + 24].reshape(5, 3)])


def _project(points_world, extr):
    cam_from_world = np.linalg.inv(np.asarray(extr, float).reshape(4, 4))
    cam = (cam_from_world @ np.hstack([points_world, np.ones((len(points_world), 1))]).T).T[:, :3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([FX * cam[:, 0] / z + CX, FY * cam[:, 1] / z + CY], axis=1)
    uv[z <= 0] = np.nan
    return uv


def _norm(p):               # translation + scale invariant (hand size)
    p = p - p[0]
    return p / (np.linalg.norm(p[1:], axis=1).mean() + 1e-9)


def _pair_err(gt6, pred6):  # per-keypoint error, fingertips matched by assignment
    g, m = _norm(gt6), _norm(pred6)
    d = np.linalg.norm(g[1:, None] - m[None, 1:], axis=2)
    r, c = linear_sum_assignment(d)
    return np.concatenate([[0.0], d[r, c]])


_ERR = DataType.struct({
    "n_gt": DataType.int64(),
    "n_matched": DataType.int64(),
    "errs": DataType.list(DataType.list(DataType.float64())),
})


@daft.func(return_dtype=_ERR)
def score(hands, state, extr):
    """Match predicted hands to the 2 GT hands (Hungarian on normalized error)."""
    gts = [uv for uv in (_project(_hand_pts(np.asarray(state, float), s), extr) for s in (0, 1)) if np.isfinite(uv).all()]
    preds = [np.asarray(h["kp2d"], float)[SIX] for h in (hands or [])]
    if not gts or not preds:
        return {"n_gt": len(gts), "n_matched": 0, "errs": []}
    pair = [[_pair_err(g, p) for p in preds] for g in gts]
    cost = np.array([[e.mean() for e in row] for row in pair])
    r, c = linear_sum_assignment(cost)   # match predicted hands to GT hands
    return {"n_gt": len(gts), "n_matched": len(r), "errs": [[float(x) for x in pair[i][j]] for i, j in zip(r, c)]}


def report(label, scores):
    n_gt = sum(s["n_gt"] for s in scores)
    matched = sum(s["n_matched"] for s in scores)
    errs = [e for s in scores for hand in s["errs"] for e in hand]
    mean_errs = [float(np.mean(hand)) for s in scores for hand in s["errs"]]
    pck = [100 * np.mean([e < t for e in errs]) if errs else 0.0 for t in THRESH]
    detect = 100 * matched / n_gt if n_gt else 0.0
    mean = float(np.mean(mean_errs)) if mean_errs else float("nan")
    print(f"{label:12} detect={detect:3.0f}%  mean_err={mean:.3f}  "
          f"PCK@.1/.2/.3 = {pck[0]:.0f}/{pck[1]:.0f}/{pck[2]:.0f}")'''


def _eval_methods(method: str) -> list[tuple[str, str]]:
    """(label, hands-column) pairs to score."""
    if method == "both":
        return [("WiLoR", "hands_wilor"), ("MediaPipe", "hands_mediapipe")]
    return [(_title_method(method), "hands")]


def _title_method(method: str) -> str:
    return {"mediapipe": "MediaPipe", "wilor": "WiLoR"}[method]


# Keypoint visualization (default for local demos). Draws the predicted skeleton on
# a few frames with cv2 (which ships with the mediapipe/wilor extras) + matplotlib.
_VIZ_HELPERS = """# --- Visualize: draw the predicted keypoints on a few frames ---
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 21-keypoint hand skeleton (wrist + 5 fingers x 4 joints)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def draw_hands(img, hands):
    img = np.ascontiguousarray(img)
    for h in hands or []:
        kp = np.asarray(h["kp2d"], float)
        for a, b in BONES:
            cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (60, 200, 60), 2)
        for p in kp:
            cv2.circle(img, tuple(p.astype(int)), 3, (255, 80, 0), -1)
    return img"""


_GT_DRAW = '''
def draw_gt(img, state, extr):
    """Draw the projected EgoDex GT hands (wrist + fingertips) in green."""
    img = np.ascontiguousarray(img)
    for side in (0, 1):
        uv = _project(_hand_pts(np.asarray(state, float), side), extr)
        if not np.isfinite(uv).all():
            continue
        wrist = tuple(uv[0].astype(int))
        for tip in uv[1:]:
            cv2.line(img, wrist, tuple(tip.astype(int)), (0, 220, 0), 2)
            cv2.circle(img, tuple(tip.astype(int)), 5, (255, 0, 0), -1)
        cv2.circle(img, wrist, 6, (0, 120, 255), -1)
    return img'''


def _viz_cells(config: DemoConfig) -> list[tuple[str, str]]:
    """Visualization cells: a GT-vs-prediction montage with eval, else an overlay."""
    methods = _eval_methods(config.method)  # (label, hands-column) pairs
    method_cols = ", ".join(f'"{c}"' for _, c in methods)
    methods_lit = "[" + ", ".join(f'("{lbl}", "{c}")' for lbl, c in methods) + "]"

    if config.with_eval:
        # GT projection helpers (_hand_pts/_project) already defined by the eval cells above.
        comparison = (
            f'viz = df.select(IMAGE_COLUMN, "observation.state", "observation.extrinsics", '
            f"{method_cols}).limit(4).to_pydict()\n"
            f'columns = [("GT", None)] + {methods_lit}\n'
            'n = len(viz["frame_index"]) if "frame_index" in viz else len(viz[IMAGE_COLUMN])\n'
            "fig, axes = plt.subplots(n, len(columns), figsize=(3 * len(columns), 3 * n), squeeze=False)\n"
            "for i in range(n):\n"
            "    img = np.asarray(viz[IMAGE_COLUMN][i])\n"
            "    for jc, (label, c) in enumerate(columns):\n"
            '        cell = (draw_gt(img.copy(), viz["observation.state"][i], viz["observation.extrinsics"][i])\n'
            "                if c is None else draw_hands(img.copy(), viz[c][i]))\n"
            "        axes[i][jc].imshow(cell)\n"
            "        axes[i][jc].set_xticks([])\n"
            "        axes[i][jc].set_yticks([])\n"
            "        if i == 0:\n"
            "            axes[i][jc].set_title(label)\n"
            'fig.suptitle("Ground truth vs predictions")\n'
            "plt.tight_layout()\n"
            "plt.show()"
        )
        return [
            (
                "markdown",
                "## Visualize: ground truth vs predictions\n\nEach row is a frame; the first column is the "
                "EgoDex ground-truth hands (green), the rest are the predicted keypoints. This is the most "
                "telling view - you see where each method is right and where it misses.",
            ),
            ("code", _VIZ_HELPERS + "\n\n" + _GT_DRAW.strip()),
            ("code", comparison),
        ]

    overlay = (
        f"viz = df.select(IMAGE_COLUMN, {method_cols}).limit(4).to_pydict()\n"
        "frames = [np.asarray(im) for im in viz[IMAGE_COLUMN]]\n"
        f"methods = {methods_lit}\n"
        "fig, axes = plt.subplots(len(methods), len(frames), "
        "figsize=(3 * len(frames), 3 * len(methods)), squeeze=False)\n"
        "for r, (label, c) in enumerate(methods):\n"
        "    for j, (im, hands) in enumerate(zip(frames, viz[c])):\n"
        "        axes[r][j].imshow(draw_hands(im, hands))\n"
        "        axes[r][j].set_xticks([])\n"
        "        axes[r][j].set_yticks([])\n"
        "    axes[r][0].set_ylabel(label, fontsize=12)\n"
        'fig.suptitle("track_hands keypoints")\n'
        "plt.tight_layout()\n"
        "plt.show()"
    )
    return [
        (
            "markdown",
            "## Visualize\n\nDraw the predicted keypoints on a few frames - this is the point of hand "
            "tracking, so let's see it. (Needs `matplotlib`; `cv2` ships with the method extra.)",
        ),
        ("code", _VIZ_HELPERS),
        ("code", overlay),
    ]


def _eval_cells_select_and_report(config: DemoConfig) -> tuple[str, str]:
    """The score-columns + report lines shared by the eval cells."""
    methods = _eval_methods(config.method)
    score_lines = "\n".join(
        f'df = df.with_column("score_{c}", score(col("{c}"), col("observation.state"), col("observation.extrinsics")))'
        for _, c in methods
    )
    select_cols = ", ".join(f'"score_{c}"' for _, c in methods)
    report_lines = "\n".join(f'report("{label}", scored["score_{c}"])' for label, c in methods)
    return f"{score_lines}\nscored = df.select({select_cols}).to_pydict()", report_lines


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


# Script form: `modal run demo.py` invokes this local entrypoint.
_MODAL_ENTRYPOINT = """

@app.local_entrypoint()
def main():
    out = run.remote()
    print(f"got {len(out['frame_index'])} frames back from Modal")
"""

# Notebook form: Modal can't use a local entrypoint in a kernel, so drive it with
# `app.run()` and pull the result back. Needs the kernel's Python to match the
# image (Modal serializes notebook-defined funcs). See:
# https://modal.com/docs/guide/jupyter-notebooks
_MODAL_NOTEBOOK_RUN = """with modal.enable_output():
    with app.run():
        out = run.remote()

print(f"got {len(out['frame_index'])} frames back from Modal")"""


def _render_modal(config: DemoConfig, tmpl: Template, entrypoint: str) -> str:
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
        entrypoint=entrypoint,
    )


def render_script(config: DemoConfig) -> str:
    """Render the standalone .py demo for this config."""
    config.validate()
    if config.runtime == "modal":
        tmpl = Template(_load_template(_TEMPLATES["modal"]))
        return _render_modal(config, tmpl, entrypoint=_MODAL_ENTRYPOINT)
    # local: render from the same cells as the notebook/markdown so all three match -
    # the first markdown cell becomes the docstring, the rest become comments.
    return _cells_to_script(_demo_cells(config))


def _cells_to_script(cells: list[tuple[str, str]]) -> str:
    parts = []
    for i, (kind, text) in enumerate(cells):
        if kind == "code":
            parts.append(text)
        elif i == 0:
            parts.append(f'"""{text.removeprefix("# ")}\n"""')
        else:
            parts.append("\n".join(f"# {_unhash(ln)}" if ln.strip() else "#" for ln in text.splitlines()))
    return "\n\n".join(parts) + "\n"


def _unhash(line: str) -> str:
    """Turn a markdown header line into plain comment text (## Foo -> Foo)."""
    return line.lstrip("#").lstrip() if line.lstrip().startswith("#") else line


def _install_hint(method: str) -> str:
    extra = {"mediapipe": "mediapipe", "wilor": "wilor", "both": "mediapipe,wilor"}[method]
    return f'"daft-physical-ai[{extra}]" matplotlib'


def render_notebook(config: DemoConfig) -> str:
    """Render the demo as a Jupyter notebook (.ipynb)."""
    config.validate()
    return _build_ipynb(_demo_cells(config))


def render_markdown(config: DemoConfig, outputs: list[str] | None = None) -> str:
    """Render the demo as a Markdown tutorial (prose + fenced code), for reading.

    `outputs`, if given, is one text output per code cell in order (empty = none);
    each non-empty output is shown as a fenced block after its code, so readers see
    results without running anything.
    """
    config.validate()
    out_iter = iter(outputs or [])
    parts = []
    for kind, text in _demo_cells(config):
        if kind == "markdown":
            parts.append(text)
            continue
        block = f"```python\n{text}\n```"
        captured = next(out_iter, "")  # one slot per code cell, in cell order
        if captured.strip():
            # raw markdown (an image link `![...]` or a `|`-delimited table) is appended
            # as-is; anything else is fenced as plain text
            is_markdown = captured.lstrip().startswith(("![", "|"))
            extra = captured.rstrip() if is_markdown else f"```\n{captured.rstrip()}\n```"
            block += f"\n\n{extra}"
        parts.append(block)
    return "\n\n".join(parts) + "\n"


def _demo_cells(config: DemoConfig) -> list[tuple[str, str]]:
    """Ordered (markdown|code) cells - the shared source for notebook and markdown."""
    method_name = {"mediapipe": "MediaPipe", "wilor": "WiLoR", "both": "MediaPipe and WiLoR"}[config.method]
    intro = (
        f"# {_title(config)}\n\n"
        f"This demo reads a LeRobot dataset, runs "
        f"hand tracking ({method_name}) as a Daft UDF with `track_hands`, and shows the keypoints. "
        "Every method returns the same schema: a list of "
        "`{handedness, confidence, kp2d, kp3d?}` per frame (`kp3d` is null for MediaPipe)."
    )
    cells: list[tuple[str, str]] = [("markdown", intro)]

    if config.runtime == "modal":
        tmpl = Template(_load_template(_TEMPLATES["modal"]))
        cells += [
            (
                "markdown",
                "## Define the Modal app\n\nThe image, the dataset volume, and the remote function "
                "that runs the Daft pipeline on a Modal worker. Modal lives only in this demo - "
                "`daft-physical-ai` itself has no Modal dependency.",
            ),
            ("code", _render_modal(config, tmpl, entrypoint="").rstrip()),
            (
                "markdown",
                "## Run on Modal\n\n`app.run()` spins up the app, runs the function on Modal, and "
                "pulls the results back here.\n\n> Modal serializes notebook-defined functions, so "
                "this kernel's Python must match the image (3.11). Prefer the script form? Run "
                "`modal run demo.py` instead.",
            ),
            ("code", _MODAL_NOTEBOOK_RUN),
        ]
    else:
        cells += [
            ("markdown", f"## Setup\n\nInstall with `pip install {_install_hint(config.method)}`, then import."),
            ("code", "from daft.datasets import lerobot\n\nfrom daft_physical_ai.hands import track_hands"),
            ("markdown", "## Configure\n\nThe dataset, the camera column to decode, and how many frames to run."),
            ("code", _config_block(config)),
            (
                "markdown",
                "## Read the dataset\n\nThe LeRobot reader gives one row per frame, decoding the "
                "camera into an image column.",
            ),
            ("code", "df = lerobot.read(DATASET, load_video_frames=IMAGE_COLUMN).limit(LIMIT)"),
            (
                "markdown",
                "## Track hands\n\n`track_hands` returns a hand-pose column. It's a lazy, batched "
                "Daft UDF, so nothing runs until we materialize below.",
            ),
            ("code", _track_lines(config.method)),
            ("markdown", "## Inspect the results\n\n`.show()` triggers execution and renders the keypoints per frame."),
            ("code", f"df.select({_result_columns(config.method)}).show()"),
        ]
        # eval first so its GT-projection helpers are available to the comparison viz
        if config.with_eval:
            cells += _eval_cells(config)
        cells += _viz_cells(config)
    return cells


def _eval_cells(config: DemoConfig) -> list[tuple[str, str]]:
    """Notebook cells appending the EgoDex GT evaluation."""
    score_block, report_lines = _eval_cells_select_and_report(config)
    return [
        (
            "markdown",
            "## Evaluate against ground truth\n\nEgoDex ships per-frame GT hand poses, so we can score the "
            "predictions: project both GT hands, match the predicted hands to them, and report detect% + PCK. "
            "The matching runs as a Daft UDF (`score`); the summary is computed from the collected results.\n\n"
            "> EgoDex-specific (GT layout + camera intrinsics). Needs `pip install scipy`.",
        ),
        ("code", _EVAL_HELPERS),
        ("code", score_block),
        ("code", f'print("EgoDex 2D accuracy:")\n{report_lines}'),
    ]


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
