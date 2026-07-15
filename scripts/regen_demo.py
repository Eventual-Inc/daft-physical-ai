#!/usr/bin/env python3
"""Regenerate a committed `examples/` demo (py + ipynb + md + image) programmatically.

`--demo hands` (default) rebuilds `examples/`; `--demo rewards` rebuilds
`examples/rewards/` (plus verbatim copies of the two Robometer server scripts).
The three formats render from one shared cell list (`daft_physical_ai._render`
/ `_render_rewards`), so they never drift. Outputs are populated by *executing*
the notebook headless and then deriving everything else from the executed copy:

  1. render the notebook (no outputs) and execute it (`nbconvert --execute`,
     `DAFT_PROGRESS_BAR=0` so Daft's progress bars don't pollute outputs);
  2. clean the executed notebook (drop run-specific `execution` timing metadata
     and the `text/plain` fallback strings that sit beside a rich image/HTML);
  3. walk it once to build the markdown's per-code-cell outputs - image cells are
     written out as a separate png and linked, stream cells are fenced as text,
     and the `.show()` HTML table is converted to a markdown table (its long
     keypoint lists abbreviated to the first point);
  4. write `demo.py`, `demo.ipynb`, `demo.md`, and the png.

Run from the repo root in the demo env. The hands demo needs the inference
stack (a Daft with the LeRobot reader, mediapipe, scipy, opencv, matplotlib,
nbconvert); the rewards demo needs huggingface_hub + matplotlib + nbconvert and
a running Robometer eval server (`ROBOMETER_URL`, plus `MODAL_KEY` /
`MODAL_SECRET` for a Modal proxy-auth deployment):

    python scripts/regen_demo.py
    ROBOMETER_URL=... python scripts/regen_demo.py --demo rewards

`--skip-exec --source <nb>` reuses an already-executed notebook instead of running
one, so steps 2-4 (the conversion logic) can run anywhere - this is how the script
is tested in the minimal dev env.
"""

from __future__ import annotations

import argparse
import base64
import copy
import html as _html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# import the package's renderers (run from the repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from daft_physical_ai import _render, _render_rewards


@dataclass
class DemoSpec:
    """One committed demo: its config, renderers, image, and output dir."""

    config: Any
    render_script: Callable[..., str]
    render_notebook: Callable[..., str]
    render_markdown: Callable[..., str]
    image_name: str
    image_alt: str
    default_dir: str


DEMOS = {
    # the local MediaPipe demo with EgoDex evaluation
    "hands": DemoSpec(
        config=_render.DemoConfig(method="mediapipe", runtime="local", with_eval=True),
        render_script=_render.render_script,
        render_notebook=_render.render_notebook,
        render_markdown=_render.render_markdown,
        image_name="demo_keypoints.png",
        image_alt="track_hands keypoints",
        default_dir="examples",
    ),
    # the Robometer reward-scoring demo on LIBERO (needs ROBOMETER_URL to execute)
    "rewards": DemoSpec(
        config=_render_rewards.RewardsDemoConfig(),
        render_script=_render_rewards.render_script,
        render_notebook=_render_rewards.render_notebook,
        render_markdown=_render_rewards.render_markdown,
        image_name="demo_progress.png",
        image_alt="Robometer per-frame task progress",
        default_dir="examples/rewards",
    ),
}


def _text(output: dict, mime: str) -> str:
    val = output.get("data", {}).get(mime, "")
    return "".join(val) if isinstance(val, list) else val


# A list of coordinate lists, e.g. kp2d's `[[x, y], [x, y], ...]`. Collapse the long
# ones to their first element so the markdown table stays readable.
_COORD_LIST = re.compile(r"\[(\[[^\[\]]*\])(?:,\s*\[[^\[\]]*\])+\]")


def _cell_text(td_html: str) -> str:
    """Plain text of one HTML table cell (Daft separates lines with <br/>)."""
    txt = re.sub(r"<br\s*/?>", " ", td_html)
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = _html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return _COORD_LIST.sub(r"[\1, ...]", txt)


def _show_table_to_markdown(html_text: str) -> str:
    """Convert a Daft `.show()` HTML table into a GitHub-flavored markdown table."""
    head = re.search(r"<thead>(.*?)</thead>", html_text, flags=re.DOTALL)
    body = re.search(r"<tbody>(.*?)</tbody>", html_text, flags=re.DOTALL)
    if not head or not body:
        return ""
    # header cells carry "name<br/>Type"; keep just the column name
    headers = [
        _html.unescape(re.sub(r"<[^>]+>", "", re.split(r"<br\s*/?>", th)[0])).strip()
        for th in re.findall(r"<th[^>]*>(.*?)</th>", head.group(1), flags=re.DOTALL)
    ]
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body.group(1), flags=re.DOTALL):
        cells = [_cell_text(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.DOTALL)]
        if cells:
            # wrap structured values (lists/structs) in backticks so they read as code
            rows.append([f"`{c}`" if c[:1] in "[{" else c for c in cells])
    if not headers or not rows:
        return ""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _clean_cell(cell: dict) -> dict:
    """Drop run-specific metadata and text/plain fallbacks that shadow a rich output."""
    cell["metadata"] = {k: v for k, v in cell.get("metadata", {}).items() if k != "execution"}
    for out in cell.get("outputs", []):
        data = out.get("data")
        if data and ("text/html" in data or any(k.startswith("image/") for k in data)):
            data.pop("text/plain", None)
    return cell


def _markdown_output(cell: dict, image_dir: Path, image_name: str, image_alt: str) -> str:
    """The one markdown output slot for a code cell (image link, stream text, or table)."""
    image_link = ""
    stream_text = ""
    table_md = ""
    for out in cell.get("outputs", []):
        png = out.get("data", {}).get("image/png")
        if png and not image_link:
            (image_dir / image_name).write_bytes(base64.b64decode("".join(png) if isinstance(png, list) else png))
            image_link = f"![{image_alt}]({image_name})"
        if out.get("output_type") == "stream":
            stream_text += "".join(out.get("text", []))
        html_out = _text(out, "text/html")
        if html_out and not table_md:
            table_md = _show_table_to_markdown(html_out)
    # image wins (the demo's payoff), else printed text, else the .show() table
    return image_link or stream_text.rstrip() or table_md


def _execute(nb_path: Path) -> None:
    env = {**os.environ, "DAFT_PROGRESS_BAR": "0"}
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb_path)],
        check=True,
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate a committed examples/ demo programmatically.")
    p.add_argument("--demo", choices=tuple(DEMOS), default="hands", help="which demo to rebuild (default: hands)")
    p.add_argument("--output-dir", help="where to write the demo (default: the demo's committed dir)")
    p.add_argument("--skip-exec", action="store_true", help="reuse --source instead of executing a fresh notebook")
    p.add_argument("--source", help="executed notebook to reuse with --skip-exec (default: <output-dir>/demo.ipynb)")
    args = p.parse_args(argv)

    spec = DEMOS[args.demo]
    out_dir = Path(args.output_dir or spec.default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / "demo.ipynb"

    # 1. obtain an executed notebook
    if args.skip_exec:
        source = Path(args.source) if args.source else nb_path
        executed = json.loads(source.read_text(encoding="utf-8"))
        print(f"reusing executed notebook: {source}")
    else:
        nb_path.write_text(spec.render_notebook(spec.config), encoding="utf-8")
        print(f"executing {nb_path} (headless)...")
        _execute(nb_path)
        executed = json.loads(nb_path.read_text(encoding="utf-8"))

    # 2. clean + 3. derive markdown outputs (one slot per code cell, in order)
    md_outputs: list[str] = []
    cleaned = copy.deepcopy(executed)
    for cell in cleaned["cells"]:
        if cell.get("cell_type") != "code":
            continue
        _clean_cell(cell)
        md_outputs.append(_markdown_output(cell, out_dir, spec.image_name, spec.image_alt))

    # 4. write all formats
    nb_path.write_text(json.dumps(cleaned, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "demo.py").write_text(spec.render_script(spec.config), encoding="utf-8")
    (out_dir / "demo.md").write_text(spec.render_markdown(spec.config, outputs=md_outputs), encoding="utf-8")

    wrote = ["demo.py", "demo.ipynb", "demo.md"]
    if (out_dir / spec.image_name).exists():
        wrote.append(spec.image_name)

    # the rewards demo tells readers the server scripts sit next to it - keep that true
    if args.demo == "rewards":
        for name in _render_rewards.SERVER_TEMPLATES:
            (out_dir / name).write_text(_render_rewards.load_server_script(name), encoding="utf-8")
            wrote.append(name)

    print("wrote:", ", ".join(str(out_dir / w) for w in wrote))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
