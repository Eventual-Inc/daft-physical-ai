"""Tests for the trim demo scaffolder (renderer + CLI); prompts are stubbed, never real."""

from __future__ import annotations

import json

import pytest

from daft_physical_ai._render_trim import (
    TrimDemoConfig,
    render_markdown,
    render_notebook,
    render_script,
)
from daft_physical_ai.cli import main


def test_rendered_script_is_valid_python() -> None:
    src = render_script(TrimDemoConfig())
    compile(src, "demo.py", "exec")  # raises SyntaxError on bad output


def test_rendered_notebook_is_valid_ipynb() -> None:
    nb = json.loads(render_notebook(TrimDemoConfig()))
    assert nb["nbformat"] == 4
    assert nb["cells"][0]["cell_type"] == "markdown"
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    for i, c in enumerate(code_cells):
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        compile(src, f"cell{i}", "exec")  # each cell must be valid Python


def test_markdown_renders_with_headers_and_code() -> None:
    md = render_markdown(TrimDemoConfig())
    assert md.startswith("# ")
    assert "```python" in md


def test_markdown_and_notebook_share_content() -> None:
    cfg = TrimDemoConfig()
    md = render_markdown(cfg)
    nb = json.loads(render_notebook(cfg))
    for c in nb["cells"]:
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        assert src in md


def test_config_reaches_generated_code() -> None:
    cfg = TrimDemoConfig(dataset="my/dataset", state="obs.state", dims=5, fps=30, shards=2)
    src = render_script(cfg)
    assert 'DATASET = "my/dataset"' in src
    assert 'STATE = "obs.state"' in src
    assert "DIMS = 5" in src
    assert "FPS = 30" in src
    assert "SHARDS = 2" in src
    assert "motion_energy(" in src
    assert "trim_windows(" in src


def test_demo_filters_the_orphan_recordings() -> None:
    # the droid_1.0.1 episode_index collision - the demo must keep canonical rows only
    src = render_script(TrimDemoConfig())
    assert 'col("index") == col("dataset_from_index") + col("frame_index")' in src


def test_validate_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        TrimDemoConfig(dims=0).validate()
    with pytest.raises(ValueError):
        TrimDemoConfig(fps=0).validate()
    with pytest.raises(ValueError):
        TrimDemoConfig(shards=0).validate()


def test_cli_default_writes_all_three_formats(tmp_path) -> None:
    out = tmp_path / "demo"
    assert main(["trim", "--no-input", "--output-dir", str(out)]) == 0
    for name in ("demo.py", "demo.ipynb", "demo.md"):
        assert (out / name).exists(), name


def test_cli_format_script_only(tmp_path) -> None:
    out = tmp_path / "demo"
    assert main(["trim", "--no-input", "--format", "script", "--output-dir", str(out)]) == 0
    assert (out / "demo.py").exists()
    assert not (out / "demo.ipynb").exists()
    assert not (out / "demo.md").exists()


def test_cli_flags_reach_config(tmp_path) -> None:
    out = tmp_path / "demo"
    args = ["trim", "--no-input", "--dataset", "my/dataset", "--dims", "5", "--shards", "3"]
    assert main([*args, "--output-dir", str(out)]) == 0
    src = (out / "demo.py").read_text()
    assert 'DATASET = "my/dataset"' in src
    assert "DIMS = 5" in src
    assert "SHARDS = 3" in src


def test_cli_refuses_overwrite_without_force(tmp_path) -> None:
    out = tmp_path / "demo"
    assert main(["trim", "--no-input", "--output-dir", str(out)]) == 0
    assert main(["trim", "--no-input", "--output-dir", str(out)]) == 1
    assert main(["trim", "--no-input", "--output-dir", str(out), "--force"]) == 0


def test_cli_bad_dims_errors(tmp_path, capsys) -> None:
    assert main(["trim", "--no-input", "--dims", "0", "--output-dir", str(tmp_path / "d")]) == 2
    assert "dims" in capsys.readouterr().err
