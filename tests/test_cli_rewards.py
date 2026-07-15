"""Tests for the rewards demo scaffolder (renderer + CLI); prompts are stubbed, never real."""

from __future__ import annotations

import json

import pytest

from daft_physical_ai._render_rewards import (
    SERVER_TEMPLATES,
    RewardsDemoConfig,
    load_server_script,
    render_markdown,
    render_notebook,
    render_script,
)
from daft_physical_ai.cli import main


def test_rendered_script_is_valid_python() -> None:
    src = render_script(RewardsDemoConfig())
    compile(src, "demo.py", "exec")  # raises SyntaxError on bad output


def test_rendered_notebook_is_valid_ipynb() -> None:
    nb = json.loads(render_notebook(RewardsDemoConfig()))
    assert nb["nbformat"] == 4
    assert nb["cells"][0]["cell_type"] == "markdown"
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    for i, c in enumerate(code_cells):
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        compile(src, f"cell{i}", "exec")  # each cell must be valid Python


def test_markdown_renders_with_headers_and_code() -> None:
    md = render_markdown(RewardsDemoConfig())
    assert md.startswith("# ")
    assert "```python" in md


def test_markdown_and_notebook_share_content() -> None:
    cfg = RewardsDemoConfig()
    md = render_markdown(cfg)
    nb = json.loads(render_notebook(cfg))
    for c in nb["cells"]:
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        assert src in md


def test_config_reaches_generated_code() -> None:
    cfg = RewardsDemoConfig(dataset="my/dataset", split="sp", video_key="cam", episodes=3, max_frames=4)
    src = render_script(cfg)
    assert 'DATASET = "my/dataset"' in src
    assert 'SPLIT = "sp"' in src
    assert 'VIDEO_KEY = "cam"' in src
    assert "EPISODES = 3" in src
    assert "MAX_FRAMES = 4" in src
    assert "score_rewards(" in src
    assert "max_frames=MAX_FRAMES" in src


def test_server_scripts_are_valid_python() -> None:
    for name in SERVER_TEMPLATES:
        compile(load_server_script(name), name, "exec")


def test_launcher_pins_commit_and_revision() -> None:
    src = load_server_script("run_robometer_server.py")
    assert "ROBOMETER_COMMIT" in src and "HF_REVISION" in src


def test_validate_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        RewardsDemoConfig(episodes=0).validate()
    with pytest.raises(ValueError):
        RewardsDemoConfig(max_frames=0).validate()


def test_cli_default_writes_demo_and_server_scripts(tmp_path) -> None:
    out = tmp_path / "demo"
    assert main(["rewards", "--no-input", "--output-dir", str(out)]) == 0
    for name in ("demo.py", "demo.ipynb", "demo.md", *SERVER_TEMPLATES):
        assert (out / name).exists(), name
    # server scripts land verbatim from the package templates
    for name in SERVER_TEMPLATES:
        assert (out / name).read_text() == load_server_script(name)


def test_cli_format_script_only(tmp_path) -> None:
    out = tmp_path / "demo"
    assert main(["rewards", "--no-input", "--format", "script", "--output-dir", str(out)]) == 0
    assert (out / "demo.py").exists()
    assert not (out / "demo.ipynb").exists()
    assert not (out / "demo.md").exists()


def test_cli_no_server_scripts_opts_out(tmp_path) -> None:
    out = tmp_path / "demo"
    assert main(["rewards", "--no-input", "--no-server-scripts", "--output-dir", str(out)]) == 0
    assert (out / "demo.py").exists()
    for name in SERVER_TEMPLATES:
        assert not (out / name).exists()


def test_cli_flags_reach_config(tmp_path) -> None:
    out = tmp_path / "demo"
    assert (
        main(
            [
                "rewards",
                "--no-input",
                "--dataset",
                "my/dataset",
                "--episodes",
                "2",
                "--max-frames",
                "4",
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    src = (out / "demo.py").read_text()
    assert 'DATASET = "my/dataset"' in src
    assert "EPISODES = 2" in src
    assert "MAX_FRAMES = 4" in src


def test_cli_refuses_overwrite_without_force(tmp_path) -> None:
    out = tmp_path / "demo"
    assert main(["rewards", "--no-input", "--output-dir", str(out)]) == 0
    assert main(["rewards", "--no-input", "--output-dir", str(out)]) == 1
    assert main(["rewards", "--no-input", "--output-dir", str(out), "--force"]) == 0


def test_cli_bad_episodes_errors(tmp_path, capsys) -> None:
    assert main(["rewards", "--no-input", "--episodes", "0", "--output-dir", str(tmp_path / "d")]) == 2
    assert "episodes" in capsys.readouterr().err
