"""Tests for the demo scaffolder (renderer + CLI), all non-interactive."""

from __future__ import annotations

import json

import pytest

from daft_physical_ai._render import DemoConfig, render_notebook, render_script
from daft_physical_ai.cli import main

ALL_COMBOS = [(method, runtime) for method in ("mediapipe", "wilor", "both") for runtime in ("local", "modal")]


def _cfg(method: str, runtime: str) -> DemoConfig:
    mano = "/weights/MANO_RIGHT.pkl" if method in ("wilor", "both") else None
    return DemoConfig(method=method, runtime=runtime, mano_path=mano)


@pytest.mark.parametrize(("method", "runtime"), ALL_COMBOS)
def test_rendered_script_is_valid_python(method: str, runtime: str) -> None:
    src = render_script(_cfg(method, runtime))
    compile(src, "demo.py", "exec")  # raises SyntaxError on bad output


@pytest.mark.parametrize(("method", "runtime"), ALL_COMBOS)
def test_rendered_notebook_is_valid_ipynb(method: str, runtime: str) -> None:
    nb = json.loads(render_notebook(_cfg(method, runtime)))
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert any(c["cell_type"] == "code" for c in nb["cells"])


def test_method_calls_match_choice() -> None:
    assert 'method="mediapipe"' in render_script(_cfg("mediapipe", "local"))
    assert 'method="wilor"' in render_script(_cfg("wilor", "local"))
    both = render_script(_cfg("both", "local"))
    assert 'method="mediapipe"' in both and 'method="wilor"' in both


def test_modal_uses_container_mano_path_and_mounts_host_path() -> None:
    src = render_script(_cfg("wilor", "modal"))
    assert 'MANO_PATH = "/mano/MANO_RIGHT.pkl"' in src  # container path used by track_hands
    assert '.add_local_file("/weights/MANO_RIGHT.pkl"' in src  # host path mounted in
    assert 'gpu="L4"' in src


def test_mediapipe_modal_has_no_gpu_or_torch() -> None:
    src = render_script(_cfg("mediapipe", "modal"))
    assert "gpu=" not in src
    assert "torch" not in src


def test_validate_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="mano_path"):
        DemoConfig(method="wilor", runtime="local").validate()
    with pytest.raises(ValueError, match="method"):
        DemoConfig(method="nope").validate()
    with pytest.raises(ValueError, match="runtime"):
        DemoConfig(runtime="nope").validate()
    with pytest.raises(ValueError, match="limit"):
        DemoConfig(limit=0).validate()


def test_cli_writes_both_files(tmp_path) -> None:
    rc = main(["--method", "mediapipe", "--runtime", "local", "--output-dir", str(tmp_path / "d"), "--no-input"])
    assert rc == 0
    assert (tmp_path / "d" / "demo.py").exists()
    assert (tmp_path / "d" / "demo.ipynb").exists()


def test_cli_format_script_only(tmp_path) -> None:
    rc = main(["--method", "mediapipe", "--output-dir", str(tmp_path / "d"), "--format", "script", "--no-input"])
    assert rc == 0
    assert (tmp_path / "d" / "demo.py").exists()
    assert not (tmp_path / "d" / "demo.ipynb").exists()


def test_cli_wilor_without_mano_errors(tmp_path, capsys) -> None:
    rc = main(["--method", "wilor", "--output-dir", str(tmp_path / "d"), "--no-input"])
    assert rc == 2
    assert "mano" in capsys.readouterr().err.lower()
    assert not (tmp_path / "d").exists()


def test_cli_refuses_overwrite_without_force(tmp_path) -> None:
    args = ["--method", "mediapipe", "--output-dir", str(tmp_path / "d"), "--no-input"]
    assert main(args) == 0
    assert main(args) == 1  # second run: files exist
    assert main(args + ["--force"]) == 0  # force overwrites
