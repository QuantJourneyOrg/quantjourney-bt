# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Regression tests for the cross-platform repository launcher."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strategy_launcher", ROOT / "strategy.py")
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_env_loader_is_non_executing_and_does_not_override(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "must-not-exist"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "QJ_API_KEY='from-file'\n"
        "export QJ_EMAIL=user@example.com\n"
        f"MALICIOUS=$(touch {marker})\n"
        "not a valid key=value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QJ_API_KEY", "from-shell")
    monkeypatch.delenv("QJ_EMAIL", raising=False)
    monkeypatch.delenv("MALICIOUS", raising=False)

    launcher.load_env_file(env_file)

    assert os.environ["QJ_API_KEY"] == "from-shell"
    assert os.environ["QJ_EMAIL"] == "user@example.com"
    assert os.environ["MALICIOUS"].startswith("$(touch ")
    assert not marker.exists()


def test_strategy_listing_and_import_check(tmp_path: Path, monkeypatch, capsys) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    strategy = strategies / "example_demo.py"
    strategy.write_text('"""Example demo strategy."""\nVALUE = 1\n', encoding="utf-8")
    monkeypatch.setattr(launcher, "STRATEGIES_DIR", strategies)

    assert launcher.list_strategies() == 0
    assert "example_demo  Example demo strategy." in capsys.readouterr().out
    assert launcher.check_strategy(strategy) == 0
    assert "Import check passed" in capsys.readouterr().out


def test_windows_wrapper_uses_venv_and_forwards_arguments() -> None:
    wrapper = (ROOT / "strategy.bat").read_text(encoding="utf-8")

    assert r".venv\Scripts\python.exe" in wrapper
    assert '"%ROOT%strategy.py" %*' in wrapper
    assert "exit /b %ERRORLEVEL%" in wrapper


def test_shell_wrapper_uses_the_same_python_launcher() -> None:
    wrapper = (ROOT / "strategy.sh").read_text(encoding="utf-8")

    assert '"$SCRIPT_DIR/strategy.py" "$@"' in wrapper


def test_python_launcher_uses_the_friendly_isolated_runner() -> None:
    source = (ROOT / "strategy.py").read_text(encoding="utf-8")

    assert '"-m", "backtester.cli.strategy_runner"' in source
