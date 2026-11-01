# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Regression tests for the cross-platform repository launcher."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strategy_launcher", ROOT / "strategy.py")
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(autouse=True)
def restore_launcher_environment(monkeypatch) -> None:
    for key in (
        "QJ_DATA_FILE",
        "QJ_DATA_CSV",
        "QJ_CSV_CONFIG",
        "QJ_SAMPLE_DATA",
        "QJ_QUIET",
        "QJ_LOG_LEVEL",
        "QJ_NO_REPORTS",
        "QJ_OUTPUT_DIR",
        "QJ_PLOT_THEME",
        "QJ_PLOT_DPI",
    ):
        value = os.environ.get(key)
        monkeypatch.setenv(key, value if value is not None else "")
        if value is None:
            monkeypatch.delenv(key)


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


def test_launcher_accepts_input_configured_inside_strategy(tmp_path, monkeypatch) -> None:
    for key in (
        "QJ_SAMPLE_DATA",
        "QJ_DATA_CSV",
        "QJ_DATA_FILE",
        "QJ_API_KEY",
        "QJ_EMAIL",
        "QJ_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "configured_local.py"
    path.write_text('"""Local data is selected inside this strategy."""\n', encoding="utf-8")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return launcher.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", execute)
    args = launcher.build_parser().parse_args([path.stem])
    assert launcher.run_strategy(path, args) == 0
    assert calls[0][-1] == str(path)


@pytest.mark.parametrize("option", ["--data-csv", "--data-file"])
def test_csv_launcher_resolves_path_and_does_not_require_credentials(
    tmp_path, monkeypatch, option
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "configure_process_environment", lambda: None)
    monkeypatch.setattr(launcher, "runtime_is_ready", lambda: True)
    for key in [
        "QJ_SAMPLE_DATA",
        "QJ_DATA_CSV",
        "QJ_DATA_FILE",
        "QJ_CSV_CONFIG",
        "QJ_API_KEY",
        "QJ_EMAIL",
        "QJ_PASSWORD",
    ]:
        monkeypatch.delenv(key, raising=False)
    captured = []
    monkeypatch.setattr(launcher, "run_strategy", lambda path, args: captured.append(args) or 0)
    assert (
        launcher.main(["example_weights_01_sma_daily", option, "prices.csv", "--csv-config"]) == 0
    )
    attribute = "data_csv" if option == "--data-csv" else "data_file"
    assert getattr(captured[0], attribute) == str(tmp_path / "prices.csv")
    launcher.apply_run_options(captured[0])
    assert launcher.credentials_are_ready(sample_data=False, check_only=False)


def test_batch_still_fails_before_starting_cloud_strategies_without_credentials(
    monkeypatch,
) -> None:
    for key in (
        "QJ_SAMPLE_DATA",
        "QJ_DATA_CSV",
        "QJ_DATA_FILE",
        "QJ_API_KEY",
        "QJ_EMAIL",
        "QJ_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Batch started a strategy without credentials")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    args = launcher.build_parser().parse_args(["--all"])
    assert launcher.run_all(args) == 1


def test_csv_launcher_rejects_environment_conflict_and_batch_override(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "configure_process_environment", lambda: None)
    monkeypatch.setenv("QJ_SAMPLE_DATA", "1")
    with pytest.raises(SystemExit):
        launcher.main(["example_weights_01_sma_daily", "--data-csv", "prices.csv"])
    monkeypatch.delenv("QJ_SAMPLE_DATA")
    with pytest.raises(SystemExit):
        launcher.main(["--all", "--data-csv", "prices.csv", "--csv-config"])
