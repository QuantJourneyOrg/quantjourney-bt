# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Tests for the friendly strategy-process validation boundary."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from backtester.cli import strategy_runner
from backtester.sdk.client import BacktesterAccessError, PrepareValidationError


def test_missing_credentials_keep_guidance_and_failure_exit(monkeypatch, capsys) -> None:
    from backtester.sdk.client import MissingCredentialsError

    def missing_credentials(*args, **kwargs):
        raise MissingCredentialsError()

    monkeypatch.setattr(strategy_runner.runpy, "run_path", missing_credentials)
    monkeypatch.setenv("QJ_STRICT_DATA_FETCH", "0")
    assert strategy_runner.main(["configured_strategy.py"]) == 1
    output = capsys.readouterr().err
    assert "No credentials set" in output
    assert "QJ_API_KEY" in output
    assert "--sample-data" in output
    assert "Traceback" not in output


def _prepare_error() -> PrepareValidationError:
    return PrepareValidationError(
        "Backtest preparation configuration is invalid.",
        field_errors=[
            {
                "field": "provider.granularity",
                "code": "unsupported_option",
                "message": "The selected data interval is not supported.",
                "hint": "Choose one of these intervals: 1d, 1h, 90m, 30m, 15m, 5m, 2m or 1m.",
            }
        ],
        request_id="req-friendly-test",
        error_code="ERR_VAL_003",
    )


def test_renderer_shows_actionable_panel_without_traceback() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=100, color_system=None, force_terminal=False)

    strategy_runner.render_prepare_validation_error(_prepare_error(), console=console)

    output = buffer.getvalue()
    assert "Configuration needs attention" in output
    assert "The strategy stopped before market data was prepared." in output
    assert "Data interval" in output
    assert "The selected data interval is not supported." in output
    assert "Suggested fix:" in output
    assert "No trades were executed and no report was created." in output
    assert "Request req-friendly-test" in output
    assert "Error ERR_VAL_003" in output
    assert "Traceback" not in output
    assert "/bt/prepare" not in output


def test_process_boundary_catches_prepare_error(monkeypatch, capsys) -> None:
    def raise_prepare_error(*args, **kwargs):
        raise _prepare_error()

    monkeypatch.setattr(strategy_runner.runpy, "run_path", raise_prepare_error)

    assert strategy_runner.main(["broken_strategy.py"]) == 2
    output = capsys.readouterr().err
    assert "Configuration needs attention" in output
    assert "Traceback" not in output


def test_renderer_shows_product_access_instructions_without_traceback() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=100, color_system=None, force_terminal=False)
    error = BacktesterAccessError(
        request_id="req-access-test",
        error_code="ERR_AUTH_403",
    )

    strategy_runner.render_backtester_access_error(error, console=console)

    output = buffer.getvalue()
    assert "Backtester access required" in output
    assert "product:backtester" in output
    assert "feature:backtester" in output
    assert "https://users.quantjourney.cloud" in output
    assert "No trades were executed and no report was created." in output
    assert "Request req-access-test" in output
    assert "Traceback" not in output


def test_process_boundary_catches_product_access_error(monkeypatch, capsys) -> None:
    def raise_access_error(*args, **kwargs):
        raise BacktesterAccessError(request_id="req-access-test")

    monkeypatch.setattr(strategy_runner.runpy, "run_path", raise_access_error)

    assert strategy_runner.main(["restricted_strategy.py"]) == 2
    output = capsys.readouterr().err
    assert "Backtester access required" in output
    assert "Traceback" not in output


@pytest.mark.parametrize("kind", ["grid", "grid_async", "optuna"])
def test_optimizers_keep_missing_credentials_fatal(kind, monkeypatch) -> None:
    import asyncio

    from backtester.sdk.client import MissingCredentialsError
    from backtester.walkforward.optimization.grid import GridSearchOptimizer
    from backtester.walkforward.optimization.optuna_ import OptunaOptimizer

    optimizer = (
        OptunaOptimizer(param_space={}, n_trials=1, sampler="random")
        if kind == "optuna"
        else GridSearchOptimizer(param_grid={"period": [2]})
    )

    def missing_credentials(*args, **kwargs):
        raise MissingCredentialsError()

    monkeypatch.setenv("QJ_REQUIRE_CREDENTIALS", "1")
    with pytest.raises(MissingCredentialsError):
        if kind == "grid_async":
            asyncio.run(optimizer.optimize(missing_credentials, "2020-01-02", "2020-01-06", {}))
        else:
            optimizer.optimize_fn(missing_credentials)

    def ordinary_failure(*args, **kwargs):
        raise ValueError("Test-only invalid trial parameter")

    if kind == "grid_async":
        result = asyncio.run(optimizer.optimize(ordinary_failure, "2020-01-02", "2020-01-06", {}))
    else:
        result = optimizer.optimize_fn(ordinary_failure)
    assert result.trials and all(trial.state == "FAIL" for trial in result.trials)


def test_strategy_process_restores_credential_policy(monkeypatch) -> None:
    import os

    monkeypatch.delenv("QJ_REQUIRE_CREDENTIALS", raising=False)
    observed = []
    monkeypatch.setattr(
        strategy_runner.runpy,
        "run_path",
        lambda *args, **kwargs: observed.append(os.getenv("QJ_REQUIRE_CREDENTIALS")),
    )
    assert strategy_runner.main(["configured_local.py"]) == 0
    assert observed == ["1"]
    assert "QJ_REQUIRE_CREDENTIALS" not in os.environ
