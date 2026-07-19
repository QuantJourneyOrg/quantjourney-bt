# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Tests for the friendly strategy-process validation boundary."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from backtester.cli import strategy_runner
from backtester.sdk.client import PrepareValidationError


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
