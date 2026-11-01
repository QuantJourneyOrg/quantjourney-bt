# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Isolated strategy process with friendly handling for prepare validation."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from backtester.sdk.client import (
    BacktesterAccessError,
    MissingCredentialsError,
    PrepareValidationError,
)

_FIELD_LABELS = {
    "instruments": "Instruments",
    "provider.source": "Data source",
    "provider.granularity": "Data interval",
    "trading_range": "Date range",
    "trading_range.start": "Start date",
    "trading_range.end": "End date",
    "backtest_period": "Date range",
    "backtest_period.start": "Start date",
    "backtest_period.end": "End date",
    "configuration": "Backtest size",
}


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _field_label(field: str) -> str:
    if field in _FIELD_LABELS:
        return _FIELD_LABELS[field]
    parts = str(field or "configuration").split(".")
    return " → ".join(part.replace("_", " ").strip().title() for part in parts)


def render_prepare_validation_error(
    error: PrepareValidationError,
    *,
    console: Console | None = None,
) -> None:
    """Render one actionable terminal panel without a traceback."""

    output = console or Console(
        stderr=True,
        highlight=False,
        force_terminal=True if _enabled("QJ_FORCE_COLOR") else None,
    )
    body = Text()
    body.append("The strategy stopped before market data was prepared.\n")
    body.append("Fix the configuration below and run it again.", style="bold")

    for index, issue in enumerate(error.validation_errors, start=1):
        field_label = _field_label(issue.get("field", "configuration"))
        body.append(f"\n\n{index}. {field_label}", style="bold yellow")
        body.append(f"\n   {issue.get('message', 'This value is invalid.')}")
        hint = issue.get("hint")
        if hint:
            body.append("\n   Suggested fix: ", style="bold")
            body.append(str(hint))

    body.append("\n\nNo trades were executed and no report was created.", style="dim")
    trace = " · ".join(
        value
        for value in (
            f"Request {error.request_id}" if error.request_id else "",
            f"Error {error.error_code}" if error.error_code else "",
        )
        if value
    )
    if trace:
        body.append(f"\n{trace}", style="dim")

    output.print(
        Panel(
            body,
            title=Text(" Configuration needs attention ", style="bold yellow"),
            border_style="yellow",
            expand=False,
            padding=(1, 2),
        )
    )

    if os.getenv("QJ_LOG_LEVEL", "").strip().upper() == "DEBUG":
        output.print(
            f"[dim]Technical: endpoint={error.endpoint} status={error.status_code} "
            f"code={error.error_code or '-'} request_id={error.request_id or '-'}[/dim]"
        )

    if _enabled("QJ_ERROR_TEST_MODE"):
        print(
            "QJ_PREPARE_ERROR="
            + json.dumps(error.to_dict(), ensure_ascii=True, separators=(",", ":")),
            file=sys.stderr,
        )


def render_backtester_access_error(
    error: BacktesterAccessError,
    *,
    console: Console | None = None,
) -> None:
    """Render the product entitlement denial without a Python traceback."""

    output = console or Console(
        stderr=True,
        highlight=False,
        force_terminal=True if _enabled("QJ_FORCE_COLOR") else None,
    )
    body = Text()
    body.append("This API key cannot run hosted Backtester jobs.\n")
    body.append("Activate Backtester for the account, then create a restricted key with:\n\n")
    for scope in error.required_scopes:
        body.append(f"  {scope}\n", style="bold yellow")
    body.append(f"\nAccount and API keys: {error.account_url}")
    body.append(f"\nBacktester: {error.product_url}")
    body.append("\n\nNo trades were executed and no report was created.", style="dim")
    trace = " · ".join(
        value
        for value in (
            f"Request {error.request_id}" if error.request_id else "",
            f"Error {error.error_code}" if error.error_code else "",
        )
        if value
    )
    if trace:
        body.append(f"\n{trace}", style="dim")

    output.print(
        Panel(
            body,
            title=Text(" Backtester access required ", style="bold yellow"),
            border_style="yellow",
            expand=False,
            padding=(1, 2),
        )
    )

    if os.getenv("QJ_LOG_LEVEL", "").strip().upper() == "DEBUG":
        output.print(
            f"[dim]Technical: endpoint={error.endpoint} status={error.status_code} "
            f"code={error.error_code or '-'} request_id={error.request_id or '-'}[/dim]"
        )

    if _enabled("QJ_ERROR_TEST_MODE"):
        print(
            "QJ_ACCESS_ERROR="
            + json.dumps(error.to_dict(), ensure_ascii=True, separators=(",", ":")),
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python -m backtester.cli.strategy_runner <strategy.py>", file=sys.stderr)
        return 2

    strategy_path = Path(args[0]).resolve()
    os.environ.setdefault("QJ_FRIENDLY_ERRORS", "1")
    previous_requirement = os.environ.get("QJ_REQUIRE_CREDENTIALS")
    os.environ["QJ_REQUIRE_CREDENTIALS"] = "1"
    try:
        runpy.run_path(str(strategy_path), run_name="__main__")
    except MissingCredentialsError as error:
        print(str(error), file=sys.stderr)
        return 1
    except BacktesterAccessError as error:
        render_backtester_access_error(error)
        return 2
    except PrepareValidationError as error:
        render_prepare_validation_error(error)
        return 2
    finally:
        if previous_requirement is None:
            os.environ.pop("QJ_REQUIRE_CREDENTIALS", None)
        else:
            os.environ["QJ_REQUIRE_CREDENTIALS"] = previous_requirement
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
