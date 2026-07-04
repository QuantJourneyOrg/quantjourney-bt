# QuantJourney Backtester

**Local quantitative strategy backtesting powered by QuantJourney market data**

[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/quantjourney-bt?color=orange)](https://pypi.org/project/quantjourney-bt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)]()
[![API](https://img.shields.io/badge/API-QuantJourney%20Cloud-1B4F72)](https://quantjourney.cloud)
[![Changelog](https://img.shields.io/badge/Changelog-backtester.quantjourney.cloud-111827)](https://backtester.quantjourney.cloud/changelog)

QuantJourney Backtester is a Python framework for researching, testing, and
reviewing systematic trading strategies. The cloud API supplies market data;
strategy logic, portfolio accounting, execution simulation, metrics, and report
generation run locally in Python.

## Example Output

Every run produces an institutional-quality report — equity curves, a monthly
returns heatmap, crisis analysis, risk and rolling statistics, a trade blotter,
and walk-forward / optimization diagnostics. A few examples:

**Cumulative returns with regime overlay**

![Cumulative returns with regime overlay](https://backtester.quantjourney.cloud/plots/cumulative_returns_with_regime.png)

**Monthly returns heatmap**

![Monthly returns heatmap](https://backtester.quantjourney.cloud/plots/monthly_returns_heatmap.png)

**Crisis analysis across historical stress periods**

![Crisis analysis](https://backtester.quantjourney.cloud/plots/crisis_summary.png)

**Walk-forward out-of-sample equity**

![Walk-forward out-of-sample equity](https://backtester.quantjourney.cloud/plots/optuna-real/wf_oos_equity.png)

More report and chart examples at
[backtester.quantjourney.cloud](https://backtester.quantjourney.cloud).

## Why QuantJourney Backtester

- **Transparent** — every metric is computed locally in readable Python; there is no black box to trust.
- **Reproducible** — runs are fingerprinted over configuration and data, and reports embed metric definitions.
- **Honest by construction** — next-bar execution (no look-ahead), realistic gap/stop/limit fills, and missing bars stay unavailable instead of becoming synthetic 0% returns.
- **Deep analytics** — portfolio returns, risk, drawdowns, rolling statistics, attribution, Monte Carlo, and crisis analysis in one report.
- **Execution-aware** — six order types with slippage, volume participation, commissions, and a full trade blotter.
- **Validated** — rolling, expanding, and anchored walk-forward with purge/embargo, plus grid and Optuna parameter optimization.

## What It Does

The engine supports two core workflows:

- **Weight mode** for portfolio research: generate target weights, apply risk
  overlays and rebalance rules, then let positions drift through time.
- **Order mode** for execution-aware strategies: submit market, limit, stop,
  stop-limit, trailing-stop, bracket, and OCO orders through a deterministic
  fill engine with slippage, volume participation, commissions, and trade
  blotter output.

The accounting path is designed for reproducible research:

- Market data is fetched through `/bt/prepare` and converted into local pandas
  containers.
- Daily and intraday bars are supported through the `granularity` setting.
- Missing market-data gaps remain unavailable assets instead of silent 0%
  return observations.
- Contract multipliers and lot sizes flow through order-mode NAV, trade value,
  position values, weights, and commission notional.
- Rebalance policies support calendar schedules, drift triggers, signal-change
  triggers, circuit breakers, turnover gates, partial rebalance, and tax-aware
  young-lot avoidance.
- Reports write a text summary, JSON/CSV metrics, equity curve CSV/PNG,
  dashboard HTML, selected chart pack, and run metadata.

The runtime package is imported as `backtester`.

## Install

```bash
pip install quantjourney-bt
```

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,data]"
pytest
```

Do not install dependencies into the Homebrew/system Python. Use a virtual
environment; otherwise macOS/Homebrew may raise an
`externally-managed-environment` error and the launcher may miss packages such
as `quantjourney_ti`.

## Repository Layout

```text
backtester/               Runtime package imported as backtester
strategies/               Runnable strategy examples
strategy.sh               Strategy launcher and report runner
benchmarks/               Benchmark-suite notes
skills/                   Strategy-authoring skill materials
tests/                    Import, packaging, and report smoke checks
CHANGELOG.md              Release history
```

The `tests/` directory is intentionally kept. It is not required at runtime, but
it gives the package a quick install/import/report safety check before release.

## Quick Start

For a full catalog of all 35 example strategies — each with a one-line
description, a link to its source, and a link to its results page — see
[strategies/README.md](strategies/README.md).

List available strategies:

```bash
./strategy.sh --list
```

Check one strategy import without credentials or a data call:

```bash
./strategy.sh example_weights_01_sma_daily --check
```

Run repository checks:

```bash
pytest -q
```

Run a real backtest after setting credentials:

```bash
export QJ_API_KEY="..."
./strategy.sh example_weights_01_sma_daily --output /tmp/qj-reports
```

API key auth is preferred for CLI runs. Email/password auth also works; if the
auth service returns an active-session conflict, the launcher retries with
`replace_existing_session=true` by default. Set
`QJ_REPLACE_EXISTING_SESSION=0` if you do not want a CLI run to replace an
existing web session.

## Data Granularity

`Backtester(..., granularity="1d")` remains the default. For yfinance-backed
`/bt/prepare` data you can request historical intraday bars with values such as
`1m`, `5m`, `15m`, `30m`, or `1h`; numeric aliases like `granularity=5` are
normalized to `5m`.

```python
strategy = MyStrategy(
    api_key="...",
    instruments=["AAPL", "MSFT"],
    backtest_period={"start": "2026-06-01", "end": "2026-06-05"},
    source="yfinance",
    granularity="5m",
)
```

Intraday availability depends on yfinance history coverage for the requested
symbols and dates.

## Strategy Skeleton

```python
import asyncio
import os
import pandas as pd

from backtester import Backtester


class MyStrategy(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        fast = close.rolling(20).mean()
        slow = close.rolling(60).mean()
        return ((fast > slow) & fast.notna() & slow.notna()).astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        signals = self.instruments_data.get_feature(
            "strategies", self.strategy_name, "signals"
        )
        active = signals.sum(axis=1).replace(0, pd.NA)
        return signals.div(active, axis=0).fillna(0.0)

    def _compute_positions(self) -> None:
        pass


async def main() -> None:
    strategy = MyStrategy(
        api_key=os.environ["QJ_API_KEY"],
        strategy_name="sma_research",
        instruments=["AAPL", "MSFT", "NVDA"],
        backtest_period={"start": "2024-01-01", "end": "2025-01-01"},
        source="yfinance",
        granularity="1d",
        initial_capital=100_000,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
```

## Reports

By default a strategy run writes outputs under `reports/<strategy_name>/` or
the directory passed to `--output`:

- `summary.txt`
- `summary.json`
- `metrics.csv`
- `equity_curve.csv`
- `equity_curve.png`
- `dashboard.html`
- selected PNG charts under `plots/`
- `run_metadata.json`

Use `--no-reports` when you only want calculation and run metadata:

```bash
./strategy.sh example_weights_01_sma_daily --no-reports --output /tmp/qj-reports
```

## License

Apache License 2.0.
