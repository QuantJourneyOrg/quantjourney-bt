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

For a full catalog of all 45 example strategies — each with a one-line
description, a link to its source, and a link to its results page — see
[strategies/README.md](strategies/README.md) or the summary below.

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

## Strategy Catalog

The repository ships **45 runnable example strategies** — 22 weight-based, 18
order-based, and 5 walk-forward / optimization. Each has source and results-page
links in the [full catalog](strategies/README.md); a summary follows.

**Weight-based (22)** — target-weight portfolios, market-neutral long/short, and risk overlays:

| # | Strategy | Idea |
|:--|:--|:--|
| W01 | Daily SMA Trend | Hold each stock while SMA(50) > SMA(200) |
| W02 | Monthly ETF Trend + Drift | SMA(50/200) trend on ETFs; month-end + 5% drift |
| W03 | Weekly RSI Reversion | Enter RSI(14) < 35, exit RSI > 60 |
| W04 | Quarterly Dual Momentum | Rank ETFs by 12-month return, hold top 2 if positive |
| W05 | Monthly Inverse Volatility | Size each ETF by inverse 63-day volatility |
| W06 | Signal-Change Defensive Rotation | Risk-on ETFs when SPY > SMA(200), else defensive |
| W07 | Intraday RSI 15m | Equal-weight basket when RSI oversold; 15-minute bars |
| W08 | Intraday EMA Scalp 1m | EMA(9/21) trend/cash; 1-minute bars |
| W09 | Intraday SMA Trend 1h | SMA(10/30) trend/cash; hourly bars |
| W10 | Monthly + Circuit Breaker | Monthly trend; flatten on a 15% drawdown + cooldown |
| W11 | Quarterly TE + Cost Gate | Momentum with tracking-error trigger and turnover budget |
| W12 | Daily Partial Drift | Trade only names past a 10% drift band |
| W13 | Pairs Trading (Ratio Z-Score) | Market-neutral KO/PEP on a log-ratio z-score |
| W14 | Pairs Trading (Hedge Ratio) | Market-neutral EWA/EWC on a rolling hedge-ratio spread |
| W15 | Cross-Sectional Momentum (L/S) | Long top-3 / short bottom-3 by 12-month return |
| W16 | Cross-Sectional Reversal (L/S) | Long losers / short winners by 1-month return |
| W17 | Vol-Targeted Trend | SMA trend basket scaled to a 10% volatility target |
| W18 | Vol-Targeted Momentum | Momentum basket scaled to a 15% volatility target |
| W19 | Risk Parity (Multi-Asset ERC) | Equal risk contribution across a multi-asset basket |
| W20 | Risk Parity + Position Cap | Sector ERC chained with a 25% per-position cap |
| W21 | Bollinger Band Reversion | Buy below the lower band, exit at the midline |
| W22 | MACD Trend | Long while MACD is above its signal line |

**Order-based (18)** — explicit orders through the fill engine (slippage, commissions, blotter):

| # | Strategy | Order type | Idea |
|:--|:--|:--|:--|
| O01 | Market SMA Crossover | Market | Buy SMA(20)>SMA(50), sell on reverse |
| O02 | Market RSI Reversion | Market | Buy RSI(14) < 35, sell RSI > 60 |
| O03 | Limit RSI Dip Buyer | Limit | Passive buy-limit below the close on weak RSI |
| O04 | Limit Trend Pullback | Limit | In an uptrend, wait for a 1% pullback to enter |
| O05 | Stop Breakout Entry | Stop | Buy-stop above the recent 20-day high |
| O06 | Protective Stop Loss | Market + Stop | Trend entry with a 5% protective stop |
| O07 | Stop-Limit Breakout | Stop-Limit | Enter breakouts but cap the maximum fill price |
| O08 | Stop-Limit Protection | Market + Stop-Limit | Trend entry, downside protected by a stop-limit sell |
| O09 | Trailing Stop Trend | Trailing Stop | Trend entry, 4% trailing stop exit |
| O10 | RSI + Trailing Stop | Trailing Stop | Oversold RSI entry, 5% trailing stop |
| O11 | Trailing Stop-Limit | Trailing Stop-Limit | Trailing stop that converts to a limit |
| O12 | Bracket Trend | Bracket | Trend entry with a +6% / −3% bracket |
| O13 | Bracket RSI Reversion | Bracket | RSI dip with a +4% / −2% bracket |
| O14 | OCO Dip or Breakout | OCO | Competing buy-limit (dip) and buy-stop (breakout) |
| O15 | Intraday 5m Bracket Reversion | Bracket | RSI dips with a tight bracket; 5-min bars |
| O16 | Intraday 30m Stop Breakout | Stop | Buy-stop above the 12-bar high; 30-min bars |
| O17 | Monthly Rotation (orders) | Market | Event-driven monthly momentum rotation via orders |
| O18 | Signal-Change Rotation (orders) | Market | Trade only on SMA trend-signal flips |

**Walk-forward & optimization (5)** — prove a strategy generalizes:

| # | Example | Idea |
|:--|:--|:--|
| WF01 | Rolling Walk-Forward | Sliding fixed-length train/test windows with purge/embargo |
| WF02 | Expanding Walk-Forward | Ever-growing training window vs sliding test window |
| WF03 | Anchored + Purge/Embargo | How purge and embargo gaps prevent train/test leakage |
| WF04 | Grid Search | Exhaustive SMA fast/slow tuning scored by real backtests |
| WF05 | Optuna TPE + Walk-Forward | Bayesian parameter search, then out-of-sample validation |

Long/short examples (W13–W16) are market-neutral; short borrow/financing is not
modeled (a documented research approximation).

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
