# QuantJourney Backtester

A Python-native backtesting engine for reproducible portfolio research.

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/quantjourney-bt?color=orange)](https://pypi.org/project/quantjourney-bt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)]()
[![API](https://img.shields.io/badge/API-QuantJourney%20Cloud-1B4F72)](https://quantjourney.cloud)
[![Changelog](https://img.shields.io/badge/Changelog-backtester.quantjourney.cloud-111827)](https://backtester.quantjourney.cloud/changelog)

QuantJourney Backtester turns strategy ideas into auditable research packets:
signals become target weights or explicit orders, orders become simulated fills,
fills update cash and positions, and NAV is reconstructed from portfolio state.

It is designed for researchers who need more than an equity curve: execution
assumptions, costs, slippage, rebalancing rules, crisis behavior, walk-forward
validation, optimization diagnostics, metrics, plots, and run metadata from
one repeatable run.

## Why It Exists

Most backtests stop at `signal x returns`. That is fast, but it hides the
questions that matter before a strategy can be trusted:

- Was there look-ahead?
- What happened to missing bars?
- How were weights converted into trades?
- Did costs and turnover destroy the edge?
- Did parameters generalize out of sample?
- Which crisis regimes broke the strategy?
- Can the run be reproduced and reviewed later?

QuantJourney Backtester makes these assumptions explicit.

## Two Research Modes

**Weight mode** is for portfolio research: factor portfolios, rotation models,
long/cash strategies, long/short books, risk overlays, volatility targeting,
and scheduled rebalancing.

**Order mode** is for execution-aware research: market, limit, stop, stop-limit,
trailing stop, bracket, and OCO orders with commissions, slippage, volume
participation, fills, positions, cash, NAV, and trade blotters.

## Engine Contract

```text
Data -> Features -> Signals -> Target Weights / Orders -> Fills -> Positions -> NAV -> Metrics -> Report Packet
```

Each stage is explicit. Data is transformed into features, features drive
signals, signals become either target weights or orders, execution assumptions
turn those decisions into fills, and portfolio state is used to reconstruct NAV,
metrics, plots, and run metadata.

### What you want to do -> what to use

| I want to... | Use |
|---|---|
| Generate long / flat / short or ranking intent | `_compute_signals()` |
| Convert intent into target portfolio exposure | `_compute_weights()` |
| Apply caps, vol targeting, inverse vol, or risk parity | `risk_model=...` |
| Trade only on calendar, drift, signal, or turnover triggers | `RebalancePolicy(...)` |
| Submit market / limit / stop / trailing / bracket / OCO orders | `execution_mode="orders"` + `_compute_orders(...)` |
| Model spread, impact, and commission assumptions | slippage & commission models |
| Validate parameters out of sample | walk-forward / Optuna |

The repository examples and tests demonstrate these engine semantics locally.

## What You Get From One Run

Each local run can produce metrics, plots, equity curves, drawdowns, rolling
risk, optimization evidence, walk-forward results, CSV/JSON artifacts, a static
HTML dashboard, and run metadata. The hosted platform adds crisis diagnostics,
interactive dashboards, execution traces, and PDF tear sheets.

## What Stays Local

Your strategy code, signals, portfolio accounting, order simulation, metrics,
plots, generated reports, and run artifacts stay local. QuantJourney Cloud is
used for market-data preparation and authentication.

## What It Is Not

This is not a broker, not a live trading system, not investment advice, and not
a guarantee that a strategy will work out of sample. Some examples intentionally
simplify assumptions such as borrow cost, financing, liquidity and market
impact. Those assumptions are documented so they can be changed, not hidden.

## Example Output

Every run produces a review-ready research packet — equity curves, monthly
returns heatmaps, drawdowns, risk and rolling statistics, and walk-forward /
optimization summaries (see [Reports](#reports) for the exact file list).
A few examples:

**Cumulative returns vs benchmark**

![Cumulative returns vs benchmark](https://backtester.quantjourney.cloud/plots/cumulative_returns.png)

**Monthly returns heatmap**

![Monthly returns heatmap](https://backtester.quantjourney.cloud/plots/monthly_returns_heatmap.png)

**Crisis analysis across historical stress periods** *(hosted platform report pack)*

![Crisis analysis](https://backtester.quantjourney.cloud/plots/crisis_summary.png)

**Walk-forward out-of-sample equity** *(hosted platform report pack)*

![Walk-forward out-of-sample equity](https://backtester.quantjourney.cloud/plots/optuna-real/wf_oos_equity.png)

The first two charts come from the open-source report pack in this repository.
Charts marked *hosted platform report pack* — crisis analysis, trade blotters,
execution traces, PDF factsheets, and interactive dashboards — are generated by
the hosted [QuantJourney platform](https://backtester.quantjourney.cloud) on
top of the same engine results. More examples at
[backtester.quantjourney.cloud](https://backtester.quantjourney.cloud).

## Install

```bash
pip install quantjourney-bt
```

The wheel installs the `backtester` library. The runnable strategy catalog and
`strategy.sh` launcher are repository assets; clone this repository when you
want to run or modify the examples below.

Optional extras: `pip install "quantjourney-bt[wf]"` adds Optuna for the
walk-forward optimization examples (WF05); `[data]` adds the yfinance
benchmark fallback.

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

## Reproducible Demo Without API Key

Run the first strategy against deterministic bundled sample data:

```bash
./strategy.sh example_weights_01_sma_daily --sample-data --output /tmp/qj-sample
```

The sample dataset is intentionally small and reproducible. It is useful for
install checks, report generation, and reading the engine flow without creating
an account. For real market data, set QuantJourney API credentials and run the
same strategy without `--sample-data`.

## Repository Layout

```text
backtester/               Runtime package imported as backtester
strategies/               Runnable strategy examples
strategy.sh               Strategy launcher and report runner
benchmarks/               Benchmark-suite notes
skills/                   Strategy-authoring skill materials
tests/                    Import, packaging, and report smoke checks
docs/                     Roadmap and supporting documentation
CHANGELOG.md              Release history
```

The public runtime includes the shared execution simulator, contract-aware
portfolio ledger, portfolio-of-strategies book, and pre-trade risk controls.
Hosted data, orchestration, and extended report packs remain outside this
repository; see [Public Scope](docs/public_scope.md).

The `tests/` directory is intentionally kept. It is not required at runtime, but
it gives the package a quick install/import/report safety check before release.

## Documentation

- [Roadmap](docs/ROADMAP.md) - direction of travel by theme, without delivery
  dates or ordering commitments.
- [Strategy catalog](strategies/README.md) - runnable examples with source and
  result links.
- [Contributing](CONTRIBUTING.md) - how to add example strategies, fixes, and
  docs (fork, branch, pull request).
- [Release process](docs/release.md) - clean-tag publishing and exact artifact
  boundary checks.

## AI Co-Pilot Skills

The `skills/` directory holds guidance packs for AI-assisted research. When you
work with an AI coding assistant, point it at the relevant `SKILL.md` so it
follows the engine's conventions instead of guessing:

| Skill | Use it to |
|---|---|
| [`qj-strategy-ideas`](skills/qj-strategy-ideas/SKILL.md) | Turn an idea into a runnable strategy — weights vs orders, the nearest example, the two-method pattern. |
| [`qj-strategy-author`](skills/qj-strategy-author/SKILL.md) | Write a clean, focused example strategy. |
| [`qj-strategy-reviewer`](skills/qj-strategy-reviewer/SKILL.md) | Review a strategy for look-ahead, exposure, cost realism, and mode fit. |
| [`qj-report-analyst`](skills/qj-report-analyst/SKILL.md) | Read a report and its plots and judge whether the result is trustworthy. |
| [`qj-config-helper`](skills/qj-config-helper/SKILL.md) | Configure the engine — parameters, rebalance policy, risk overlays, granularity. |

For example: when writing a new strategy, open
`skills/qj-strategy-ideas/SKILL.md` and follow the pattern; when reviewing one,
use `skills/qj-strategy-reviewer/SKILL.md`; to make sense of the output, use
`skills/qj-report-analyst/SKILL.md`.

## Quick Start

For a full catalog of all 50 example strategies — each with a one-line
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

Run a deterministic demo without API credentials:

```bash
./strategy.sh example_weights_01_sma_daily --sample-data --output /tmp/qj-sample
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

The repository ships **50 runnable example strategies** — 25 weight-based, 20
order-based, and 5 walk-forward / optimization. Each has source and results-page
links in the [full catalog](strategies/README.md); a summary follows.

**Weight-based (25)** — target-weight portfolios, market-neutral long/short, and risk overlays:

| # | Strategy | Idea | Code | Results |
|:--|:--|:--|:--|:--|
| W01 | Daily SMA Trend | Hold each stock while SMA(50) > SMA(200); daily rebalance | [source](strategies/example_weights_01_sma_daily.py) | [view](https://backtester.quantjourney.cloud/strategies/daily-sma-trend) |
| W02 | Monthly ETF Trend + Drift | SMA(50/200) trend on ETFs; month-end + 5% drift band | [source](strategies/example_weights_02_monthly_drift_etf.py) | [view](https://backtester.quantjourney.cloud/strategies/monthly-drift-etf) |
| W03 | Weekly RSI Reversion | Enter RSI(14) < 35, exit RSI > 60; weekly (Fri) | [source](strategies/example_weights_03_weekly_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/weekly-rsi-reversion) |
| W04 | Quarterly Dual Momentum | Rank ETFs by 12-month return, hold top 2 if positive; quarter-end | [source](strategies/example_weights_04_quarterly_dual_momentum.py) | [view](https://backtester.quantjourney.cloud/strategies/quarterly-dual-momentum) |
| W05 | Monthly Inverse Volatility | Size each ETF by inverse 63-day volatility; month-end | [source](strategies/example_weights_05_monthly_inverse_vol.py) | [view](https://backtester.quantjourney.cloud/strategies/monthly-inverse-vol) |
| W06 | Signal-Change Defensive Rotation | SPY > SMA(200) -> risk-on ETFs, else defensive; on signal change | [source](strategies/example_weights_06_signal_change_defensive.py) | [view](https://backtester.quantjourney.cloud/strategies/signal-change-defensive) |
| W07 | Intraday RSI 15m | Equal-weight basket when RSI oversold; 15-minute bars | [source](strategies/example_weights_07_intraday_rsi_15m.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W08 | Intraday EMA Scalp 1m | EMA(9/21) trend/cash; 1-minute bars | [source](strategies/example_weights_08_intraday_1m_ema_scalp.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W09 | Intraday SMA Trend 1h | SMA(10/30) trend/cash; hourly bars | [source](strategies/example_weights_09_intraday_1h_sma_trend.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W10 | Monthly + Circuit Breaker | Monthly ETF trend; flatten on a 15% drawdown + cooldown | [source](strategies/example_weights_10_monthly_circuit_breaker.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W11 | Quarterly TE + Cost Gate | Momentum with tracking-error trigger and turnover budget | [source](strategies/example_weights_11_quarterly_te_cost_gate.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W12 | Daily Partial Drift | Momentum tilt; trade only names past a 10% drift band | [source](strategies/example_weights_12_daily_partial_drift.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W13 | Pairs Trading (Ratio Z-Score) | Market-neutral KO/PEP on a log-ratio z-score | [source](strategies/example_weights_13_pairs_ratio_zscore.py) | [view](https://backtester.quantjourney.cloud/strategies/pairs-trading) |
| W14 | Pairs Trading (Hedge Ratio) | Market-neutral EWA/EWC on a rolling OLS hedge-ratio spread | [source](strategies/example_weights_14_pairs_hedge_ratio.py) | [view](https://backtester.quantjourney.cloud/strategies/pairs-trading) |
| W15 | Cross-Sectional Momentum (L/S) | Long top-3 / short bottom-3 by 12-month return; monthly | [source](strategies/example_weights_15_cross_sectional_momentum.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W16 | Cross-Sectional Reversal (L/S) | Long losers / short winners by 1-month return; weekly | [source](strategies/example_weights_16_cross_sectional_reversal.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W17 | Vol-Targeted Trend | SMA trend basket scaled to a 10% volatility target | [source](strategies/example_weights_17_vol_target_trend.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W18 | Vol-Targeted Momentum | Momentum basket scaled to a 15% volatility target | [source](strategies/example_weights_18_vol_target_momentum.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W19 | Risk Parity (Multi-Asset ERC) | Equal risk contribution across a multi-asset basket | [source](strategies/example_weights_19_risk_parity_multiasset.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W20 | Risk Parity + Position Cap | Sector ERC chained with a 25% per-position cap | [source](strategies/example_weights_20_risk_parity_capped.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W21 | Bollinger Band Reversion | Buy below the lower band, exit at the midline | [source](strategies/example_weights_21_bollinger_reversion.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W22 | MACD Trend | Long while MACD is above its signal line | [source](strategies/example_weights_22_macd_trend.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W23 | FX Time-Series Momentum | Six-month trend across USD-quoted spot pairs; inverse-vol weights | [source](strategies/example_weights_23_fx_time_series_momentum.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W24 | FX Cross-Sectional Momentum | Long strongest / short weakest XXX/USD pair; monthly | [source](strategies/example_weights_24_fx_cross_sectional_momentum.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W25 | Continuous Futures Trend Proxy | Diversified long/short trend on provider continuous series | [source](strategies/example_weights_25_continuous_futures_trend.py) | [browse](https://backtester.quantjourney.cloud/strategies) |

W23-W25 are price-return research proxies. They do not apply FX lots or futures
multipliers to PnL and do not model financing, margin, or controlled futures
rolls.

**Order-based (20)** — explicit orders through the fill engine (slippage, commissions, blotter):

| # | Strategy | Order type | Idea | Code | Results |
|:--|:--|:--|:--|:--|:--|
| O01 | Market SMA Crossover | Market | Buy SMA(20) crossing above SMA(50), sell on reverse | [source](strategies/example_orders_01_market_sma_cross.py) | [view](https://backtester.quantjourney.cloud/strategies/market-sma-cross) |
| O02 | Market RSI Reversion | Market | Buy RSI(14) < 35, sell RSI > 60 | [source](strategies/example_orders_02_market_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/market-rsi-reversion) |
| O03 | Limit RSI Dip Buyer | Limit | Passive buy-limit below the close on weak RSI | [source](strategies/example_orders_03_limit_rsi_dip.py) | [view](https://backtester.quantjourney.cloud/strategies/limit-rsi-dip) |
| O04 | Limit Trend Pullback | Limit | In an uptrend, wait for a 1% pullback to enter | [source](strategies/example_orders_04_limit_trend_pullback.py) | [view](https://backtester.quantjourney.cloud/strategies/limit-trend-pullback) |
| O05 | Stop Breakout Entry | Stop | Buy-stop above the recent 20-day high | [source](strategies/example_orders_05_stop_breakout_entry.py) | [view](https://backtester.quantjourney.cloud/strategies/stop-breakout-entry) |
| O06 | Protective Stop Loss | Market + Stop | Trend entry with a 5% protective stop | [source](strategies/example_orders_06_stop_loss_protection.py) | [view](https://backtester.quantjourney.cloud/strategies/protective-stop-loss) |
| O07 | Stop-Limit Breakout | Stop-Limit | Enter breakouts but cap the maximum fill price | [source](strategies/example_orders_07_stop_limit_breakout.py) | [view](https://backtester.quantjourney.cloud/strategies/stop-limit-breakout) |
| O08 | Stop-Limit Protection | Market + Stop-Limit | Trend entry, downside protected by a stop-limit sell | [source](strategies/example_orders_08_stop_limit_protection.py) | [view](https://backtester.quantjourney.cloud/strategies/stop-limit-protection) |
| O09 | Trailing Stop Trend | Trailing Stop | Trend entry, 4% trailing stop manages the exit | [source](strategies/example_orders_09_trailing_stop_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/trailing-stop-trend) |
| O10 | RSI + Trailing Stop | Trailing Stop | Oversold RSI entry, 5% trailing stop for risk | [source](strategies/example_orders_10_trailing_stop_rsi.py) | [view](https://backtester.quantjourney.cloud/strategies/trailing-stop-rsi) |
| O11 | Trailing Stop-Limit | Trailing Stop-Limit | Trailing stop that converts to a limit on trigger | [source](strategies/example_orders_11_trailing_stop_limit.py) | [view](https://backtester.quantjourney.cloud/strategies/trailing-stop-limit) |
| O12 | Bracket Trend | Bracket | Trend entry with a +6% / -3% bracket | [source](strategies/example_orders_12_bracket_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/bracket-trend) |
| O13 | Bracket RSI Reversion | Bracket | RSI dip with a +4% / -2% bracket | [source](strategies/example_orders_13_bracket_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/bracket-rsi-reversion) |
| O14 | OCO Dip or Breakout | OCO | Competing buy-limit (dip) and buy-stop (breakout) | [source](strategies/example_orders_14_oco_dip_or_breakout.py) | [view](https://backtester.quantjourney.cloud/strategies/oco-dip-or-breakout) |
| O15 | Intraday 5m Bracket Reversion | Bracket | Oversold-RSI dips with a tight +0.6% / -0.4% bracket; 5-min bars | [source](strategies/example_orders_15_intraday_5m_bracket_reversion.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O16 | Intraday 30m Stop Breakout | Stop | Buy-stop above the 12-bar high, fixed holding period; 30-min bars | [source](strategies/example_orders_16_intraday_30m_stop_breakout.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O17 | Monthly Rotation (orders) | Market | Event-driven monthly momentum rotation, executed with orders | [source](strategies/example_orders_17_monthly_rotation_orders.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O18 | Signal-Change Rotation (orders) | Market | Trade only on SMA trend-signal flips (no calendar) | [source](strategies/example_orders_18_signal_change_rotation_orders.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O19 | FX Momentum with Standard Lots | Market | Contract-aware whole-lot momentum on USD-quoted spot pairs | [source](strategies/example_orders_19_fx_momentum_lots.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O20 | Futures Donchian Contracts | Market | Whole-contract ATR sizing on provider continuous futures | [source](strategies/example_orders_20_futures_donchian_contracts.py) | [browse](https://backtester.quantjourney.cloud/strategies) |

O19-O20 consume provider contract specifications. Multipliers and lot sizes are
applied. Cross-currency conversion, FX swaps, dated-futures selection, and
controlled roll execution remain out of scope; unsupported cross-currency FX
accounting is rejected rather than approximated.

**Walk-forward & optimization (5)** — prove a strategy generalizes:

| # | Example | Idea | Code | Results |
|:--|:--|:--|:--|:--|
| WF01 | Rolling Walk-Forward | Sliding fixed-length train/test windows with purge/embargo | [source](strategies/example_wf_01_rolling_walkforward.py) | [view](https://backtester.quantjourney.cloud/strategies/walkforward-case-study) |
| WF02 | Expanding Walk-Forward | Ever-growing training window vs sliding test window | [source](strategies/example_wf_02_expanding_walkforward.py) | [view](https://backtester.quantjourney.cloud/strategies/walkforward-case-study) |
| WF03 | Anchored + Purge/Embargo | How purge and embargo gaps prevent train/test leakage | [source](strategies/example_wf_03_anchored_purge_embargo.py) | [view](https://backtester.quantjourney.cloud/strategies/walkforward-case-study) |
| WF04 | Grid Search | Exhaustive SMA fast/slow tuning scored by real backtests | [source](strategies/example_wf_04_grid_search_optimization.py) | [view](https://backtester.quantjourney.cloud/strategies/optuna-optimization) |
| WF05 | Optuna TPE + Walk-Forward | Bayesian parameter search, then out-of-sample validation | [source](strategies/example_wf_05_optuna_tpe_optimization.py) | [view](https://backtester.quantjourney.cloud/strategies/optuna-optimization) |

WF01-WF03 default to `slice_diagnostics`: train/test metrics are computed from
one full-period NAV so examples run quickly and remain easy to inspect. To run
a fuller per-fold strategy re-run/refit, enable:

```bash
QJ_WF_MODE=per_fold_refit ./strategy.sh example_wf_01_rolling_walkforward
```

The logs and walk-forward summary report the active mode. Per-fold refit is
slower because each fold runs the strategy again through the data/preparation
pipeline. WF04-WF05 are optimization workflows and should be read as selection
diagnostics rather than a simple winner-takes-all backtest.

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

## Assumptions & Limitations

- Example strategies are research templates, not production trading systems.
- Long/short examples do not model borrow fees, stock-loan availability,
  financing, or margin interest by default.
- Commissions, slippage, volume participation, and market impact are model
  assumptions. Treat them as part of the research contract.
- The deterministic sample dataset is illustrative. It is not historical market
  data and should not be used to judge strategy quality.
- Historical intraday availability depends on the upstream provider and the
  requested symbols, dates, and granularity.
- Walk-forward and optimization diagnostics help expose overfit risk, but a
  good in-sample or single out-of-sample result is not proof of robustness.
  Check whether a run used `slice_diagnostics` or `per_fold_refit`.

The runtime package is imported as `backtester`.

## License

Apache License 2.0.
