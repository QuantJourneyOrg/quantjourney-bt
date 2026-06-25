# QuantJourney Backtester Public Light

Public/light release surface for the QuantJourney backtester.

This repository contains a runnable light backtester package, 20 public strategy examples, and supporting comparison materials that can be linked from QuantJourney docs and the Compare page.

The PyPI package name is `quantjourney-bt`. The runtime package follows the current strategy API and is imported as `backtester`.

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

## Repository Layout

```text
backtester/               Public/light backtester package
strategies/               20 runnable public strategy examples
strategy.sh               Strategy launcher
benchmarks/               Benchmark-suite notes
compare/                  Cross-engine comparison notes
docs/                     Public scope and release notes
skills/                   Public strategy-authoring skill
tests/                    Public repository checks
```

## Included Strategies

The public strategy suite contains 20 files:

- `example_orders_01_market_sma_cross.py`
- `example_orders_02_market_rsi_reversion.py`
- `example_orders_03_limit_rsi_dip.py`
- `example_orders_04_limit_trend_pullback.py`
- `example_orders_05_stop_breakout_entry.py`
- `example_orders_06_stop_loss_protection.py`
- `example_orders_07_stop_limit_breakout.py`
- `example_orders_08_stop_limit_protection.py`
- `example_orders_09_trailing_stop_trend.py`
- `example_orders_10_trailing_stop_rsi.py`
- `example_orders_11_trailing_stop_limit.py`
- `example_orders_12_bracket_trend.py`
- `example_orders_13_bracket_rsi_reversion.py`
- `example_orders_14_oco_dip_or_breakout.py`
- `example_weights_01_sma_daily.py`
- `example_weights_02_monthly_drift_etf.py`
- `example_weights_03_weekly_rsi_reversion.py`
- `example_weights_04_quarterly_dual_momentum.py`
- `example_weights_05_monthly_inverse_vol.py`
- `example_weights_06_signal_change_defensive.py`

## Check That It Works

List strategies:

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

API key auth is preferred for CLI runs. Email/password auth also works; if the auth service returns an active-session conflict, the launcher retries with `replace_existing_session=true` by default. Set `QJ_REPLACE_EXISTING_SESSION=0` if you do not want a CLI run to replace an existing web session.

The real backtest path fetches market data through QuantJourney credentials. The public/light report writes a text summary, `summary.json`, `metrics.csv`, `equity_curve.csv`, `equity_curve.png`, `dashboard.html`, a selected native QuantJourney chart pack under `plots/`, and `run_metadata.json`. Full `portfolio_data.pkl`, `instruments_data.pkl`, and `blotter.pkl` debug archives are disabled by default; set `QJ_SAVE_PICKLE_ARCHIVE=1` only when you explicitly want local audit pickles. Full PDF packets, narrative generation, walk-forward validation, optimization and deeper institutional diagnostics are QuantJourney Backtester Pro/SaaS features.

Use `--no-reports` when you only want calculation and archive output:

```bash
./strategy.sh example_weights_01_sma_daily --no-reports --output /tmp/qj-reports
```

The `--check` path verifies the public package and strategy code without contacting the API.

Do not install dependencies into the Homebrew/system Python. Use the local `.venv`; otherwise macOS/Homebrew may raise an `externally-managed-environment` error and the launcher may miss packages such as `quantjourney_ti`.

## Public Scope

Included:

- Portfolio weight-mode examples.
- Order-mode examples for market, limit, stop, stop-limit, trailing stop, bracket and OCO behavior.
- Strategy launcher and output directory support.
- Public/light engine modules required by those examples.
- Public report artifacts built from the native QuantJourney metric and plotting modules: text summary, JSON/CSV metrics, equity CSV, dashboard HTML, equity PNG, selected PNG chart pack and run metadata.

Excluded:

- Full PDF factsheets and institutional report packets.
- Pickle archives by default; `portfolio_data.pkl`, `instruments_data.pkl` and `blotter.pkl` are opt-in local debug artifacts.
- Pro-only diagnostics: full plot orchestration, crisis analysis, trace plots, blotter plots and narrative generation.
- Walk-forward validation and optimization.
- Private deployment scripts, credentials and infrastructure files.
- Private research orchestration code.
- Internal-only report publication workflows.

## License

Apache License 2.0.
