# Strategy Examples

This folder contains **50 runnable example strategies** for the QuantJourney
Backtester — **25 weight-based**, **20 order-based**, and **5 walk-forward /
optimization** examples. Each file is a complete, self-contained template: copy
the one closest to your idea, change the rule, and you are testing your own
strategy in minutes.

Every strategy links to its **source** (the code) and, where published, to its
**results page** on [backtester.quantjourney.cloud](https://backtester.quantjourney.cloud/strategies)
with its actual setup, metrics, logs and plot pack. The repository also keeps a
compact set of [45 generated result previews](../docs/strategy-results); the
five WF examples publish workflow diagnostics rather than a borrowed portfolio curve.

## Run one

```bash
# import-only check (no credentials, no data call)
./strategy.sh example_weights_01_sma_daily --check

# real backtest (after setting credentials)
export QJ_API_KEY="..."
./strategy.sh example_weights_01_sma_daily --output /tmp/qj-reports
```

Run the complete catalog sequentially with the same launcher:

```bash
./strategy.sh --all --output ./reports
```

Batch logs and a final `summary.tsv` are written below
`reports/_batch/<timestamp>/`; one failed strategy does not stop the remaining
runs.

Naming: `example_<mode>_<NN>_<name>.py`, where mode is `weights`, `orders`, or
`wf` (walk-forward / optimization).

---

## Weight-based strategies (25)

Portfolio thinking — produce target weights, let the rebalance engine (and any
risk overlay) trade them. Includes long/cash, market-neutral long/short, and
risk-overlay templates.

| # | Strategy | Idea | Code | Results |
|:--|:--|:--|:--|:--|
| W01 | Daily SMA Trend | Hold each sector ETF while SMA(50) > SMA(200); daily rebalance | [source](./example_weights_01_sma_daily.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-01-sma-daily) |
| W02 | Monthly ETF Trend + Drift | SMA(50/200) trend on ETFs; month-end + 5% drift band | [source](./example_weights_02_monthly_drift_etf.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-02-monthly-drift-etf) |
| W03 | Weekly RSI Reversion | Enter RSI(14) < 35, exit RSI > 60; weekly (Fri) | [source](./example_weights_03_weekly_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-03-weekly-rsi-reversion) |
| W04 | Quarterly Dual Momentum | Rank ETFs by 12-month return, hold top 2 if positive; quarter-end | [source](./example_weights_04_quarterly_dual_momentum.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-04-quarterly-dual-momentum) |
| W05 | Monthly Inverse Volatility | Size each ETF by inverse 63-day volatility; month-end | [source](./example_weights_05_monthly_inverse_vol.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-05-monthly-inverse-vol) |
| W06 | Signal-Change Defensive Rotation | SPY > SMA(200) → risk-on ETFs, else defensive; on signal change | [source](./example_weights_06_signal_change_defensive.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-06-signal-change-defensive) |
| W07 | Intraday RSI 15m | Equal-weight basket when RSI oversold; 15-minute bars | [source](./example_weights_07_intraday_rsi_15m.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-07-intraday-rsi-15m) |
| W08 | Intraday EMA Scalp 1m | EMA(9/21) trend/cash; 1-minute bars | [source](./example_weights_08_intraday_1m_ema_scalp.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-08-intraday-1m-ema-scalp) |
| W09 | Intraday SMA Trend 1h | SMA(10/30) trend/cash; hourly bars | [source](./example_weights_09_intraday_1h_sma_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-09-intraday-1h-sma-trend) |
| W10 | Monthly + Circuit Breaker | Monthly ETF trend; flatten on a 15% drawdown + cooldown | [source](./example_weights_10_monthly_circuit_breaker.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-10-monthly-circuit-breaker) |
| W11 | Quarterly TE + Cost Gate | Momentum with tracking-error trigger and turnover budget | [source](./example_weights_11_quarterly_te_cost_gate.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-11-quarterly-te-cost-gate) |
| W12 | Daily Partial Drift | Momentum tilt; trade only names past a 10% drift band | [source](./example_weights_12_daily_partial_drift.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-12-daily-partial-drift) |
| W13 | Pairs Trading (Ratio Z-Score) | Market-neutral KO/PEP on a log-ratio z-score | [source](./example_weights_13_pairs_ratio_zscore.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-13-pairs-ratio-zscore) |
| W14 | Pairs Trading (Hedge Ratio) | Market-neutral EWA/EWC on a rolling OLS hedge-ratio spread | [source](./example_weights_14_pairs_hedge_ratio.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-14-pairs-hedge-ratio) |
| W15 | Cross-Sectional Momentum (L/S) | Long top-3 / short bottom-3 by 12-month return; monthly | [source](./example_weights_15_cross_sectional_momentum.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-15-cross-sectional-momentum) |
| W16 | Cross-Sectional Reversal (L/S) | Long losers / short winners by 1-month return; weekly | [source](./example_weights_16_cross_sectional_reversal.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-16-cross-sectional-reversal) |
| W17 | Vol-Targeted Trend | SMA trend basket scaled to a 10% volatility target | [source](./example_weights_17_vol_target_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-17-vol-target-trend) |
| W18 | Vol-Targeted Momentum | Momentum basket scaled to a 15% volatility target | [source](./example_weights_18_vol_target_momentum.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-18-vol-target-momentum) |
| W19 | Risk Parity (Multi-Asset ERC) | Equal risk contribution across a multi-asset basket | [source](./example_weights_19_risk_parity_multiasset.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-19-risk-parity-multiasset) |
| W20 | Risk Parity + Position Cap | Sector ERC chained with a 25% per-position cap | [source](./example_weights_20_risk_parity_capped.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-20-risk-parity-capped) |
| W21 | Bollinger Band Reversion | Buy below the lower band, exit at the midline | [source](./example_weights_21_bollinger_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-21-bollinger-reversion) |
| W22 | MACD Trend | Long while MACD is above its signal line | [source](./example_weights_22_macd_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-22-macd-trend) |
| W23 | FX Time-Series Momentum | Six-month trend across USD-quoted spot pairs; inverse-vol weights | [source](./example_weights_23_fx_time_series_momentum.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-23-fx-time-series-momentum) |
| W24 | FX Cross-Sectional Momentum | Long strongest / short weakest XXX/USD pair; monthly | [source](./example_weights_24_fx_cross_sectional_momentum.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-24-fx-cross-sectional-momentum) |
| W25 | Continuous Futures Trend Proxy | Diversified long/short trend on provider continuous series | [source](./example_weights_25_continuous_futures_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/example-weights-25-continuous-futures-trend) |

The long/short examples (W13–W16) are market-neutral; short borrow/financing is
not modeled (a documented research approximation).

W23-W25 are price-return research proxies. They do not apply FX lots or futures
multipliers to PnL and do not model financing, margin, or controlled futures
rolls.

## Order-based strategies (20)

Execution thinking — submit explicit orders through the fill engine with
slippage, commissions, and a trade blotter.

| # | Strategy | Order type | Idea | Code | Results |
|:--|:--|:--|:--|:--|:--|
| O01 | Market SMA Crossover | Market | Buy SMA(20) crossing above SMA(50), sell on reverse | [source](./example_orders_01_market_sma_cross.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-01-market-sma-cross) |
| O02 | Market RSI Reversion | Market | Buy RSI(14) < 35, sell RSI > 60 | [source](./example_orders_02_market_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-02-market-rsi-reversion) |
| O03 | Limit RSI Dip Buyer | Limit | Passive buy-limit below the close on weak RSI | [source](./example_orders_03_limit_rsi_dip.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-03-limit-rsi-dip) |
| O04 | Limit Trend Pullback | Limit | In an uptrend, wait for a 1% pullback to enter | [source](./example_orders_04_limit_trend_pullback.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-04-limit-trend-pullback) |
| O05 | Stop Breakout Entry | Stop | Buy-stop above the recent 20-day high | [source](./example_orders_05_stop_breakout_entry.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-05-stop-breakout-entry) |
| O06 | Protective Stop Loss | Market + Stop | Trend entry with a 5% protective stop | [source](./example_orders_06_stop_loss_protection.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-06-stop-loss-protection) |
| O07 | Stop-Limit Breakout | Stop-Limit | Enter breakouts but cap the maximum fill price | [source](./example_orders_07_stop_limit_breakout.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-07-stop-limit-breakout) |
| O08 | Stop-Limit Protection | Market + Stop-Limit | Trend entry, downside protected by a stop-limit sell | [source](./example_orders_08_stop_limit_protection.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-08-stop-limit-protection) |
| O09 | Trailing Stop Trend | Trailing Stop | Trend entry, 4% trailing stop manages the exit | [source](./example_orders_09_trailing_stop_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-09-trailing-stop-trend) |
| O10 | RSI + Trailing Stop | Trailing Stop | Oversold RSI entry, 5% trailing stop for risk | [source](./example_orders_10_trailing_stop_rsi.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-10-trailing-stop-rsi) |
| O11 | Trailing Stop-Limit | Trailing Stop-Limit | Trailing stop that converts to a limit on trigger | [source](./example_orders_11_trailing_stop_limit.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-11-trailing-stop-limit) |
| O12 | Bracket Trend | Bracket | Trend entry with a +6% / −3% bracket | [source](./example_orders_12_bracket_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-12-bracket-trend) |
| O13 | Bracket RSI Reversion | Bracket | RSI dip with a +4% / −2% bracket | [source](./example_orders_13_bracket_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-13-bracket-rsi-reversion) |
| O14 | OCO Dip or Breakout | OCO | Competing buy-limit (dip) and buy-stop (breakout) | [source](./example_orders_14_oco_dip_or_breakout.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-14-oco-dip-or-breakout) |
| O15 | Intraday 5m Bracket Reversion | Bracket | Oversold-RSI dips with a tight +0.6% / −0.4% bracket; 5-min bars | [source](./example_orders_15_intraday_5m_bracket_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-15-intraday-5m-bracket-reversion) |
| O16 | Intraday 30m Stop Breakout | Stop | Buy-stop above the 12-bar high, fixed holding period; 30-min bars | [source](./example_orders_16_intraday_30m_stop_breakout.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-16-intraday-30m-stop-breakout) |
| O17 | Monthly Rotation (orders) | Market | Event-driven monthly momentum rotation, executed with orders | [source](./example_orders_17_monthly_rotation_orders.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-17-monthly-rotation-orders) |
| O18 | Signal-Change Rotation (orders) | Market | Trade only on SMA trend-signal flips (no calendar) | [source](./example_orders_18_signal_change_rotation_orders.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-18-signal-change-rotation-orders) |
| O19 | FX Momentum with Standard Lots | Market | Contract-aware whole-lot momentum on USD-quoted spot pairs | [source](./example_orders_19_fx_momentum_lots.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-19-fx-momentum-lots) |
| O20 | Futures Donchian Contracts | Market | Whole-contract ATR sizing on provider continuous futures | [source](./example_orders_20_futures_donchian_contracts.py) | [view](https://backtester.quantjourney.cloud/strategies/example-orders-20-futures-donchian-contracts) |

O19-O20 consume `instrument_specs` returned by qj-api. Multipliers and lot sizes
are applied, but margin/buying-power enforcement, FX swaps and currency
conversion, dated futures selection, and controlled roll execution remain out
of scope for these examples.

## Walk-forward & optimization examples (5)

Validate temporal robustness and tune parameters. WF01-WF03 default to the
fast `slice_diagnostics` mode, which scores train/test slices from one full
strategy run. For a fuller per-fold re-run/refit, use
`QJ_WF_MODE=per_fold_refit`; this is slower because every fold runs the
strategy again. WF04-WF05 are optimization workflows and should be read as
selection diagnostics, not as a simple winner-takes-all backtest.

| # | Example | Idea | Code | Results |
|:--|:--|:--|:--|:--|
| WF01 | Rolling Walk-Forward | Sliding fixed-length train/test windows with a pre-OOS purge | [source](./example_wf_01_rolling_walkforward.py) | [view](https://backtester.quantjourney.cloud/strategies/example-wf-01-rolling-walkforward) |
| WF02 | Expanding Walk-Forward | Ever-growing training window vs sliding test window | [source](./example_wf_02_expanding_walkforward.py) | [view](https://backtester.quantjourney.cloud/strategies/example-wf-02-expanding-walkforward) |
| WF03 | Anchored + Pre-OOS Purging | Fixed and percentage-based exclusions before each test window | [source](./example_wf_03_anchored_purge_embargo.py) | [view](https://backtester.quantjourney.cloud/strategies/example-wf-03-anchored-purge-embargo) |
| WF04 | Grid Search | Exhaustive SMA fast/slow tuning scored by real backtests | [source](./example_wf_04_grid_search_optimization.py) | [view](https://backtester.quantjourney.cloud/strategies/example-wf-04-grid-search-optimization) |
| WF05 | Optuna TPE + Walk-Forward | Bayesian parameter search, then out-of-sample validation | [source](./example_wf_05_optuna_tpe_optimization.py) | [view](https://backtester.quantjourney.cloud/strategies/example-wf-05-optuna-tpe-optimization) |

---

WF05 requires the optional Optuna dependency: `pip install optuna`.

See the repository [README](../README.md) for install, data granularity,
report output, and the strategy skeleton.
