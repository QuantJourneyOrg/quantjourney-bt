# Strategy Examples

This folder contains **45 runnable example strategies** for the QuantJourney
Backtester — **22 weight-based**, **18 order-based**, and **5 walk-forward /
optimization** examples. Each file is a complete, self-contained template: copy
the one closest to your idea, change the rule, and you are testing your own
strategy in minutes.

Every strategy links to its **source** (the code) and, where published, to its
**results page** on [backtester.quantjourney.cloud](https://backtester.quantjourney.cloud/strategies)
with equity curves, tearsheets, and charts. Result pages for the newest
examples are rolling out — [browse the full gallery](https://backtester.quantjourney.cloud/strategies).

## Run one

```bash
# import-only check (no credentials, no data call)
./strategy.sh example_weights_01_sma_daily --check

# real backtest (after setting credentials)
export QJ_API_KEY="..."
./strategy.sh example_weights_01_sma_daily --output /tmp/qj-reports
```

Naming: `example_<mode>_<NN>_<name>.py`, where mode is `weights`, `orders`, or
`wf` (walk-forward / optimization).

---

## Weight-based strategies (22)

Portfolio thinking — produce target weights, let the rebalance engine (and any
risk overlay) trade them. Includes long/cash, market-neutral long/short, and
risk-overlay templates.

| # | Strategy | Idea | Code | Results |
|:--|:--|:--|:--|:--|
| W01 | Daily SMA Trend | Hold each stock while SMA(50) > SMA(200); daily rebalance | [source](./example_weights_01_sma_daily.py) | [view](https://backtester.quantjourney.cloud/strategies/daily-sma-trend) |
| W02 | Monthly ETF Trend + Drift | SMA(50/200) trend on ETFs; month-end + 5% drift band | [source](./example_weights_02_monthly_drift_etf.py) | [view](https://backtester.quantjourney.cloud/strategies/monthly-drift-etf) |
| W03 | Weekly RSI Reversion | Enter RSI(14) < 35, exit RSI > 60; weekly (Fri) | [source](./example_weights_03_weekly_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/weekly-rsi-reversion) |
| W04 | Quarterly Dual Momentum | Rank ETFs by 12-month return, hold top 2 if positive; quarter-end | [source](./example_weights_04_quarterly_dual_momentum.py) | [view](https://backtester.quantjourney.cloud/strategies/quarterly-dual-momentum) |
| W05 | Monthly Inverse Volatility | Size each ETF by inverse 63-day volatility; month-end | [source](./example_weights_05_monthly_inverse_vol.py) | [view](https://backtester.quantjourney.cloud/strategies/monthly-inverse-vol) |
| W06 | Signal-Change Defensive Rotation | SPY > SMA(200) → risk-on ETFs, else defensive; on signal change | [source](./example_weights_06_signal_change_defensive.py) | [view](https://backtester.quantjourney.cloud/strategies/signal-change-defensive) |
| W07 | Intraday RSI 15m | Equal-weight basket when RSI oversold; 15-minute bars | [source](./example_weights_07_intraday_rsi_15m.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W08 | Intraday EMA Scalp 1m | EMA(9/21) trend/cash; 1-minute bars | [source](./example_weights_08_intraday_1m_ema_scalp.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W09 | Intraday SMA Trend 1h | SMA(10/30) trend/cash; hourly bars | [source](./example_weights_09_intraday_1h_sma_trend.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W10 | Monthly + Circuit Breaker | Monthly ETF trend; flatten on a 15% drawdown + cooldown | [source](./example_weights_10_monthly_circuit_breaker.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W11 | Quarterly TE + Cost Gate | Momentum with tracking-error trigger and turnover budget | [source](./example_weights_11_quarterly_te_cost_gate.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W12 | Daily Partial Drift | Momentum tilt; trade only names past a 10% drift band | [source](./example_weights_12_daily_partial_drift.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W13 | Pairs Trading (Ratio Z-Score) | Market-neutral KO/PEP on a log-ratio z-score | [source](./example_weights_13_pairs_ratio_zscore.py) | [view](https://backtester.quantjourney.cloud/strategies/pairs-trading) |
| W14 | Pairs Trading (Hedge Ratio) | Market-neutral EWA/EWC on a rolling OLS hedge-ratio spread | [source](./example_weights_14_pairs_hedge_ratio.py) | [view](https://backtester.quantjourney.cloud/strategies/pairs-trading) |
| W15 | Cross-Sectional Momentum (L/S) | Long top-3 / short bottom-3 by 12-month return; monthly | [source](./example_weights_15_cross_sectional_momentum.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W16 | Cross-Sectional Reversal (L/S) | Long losers / short winners by 1-month return; weekly | [source](./example_weights_16_cross_sectional_reversal.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W17 | Vol-Targeted Trend | SMA trend basket scaled to a 10% volatility target | [source](./example_weights_17_vol_target_trend.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W18 | Vol-Targeted Momentum | Momentum basket scaled to a 15% volatility target | [source](./example_weights_18_vol_target_momentum.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W19 | Risk Parity (Multi-Asset ERC) | Equal risk contribution across a multi-asset basket | [source](./example_weights_19_risk_parity_multiasset.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W20 | Risk Parity + Position Cap | Sector ERC chained with a 25% per-position cap | [source](./example_weights_20_risk_parity_capped.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W21 | Bollinger Band Reversion | Buy below the lower band, exit at the midline | [source](./example_weights_21_bollinger_reversion.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| W22 | MACD Trend | Long while MACD is above its signal line | [source](./example_weights_22_macd_trend.py) | [browse](https://backtester.quantjourney.cloud/strategies) |

The long/short examples (W13–W16) are market-neutral; short borrow/financing is
not modeled (a documented research approximation).

## Order-based strategies (18)

Execution thinking — submit explicit orders through the fill engine with
slippage, commissions, and a trade blotter.

| # | Strategy | Order type | Idea | Code | Results |
|:--|:--|:--|:--|:--|:--|
| O01 | Market SMA Crossover | Market | Buy SMA(20) crossing above SMA(50), sell on reverse | [source](./example_orders_01_market_sma_cross.py) | [view](https://backtester.quantjourney.cloud/strategies/market-sma-cross) |
| O02 | Market RSI Reversion | Market | Buy RSI(14) < 35, sell RSI > 60 | [source](./example_orders_02_market_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/market-rsi-reversion) |
| O03 | Limit RSI Dip Buyer | Limit | Passive buy-limit below the close on weak RSI | [source](./example_orders_03_limit_rsi_dip.py) | [view](https://backtester.quantjourney.cloud/strategies/limit-rsi-dip) |
| O04 | Limit Trend Pullback | Limit | In an uptrend, wait for a 1% pullback to enter | [source](./example_orders_04_limit_trend_pullback.py) | [view](https://backtester.quantjourney.cloud/strategies/limit-trend-pullback) |
| O05 | Stop Breakout Entry | Stop | Buy-stop above the recent 20-day high | [source](./example_orders_05_stop_breakout_entry.py) | [view](https://backtester.quantjourney.cloud/strategies/stop-breakout-entry) |
| O06 | Protective Stop Loss | Market + Stop | Trend entry with a 5% protective stop | [source](./example_orders_06_stop_loss_protection.py) | [view](https://backtester.quantjourney.cloud/strategies/protective-stop-loss) |
| O07 | Stop-Limit Breakout | Stop-Limit | Enter breakouts but cap the maximum fill price | [source](./example_orders_07_stop_limit_breakout.py) | [view](https://backtester.quantjourney.cloud/strategies/stop-limit-breakout) |
| O08 | Stop-Limit Protection | Market + Stop-Limit | Trend entry, downside protected by a stop-limit sell | [source](./example_orders_08_stop_limit_protection.py) | [view](https://backtester.quantjourney.cloud/strategies/stop-limit-protection) |
| O09 | Trailing Stop Trend | Trailing Stop | Trend entry, 4% trailing stop manages the exit | [source](./example_orders_09_trailing_stop_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/trailing-stop-trend) |
| O10 | RSI + Trailing Stop | Trailing Stop | Oversold RSI entry, 5% trailing stop for risk | [source](./example_orders_10_trailing_stop_rsi.py) | [view](https://backtester.quantjourney.cloud/strategies/trailing-stop-rsi) |
| O11 | Trailing Stop-Limit | Trailing Stop-Limit | Trailing stop that converts to a limit on trigger | [source](./example_orders_11_trailing_stop_limit.py) | [view](https://backtester.quantjourney.cloud/strategies/trailing-stop-limit) |
| O12 | Bracket Trend | Bracket | Trend entry with a +6% / −3% bracket | [source](./example_orders_12_bracket_trend.py) | [view](https://backtester.quantjourney.cloud/strategies/bracket-trend) |
| O13 | Bracket RSI Reversion | Bracket | RSI dip with a +4% / −2% bracket | [source](./example_orders_13_bracket_rsi_reversion.py) | [view](https://backtester.quantjourney.cloud/strategies/bracket-rsi-reversion) |
| O14 | OCO Dip or Breakout | OCO | Competing buy-limit (dip) and buy-stop (breakout) | [source](./example_orders_14_oco_dip_or_breakout.py) | [view](https://backtester.quantjourney.cloud/strategies/oco-dip-or-breakout) |
| O15 | Intraday 5m Bracket Reversion | Bracket | Oversold-RSI dips with a tight +0.6% / −0.4% bracket; 5-min bars | [source](./example_orders_15_intraday_5m_bracket_reversion.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O16 | Intraday 30m Stop Breakout | Stop | Buy-stop above the 12-bar high, fixed holding period; 30-min bars | [source](./example_orders_16_intraday_30m_stop_breakout.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O17 | Monthly Rotation (orders) | Market | Event-driven monthly momentum rotation, executed with orders | [source](./example_orders_17_monthly_rotation_orders.py) | [browse](https://backtester.quantjourney.cloud/strategies) |
| O18 | Signal-Change Rotation (orders) | Market | Trade only on SMA trend-signal flips (no calendar) | [source](./example_orders_18_signal_change_rotation_orders.py) | [browse](https://backtester.quantjourney.cloud/strategies) |

## Walk-forward & optimization examples (5)

Prove a strategy generalizes — validate out-of-sample and tune parameters.

| # | Example | Idea | Code | Results |
|:--|:--|:--|:--|:--|
| WF01 | Rolling Walk-Forward | Sliding fixed-length train/test windows with purge/embargo | [source](./example_wf_01_rolling_walkforward.py) | [view](https://backtester.quantjourney.cloud/strategies/walkforward-case-study) |
| WF02 | Expanding Walk-Forward | Ever-growing training window vs sliding test window | [source](./example_wf_02_expanding_walkforward.py) | [view](https://backtester.quantjourney.cloud/strategies/walkforward-case-study) |
| WF03 | Anchored + Purge/Embargo | How purge and embargo gaps prevent train/test leakage | [source](./example_wf_03_anchored_purge_embargo.py) | [view](https://backtester.quantjourney.cloud/strategies/walkforward-case-study) |
| WF04 | Grid Search | Exhaustive SMA fast/slow tuning scored by real backtests | [source](./example_wf_04_grid_search_optimization.py) | [view](https://backtester.quantjourney.cloud/strategies/optuna-optimization) |
| WF05 | Optuna TPE + Walk-Forward | Bayesian parameter search, then out-of-sample validation | [source](./example_wf_05_optuna_tpe_optimization.py) | [view](https://backtester.quantjourney.cloud/strategies/optuna-optimization) |

---

WF05 requires the optional Optuna dependency: `pip install optuna`.

See the repository [README](../README.md) for install, data granularity,
report output, and the strategy skeleton.
