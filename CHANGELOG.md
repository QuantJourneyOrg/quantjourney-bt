# QuantJourney Backtester Changelog

## 0.12.1 - 2026-07-18

### Added
- Added `deflated_sharpe_method` to serialized walk-forward results so each
  reported probability carries its statistical construction.

### Changed
- Walk-forward trial objectives pooled across chronological folds are now
  labelled `pooled_walk_forward_dsr_style`, rather than canonical DSR for one
  common trial population.
- An N=1 Sharpe probability is labelled `probabilistic_sharpe_n1`, making clear
  that multiple-testing deflation has vanished and the statistic is PSR.

## 0.12.0 - 2026-07-18

### Added
- Added `extra_pre_oos_purge_pct`, an honestly named percentage extension of
  the exclusion immediately before each OOS window. The historical
  `embargo_pct` input remains a compatibility alias for 0.12.x and does not
  claim classical post-test embargo semantics.
- Added `rank_stability_trials` and
  `walk_forward_top_k_rank_failure_rate` for the rolling top-K OOS rank
  diagnostic, plus deprecated read aliases for the former `pbo_*` names.
- Added `dsr_effective_n_trials` and result fields that report both the raw
  finite completed-trial count and the effective trial count used by DSR.

### Changed
- Renamed the rolling top-K rank diagnostic so it is no longer presented as
  canonical CSCV PBO. Canonical PBO requires a complete configuration matrix
  and symmetric CSCV combinations; no CSCV threshold is applied to this
  rolling sensitivity metric.
- Documented the raw completed-trial count as a conservative DSR approximation
  when dependence between parameter variants is not estimated.

### Fixed
- Per-fold refit examples now propagate the runner's ISO `train_start` and
  `oos_end` values into each strategy's `backtest_period`.
- Per-fold refits and top-K candidate reruns now fail closed when returned NAV
  is undated, contains `NaT`, or escapes the requested fold date bounds.
- Corrected public documentation that previously described the percentage
  pre-OOS purge extension as classical post-test embargo.

## 0.11.0 - 2026-07-18

### Added
- Added `volume_lookback` and `expected_open_volume_fraction` controls for
  forecasting the liquidity available to fills at the open from lagged data.

### Changed
- Fast-weight costs, implied quantities, positions and NAV now share one
  recursive post-cost capital path. Costed quantity deltas reconcile with
  reported position changes instead of being scaled from a separate gross NAV.
- `min_trade_value` is a cost-materiality threshold: sub-threshold implied
  trades remain visible in the quantity and notional audit while incurring no
  modeled cost.

### Fixed
- Opening fills no longer expose the current bar's later high, low or close to
  range-sensitive slippage models.
- Opening participation caps now use lagged observed volume rather than the
  completed current bar's future full-day volume, and fail closed when no
  lagged observation exists.

## 0.10.2 - 2026-07-18

### Added
- Added native Windows repository support through `strategy.bat`, backed by the
  same cross-platform Python launcher used by `strategy.sh`.
- Added `WINDOWS.md`, automatic non-executing `.env` loading, and Windows CI
  coverage for listing and import-checking strategies.

### Fixed
- Preserved long and short directions when applying pure inverse-volatility
  weighting to active instruments.

## 0.10.1 - 2026-07-11

### Added
- Added `qj-bt data`, an unauthenticated public data-catalog CLI.
- Added an optional terminal browser with Rich tables and keyboard navigation.
- Added deterministic `overview`, `sources`, `granularities`, `datasets`,
  `asset-classes`, `universes`, `example-symbols`, and `all` commands.
- Added `--json` output for scripts, agents, and automation.
- Added `RebalancePolicy(calendar_dates=...)` for explicit rebalance timestamps
  alongside frequency-based schedules.
- Added a public six-engine benchmark harness under `compare/native/`: five
  strategies implemented natively in QJ, VectorBT, pmorissette/bt, Zipline,
  Backtrader and QuantConnect LEAN, with an invariant-gated runner and a QJ
  adapter that runs on this package. The harness is repository-only and is not
  included in the wheel.

### Changed
- Interactive browsing is enabled only for terminal sessions; non-interactive
  calls return the catalog overview without prompting.
- Public catalog requests do not require, read, or transmit an API key.
- Catalog output identifies symbols derived from example universes as example
  symbols rather than an exhaustive list of available instruments.
- Registered the `qj-bt` console entry point and updated the locked dependency
  set and public release manifest.
- Weight-mode accounting runs on a vectorized path. Results are unchanged and
  were verified bit-identical across the full benchmark suite.
- Position-limit risk checks skip rows where no limit can bind.

### Fixed
- Extended release verification to validate the exact console entry point in
  wheel and source-distribution artifacts.
- Added coverage for metadata normalization, fallback data, terminal and
  non-terminal behavior, JSON output, errors, and release artifacts.
- Start-anchored rebalance frequencies (`MS`, `QS`, `YS`, ...) snap forward to
  the next trading session instead of backward into the prior period.

## 0.10.0 - 2026-07-10

### Added
- Added the shared execution simulator, contract-aware portfolio ledger,
  portfolio-of-strategies book, and pre-trade risk controls to the explicitly
  approved Apache-2.0 public scope.
- Expanded the runnable catalog to 50 examples: 25 weight-based, 20
  order-based, and 5 walk-forward/optimization strategies, including the new
  FX and continuous-futures research examples.
- Added a fail-closed release boundary: local commit/push guards, a reviewed
  source manifest, clean-tag publishing, and exact wheel/sdist verification.

### Fixed
- Rejected unsupported cross-currency FX accounting in core and standalone
  ledger paths instead of treating quote-currency PnL as portfolio currency;
  incomplete or invalid external FX/futures contract metadata now fails closed.
- Standardized strategy storage on `(strategy, field, instrument)`, rejected
  silently misaligned strategy output, and preserved valid all-zero strategies.
- Scoped ETag response caches to one immutable per-request tenant/principal
  snapshot, including during concurrent context changes; cache keys are fully
  opaque and invalidated whenever the security context changes.

### Changed
- Restored the public README and changelog boundary, then documented only the
  approved public runtime and strategy additions.
- Modernized Python 3.11 typing syntax, removed the Ruff backlog, and made full
  lint/format checks blocking.
- Added a locked full-package mypy baseline gate that fails on every
  unreviewed diagnostic change while the existing typing debt is reduced.
- Declared and continuously tests Python 3.11 through 3.14.
- External PyPI artifacts are built only from a clean version-matched tag after
  tests and artifact-manifest checks; local uploads are unsupported.

## 0.9.1 - 2026-07-09

### Changed
- Hardened missing-data accounting with frozen/held semantics so portfolio NAV, risk triggers and reported weights stay consistent through temporary price gaps.
- Normalized timezone alignment before reindexing benchmark and cash-buffer inputs.
- Improved order-mode idempotency and bracket/OCA lifecycle handling.
- Tightened circuit-breaker re-entry, weekly holiday scheduling and PBO failed-candidate handling.

## 0.8.9 - 2026-07-08

### Changed
- Weight-mode accounting now books the full price move across missing-data gaps on the resume bar (matching order-mode economics), and gapped positions are reported at their carried value.
- Honest walk-forward reporting: slice-diagnostics output is labeled in-sample end-to-end, failed folds are reported as failed (not zero Sharpe), verdicts are gated so losing strategies never render green, and the composite Sharpe carries a bootstrap confidence interval.
- Stricter, louder input handling: unknown constructor kwargs raise, execution mode and fill timing are validated, missing OHLC data degrades with a clear warning, inactive configuration knobs warn once per mode, and the tracking-error trigger activates when `benchmark_returns` is provided.
- Re-running a backtest on the same instance is idempotent (execution state resets per run); weekly calendar rebalances snap to the prior trading day on holidays; documentation now spells out the execution-timing contract and calendar-convention sensitivity.

## 0.8.8 - 2026-07-08

### Changed
- Refined same-bar fill priority for OCO orders and aligned the tracking-error trigger with the NAV accounting basis.
- Cleaner walk-forward metadata and configuration errors (empty purge windows, overlapping-OOS warning, early CPCV validation).

## 0.8.7 - 2026-07-07

### Changed
- Tightened rebalance and risk-event accounting so daily returns are always booked on the weights actually held, with improved circuit-breaker recovery behavior.
- Hardened order execution around partial fills and edge conditions: trailing-stop activation, OCO and bracket lifecycle, bar-expiry counting, stop-limit trigger bounds, and volume-cap handling with incomplete data.
- More robust input validation across sizing, weights and configuration (non-finite prices and volumes, cash-buffer types, rebalance frequency and holiday scheduling).
- Protective `stop_loss()`/`take_profit()` exits now link as an OCO pair automatically.
- Walk-forward upgrades: more reliable optimizer execution with loud failure reporting, `direction="minimize"` support, deflated Sharpe aligned with Bailey & López de Prado (2014), and honest availability reporting for overfitting statistics (`pbo_trials` opt-in).

## 0.8.6 - 2026-07-06

### Added
- Added explicit walk-forward mode reporting in logs, summaries and archived metadata: `slice_diagnostics` for fast NAV-slice diagnostics and `per_fold_refit` when a fold-local `backtester_factory` reruns the strategy.
- Added `QJ_WF_MODE=per_fold_refit` support to WF01-WF03 examples, with optional `QJ_WF_REPORT_PACKET=1` report/plot packet generation.

### Changed
- Deduplicated the walk-forward slice-diagnostics warning so it appears once per result instead of once per fold.
- Clarified README and strategy-catalog language around walk-forward diagnostics, per-fold refit cost, and optimization workflow interpretation.

## 0.8.5 - 2026-07-04

### Fixed
- Fixed order-mode trade recording so fill metadata from the execution engine (`slippage`, theoretical price and fill status) is accepted by the blotter and preserved in trade artifacts.
- Hardened the 30-minute intraday stop-breakout example against sparse provider bars by skipping invalid NAV, price and breakout-reference observations before sizing orders.

## 0.8.4 - 2026-07-04

### Added
- Added a deterministic bundled sample-data path for `example_weights_01_sma_daily` via `./strategy.sh example_weights_01_sma_daily --sample-data`, allowing a reproducible demo without API credentials.

## 0.8.3 - 2026-07-04

### Added
- Grew the example strategy suite to 45. Added 10 weight-based templates: long/short pairs trading with a ratio z-score and with a rolling OLS hedge-ratio spread; dollar-neutral cross-sectional momentum and short-term reversal; volatility-targeted trend and momentum baskets; risk-parity (equal risk contribution) standalone and chained with a per-position cap; Bollinger Band mean reversion; and MACD trend.
- Added a per-folder strategy catalog (`strategies/README.md`) that links each example's source and, where published, its results page, and embedded a summary catalog in the main README.
- Attached the risk-overlay modules (volatility targeting, risk parity, position limits, chained overlays) to runnable examples via the `risk_model=` hook.

## 0.8.2 - 2026-07-04

### Added
- Expanded the example strategy suite to 35 runnable templates: 12 weight-based, 18 order-based, and 5 walk-forward / optimization examples.
- Added an intraday timeframe grid so a single engine spans minute-to-hour cadences: `example_weights_08_intraday_1m_ema_scalp` (1m EMA scalp), `example_orders_15_intraday_5m_bracket_reversion` (5m bracket reversion), `example_orders_16_intraday_30m_stop_breakout` (30m stop breakout), and `example_weights_09_intraday_1h_sma_trend` (1h SMA trend).
- Added rebalance-focused templates that exercise the full policy stack: `example_weights_10_monthly_circuit_breaker` (drawdown circuit breaker + cooldown), `example_weights_11_quarterly_te_cost_gate` (tracking-error trigger + turnover budget), and `example_weights_12_daily_partial_drift` (partial drift-band rebalance).
- Added event-driven order-mode rotation templates: `example_orders_17_monthly_rotation_orders` (calendar rebalance expressed as orders) and `example_orders_18_signal_change_rotation_orders` (trade only on trend-signal flips).
- Added dedicated walk-forward and optimization examples: rolling, expanding, and anchored walk-forward (`example_wf_01`–`example_wf_03`), exhaustive grid search (`example_wf_04`), and Optuna TPE search with out-of-sample validation (`example_wf_05`).
- Surfaced walk-forward traffic-light interpretation (overfit ratio, efficiency, Sharpe decay) directly in the walk-forward examples.

### Changed
- Refreshed public documentation: rewrote the README around what the engine does, why it is reproducible, how to run it, and example report output; removed fixed strategy-count language so the suite can grow without stale numbers.
- Aligned every example strategy configuration with the verified `RebalancePolicy` fields (`drift_threshold`, `tracking_error_threshold`, `max_drawdown_trigger`, `max_annual_turnover`, `partial_rebalance`, `rebalance_on_signal_change`).
- Reworked the packaging smoke tests to check tracked files rather than a fixed strategy count, so new untracked example strategies no longer break local test runs.

### Fixed
- Preserved intraday NaN gaps in `Universe.returns` and `Universe.log_returns` using `pct_change(fill_method=None)`, so halted or illiquid intraday bars are excluded from allocation instead of being counted as 0% return observations.
- Seeded only the first available bar per instrument to a defined return, keeping pre-listing and post-halt gaps as unavailable rather than synthetic zeros.

## 0.8.1 - 2026-07-04

### Added
- Added `granularity` normalization for API-backed backtests, including yfinance historical intraday values such as `1m`, `5m`, `15m`, `30m`, and `1h`.
- Added `example_weights_07_intraday_rsi_15m.py`, a simple RSI strategy using `granularity="15m"`.
- Added true walk-forward refit hooks via fold-local `backtester_factory`/optimizer support.
- Added date/time-aware x-axis labels for intraday time-series plots.
- Added a per-fold "slice diagnostics" warning so walk-forward runs without a refit factory are labeled honestly rather than reported as true out-of-sample.

### Fixed
- Fixed order-mode contract accounting so cash, NAV, position values, weights, and trade value use `ContractSpec` multipliers and lot sizes.
- Fixed order-mode commission notional to use `ContractSpec` multipliers and lot sizes, aligning bps/max-pct fees with contract-aware NAV and PnL.
- Fixed limit and stop-limit execution semantics so slippage never fills worse than the limit price.
- Fixed partial-fill commissions so per-order minimums, caps, and tiers are applied cumulatively to the parent order.
- Fixed reporting-frequency resampling so the first reporting bucket keeps its actual return instead of a synthetic zero.
- Fixed Sharpe, volatility, and Sortino calculations to use metric returns that exclude the synthetic day-0 zero.
- Fixed position-limit redistribution to iteratively enforce `max_weight` after redistributing capped excess.
- Fixed partial rebalance and tax-aware rebalance normalization so frozen positions are not moved by the traded sleeve.
- Fixed benchmark return handling to prefer adjusted total-return sources and warn on ambiguous raw-close or missing-day alignment.
- Fixed dollar valuation semantics to use raw close prices for exposure, turnover, and dollar PnL, with bounded forward-fill for stale valuation prices.
- Preserved NaN market-data gaps in `Universe` and weight-mode rebalancing so unavailable instruments are excluded from allocation instead of treated as 0% return assets.

## 0.8.0 - 2026-06-20

### Added
- Added AI Co-Pilot materials for strategy authoring, strategy review, and report interpretation.
- Added explicit NAV accounting identity validation.
- Added per-instrument strategy trace charts combining price, indicators, signal state, realized exposure, and portfolio context.
- Added reporting semantics documentation for FIFO lot matching, exposure path checks, loss-positive risk fields, and execution assumptions.
- Added a reproducibility fingerprint over configuration and input data for auditable research runs.

### Changed
- Hardened instrument analytics contracts: units now mean executed position quantities, and weight-based attribution requires explicit weights.
- Standardized turnover to institutional half-turnover with separate gross weight churn diagnostics.
- Reworked VaR/CVaR outputs to use a loss-positive convention.

## 0.7.0 - 2026-06-12

### Added
- Added reporting-frequency support for daily, weekly, monthly, and quarterly report cadences.
- Added frequency-aware annualization, rolling windows, labels, and table wording.

### Fixed
- Removed hard-coded daily wording from risk and volatility report labels.
- Guarded resampling paths so reporting-frequency changes preserve strategy and benchmark alignment.

## 0.6.0 - 2026-05-30

### Added
- Added website changelog and documentation pages for the QuantJourney Backtester.
- Added architecture diagrams and publishing-oriented documentation.
- Added default QuantJourney plot styling, source stamps, and report metadata.
- Added multiple visual themes for report charts, including terminal, dark, and paper-ready academic styles.

### Changed
- Refined plot readability for cumulative returns, drawdown, rolling risk, rolling beta, rolling alpha, and holdings charts.

## 0.5.0 - 2026-02-01

### Added
- Added anchored, rolling, and expanding walk-forward validation.
- Added grid-search and Optuna optimization integration for strategy parameter studies.
- Added fold-level in-sample/out-of-sample diagnostics and summary metrics.
- Added purge and embargo gaps between train and test windows to reduce information leakage.
- Added overfit-ratio, efficiency, and Sharpe-decay traffic-light interpretation of walk-forward results.

### Changed
- Separated optimizer configuration from strategy configuration so research runs can be reproduced from saved metadata.

## 0.4.2 - 2026-01-10

### Added
- Added cross-engine comparison materials for QuantJourney, vectorbt, Backtrader, Zipline, and QuantConnect-style strategies.
- Added fair-metric comparison helpers for equity curves, timing, drawdown, and return statistics.

### Fixed
- Improved cash handling and signal timing checks for cross-engine benchmark parity.

## 0.4.1 - 2025-12-01

### Added
- Added order-mode strategy examples for market, limit, stop, stop-limit, trailing stop, bracket, and OCO behavior.
- Added trade blotter export paths for execution-mode strategies.

### Fixed
- Tightened same-bar stop/limit execution ordering and volume-participation partial fill behavior.

## 0.4.0 - 2025-10-15

### Added
- Added deterministic order-mode execution simulation.
- Added slippage and commission model interfaces.
- Added initial futures/FX contract specification types for multiplier, lot size, tick size, and margin metadata.

### Changed
- Split target-weight accounting from order-based accounting so strategies can choose the appropriate simulation mode.

## 0.3.0 - 2025-08-15

### Added
- Added portfolio analytics for returns, volatility, drawdowns, rolling statistics, attribution, exposure, and turnover.
- Added dashboard-oriented report artifacts: summary text, JSON metrics, CSV metrics, equity curve CSV, and PNG charts.
- Added benchmark comparison support for common index and ETF references.
- Added crisis-window analysis across predefined historical stress periods.
- Added Monte Carlo block-bootstrap resampling for NAV confidence bands and probability-of-ruin estimates.

### Fixed
- Improved NAV, cash, position, and weight alignment checks.

## 0.2.2 - 2025-06-20

### Added
- Added risk-model hooks for inverse volatility, volatility targeting, risk parity, and position limits.
- Added calendar, drift, signal-change, turnover-gate, and circuit-breaker rebalance policies.
- Added partial rebalance and tax-aware young-lot avoidance options.

### Changed
- Moved rebalance behavior into a dedicated engine so strategy logic stays focused on signals and target weights.

## 0.2.1 - 2025-04-20

### Added
- Added technical indicator configuration for SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and related features.
- Added reusable strategy examples for daily, weekly, monthly, and quarterly research workflows.

### Fixed
- Improved strategy data alignment between indicator frames, signal frames, and target-weight frames.

## 0.2.0 - 2025-03-10

### Added
- Added QuantJourney Cloud API-backed market-data fetch through `/bt/prepare`.
- Added SDK client integration, API-key auth support, and session/dataset identifiers.
- Added local pandas containers for instruments, portfolio data, signals, weights, and positions.

### Changed
- Established the design principle that market data comes from the cloud while strategy computation runs locally.

## 0.1.0 - 2025-02-01

### Added
- Initial QuantJourney Backtester package.
- Added the `Backtester` base class with signal, weight, and position hooks.
- Added local NAV calculation, basic portfolio returns, and strategy summary output.
- Added the first runnable SMA-style strategy skeleton.
