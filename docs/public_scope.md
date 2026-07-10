# Public Scope

This repository is the open-source distribution of the QuantJourney
Backtester. It is developed alongside the QuantJourney platform; this page
documents what ships here versus what is provided by the hosted platform.

## Included in this repository (Apache-2.0)

- Deterministic backtesting engine: portfolio accounting, order lifecycle
  (market, limit, stop, stop-limit, trailing, bracket, OCO), rebalancing
  modes, transaction-cost and slippage models.
- Shared execution simulator and contract-aware portfolio ledger, including
  portfolio-of-strategies allocation and pre-trade risk controls.
- Walk-forward validation and optimization: rolling, expanding, anchored,
  and purged/embargoed fold schemes; grid search and Optuna (TPE) parameter
  optimization; overfitting diagnostics (deflated Sharpe ratio, PBO,
  overfit-ratio interpretation).
- 50 example strategies (25 weights-based, 20 order-based, 5 walk-forward)
  with a reproducible sample-data mode that requires no account.
- Static report generation: metrics, PNG chart pack, and a static HTML
  dashboard.
- SDK client for the QuantJourney cloud API (optional; the bundled sample-data
  path runs fully offline).

## Provided by the hosted platform (not in this repository)

- Market data warehouse and authenticated data APIs.
- Interactive dashboards and hosted report sharing.
- PDF factsheets, narrative report generation, and extended diagnostic
  plot packs (crisis analysis, trade blotter, execution traces).
- Data ingestion, scheduling, and platform orchestration.

The split is intentional: everything needed to run, validate, and trust a
backtest locally is open source; hosted convenience and data services are
part of the platform at [quantjourney.cloud](https://backtester.quantjourney.cloud).
