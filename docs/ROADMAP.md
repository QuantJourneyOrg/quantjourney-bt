# Roadmap

This is the direction of travel for the QuantJourney Backtester, not a
schedule. Items are grouped by theme and listed without dates or ordering
commitments; priorities shift with user feedback. Nothing here is a promise of
delivery.

The guiding goal is unchanged: make research assumptions explicit, keep runs
reproducible, and close the gap between a good-looking backtest and one that
survives due diligence.

## Data Depth

- **Point-in-time (PIT) fundamentals.** Company financials and derived ratios
  stamped with the date each figure actually became public, so strategies never
  see a number before it was knowable. Sourced from processed regulatory
  filings with restatement tracking.
- **Survivorship-free universes.** Historical index membership as it was on
  each date, including names that later delisted, so index backtests stop
  implicitly betting on today's winners.
- **Point-in-time reference data.** Temporal ticker mapping and sector/industry
  classifications, so long histories do not break on symbol changes.

## Accounting Fidelity

- **Corporate-actions ledger.** Dividends credited as real cash, delisting
  returns realized, and splits handled on position quantities - separating
  price return from total return in an auditable way.
- **Financing, borrow, and margin.** Short borrow fees, funding on leverage,
  interest on cash, and margin modeling - so long/short and futures research
  reflects the cost of carry instead of omitting it. This closes the borrow and
  financing assumptions noted in the README limitations.
- **Multi-currency accounting.** Converting non-base-currency positions and PnL
  to the portfolio's base currency for global and cross-asset books.

## Research Workflow

- **Faster reproducible starts.** Predefined data bundles for the example
  universes, so common strategies run in seconds without repeated data fetches.
- **Cross-sectional and factor tooling.** First-class helpers for ranking,
  neutralizing, and combining signals across a universe, building on the
  weight-mode long/short examples.

## Validation

- **Deeper overfit detection.** Combinatorial purged cross-validation,
  probability of backtest overfitting, and deflated Sharpe on top of the
  existing walk-forward, purge/embargo, and optimization diagnostics.

## Reach

- **Multi-timeframe research.** Combining signals observed on one cadence with
  execution on another.
- **Richer reporting.** More report formats and interactive views alongside the
  current static charts and PDF tear sheets.

---

Have a request or a use case that is not covered? Feedback shapes what moves up
this list.
