# Native cross-engine strategy benchmark

Six backtesting engines, five strategies, one research contract — and **every
engine computes its own indicators, signals, state, target weights and
rebalance dates**. No adapter reads a precomputed decision matrix, execution
weights, share counts, or any other engine's output. The only shared inputs
are the immutable OHLCV panel and the explicit contract below; the only shared
code is data loading, contract constants and result serialization
(`common.py`).

Published results and full methodology:
<https://backtester.quantjourney.cloud/compare>

## Engines

| Engine | Adapter | Position model |
|---|---|---|
| QJ Backtester | `qj_engine.py` (ships with `quantjourney-bt >= 0.10.1`) | fractional |
| VectorBT | `vectorbt_engine.py` | fractional |
| pmorissette/bt | `pm_bt_engine.py` | fractional |
| Zipline Reloaded | `zipline_engine.py` | whole shares |
| Backtrader | `backtrader_engine.py` | whole shares |
| QuantConnect LEAN | `lean_engine.py` + `lean_algorithm.py` (official Docker image) | whole shares |

## Contract

- Instruments: AAPL, MSFT, NVDA, GOOGL, and AMZN.
- Shared adjusted OHLCV: 2015-01-02 through 2024-12-31.
- Warm-up: calendar year 2015.
- Reported evaluation: 2016-01-04 through 2024-12-31 (2,264 sessions).
- For daily strategies, the 2015-12-31 decision executes on the first
  evaluation session; no portfolio positions are held during warm-up.
- Initial capital: $100,000.
- Decision timing: after `close(t)`. Execution timing: `close(t+1)`.
- Costs and slippage: zero. Cash buffer: 0.1%.
- Long-only; gross target exposure no greater than 100%.
- Fractional accounting where the engine supports it naturally; whole-share
  orders in Backtrader, Zipline, and LEAN.

## Data panel

The benchmark expects `compare/protocol_artifacts/ohlcv.parquet`:

- a pandas DataFrame with a tz-naive `DatetimeIndex` (exchange sessions,
  2015-01-02 → 2024-12-31) and two-level columns `(ticker, field)` for the
  five tickers and fields `open, high, low, close, volume`;
- split/dividend-adjusted prices (total-return basis), no missing cells,
  values rounded to 6 decimals.

We do not redistribute market data. Build the panel from your own data
source; the canonical panel used for the published results has SHA-256

```
a7a986fbd50f21d2e340911df27cc87f374e5c77842b3ce526c9854ea03c459a
```

and every result JSON embeds the hash of the panel it was computed on, so
any reproduction states its data provenance explicitly. With a different
data vintage, expect small NAV differences but identical benchmark structure
(decision agreement, divergence attribution, fractional-trio identity).

## Strategy implementations

1. SMA(50/200), daily decisions, 25% binding per-position cap, residual cash.
2. RSI(14) state machine, enter below 30 and exit above 70. Each engine uses
   its native/default RSI semantics; this intentionally exposes indicator
   differences such as VectorBT rolling RSI versus Wilder smoothing.
3. Equal-weight monthly rebalance on the first exchange session.
4. Top-3 12m-minus-1m momentum, 63-session volatility estimate, 15% target,
   no leverage, monthly rebalance.
5. Top-2 12-month dual momentum with an absolute-momentum cash gate.

## Environments

Each engine runs in its own interpreter so dependency stacks never mix:

- `compare/.envs/vbt/`, `compare/.envs/pm_bt/`, `compare/.envs/zipline/`,
  `compare/.envs/backtrader/` — create with `uv venv` / `pip install`
  (vectorbt, bt, zipline-reloaded, backtrader respectively, plus pandas
  and pyarrow).
- QJ and the runner use the repository `.venv` (`pip install quantjourney-bt`).
- LEAN runs in the official `quantconnect/lean` Docker image; the harness in
  `compare/quantconnect/run_lean.py` converts the panel to LEAN's data format
  and extracts auxiliary files from the image automatically.

## Run

From the repository root:

```bash
# Complete 5 × 6 native matrix (Docker required for LEAN)
.venv/bin/python compare/native/runner.py

# One engine or strategy
.venv/bin/python compare/native/runner.py --engine vectorbt
.venv/bin/python compare/native/runner.py --strategy 2

# Rebuild reports and invariant checks from existing outputs
.venv/bin/python compare/native/runner.py --summary-only
```

Outputs land in `compare/native_results/` (gitignored): one NAV CSV and
metrics JSON per engine/strategy, independently computed decision-weight
evidence, `summary.csv`, and `decision_divergence.csv` reporting the first
target difference of every engine versus QJ.

The runner's invariant gate requires: a complete 5 × 6 result matrix, a single
data hash, one shared calendar, and **no unattributed decision divergences** —
the only accepted divergence is VectorBT's documented rolling-RSI semantics.

Identical results are allowed when independent implementations produce the
same targets and share fractional accounting semantics. Differences are also
allowed; they must be attributed to indicator, calendar, sizing, rounding, or
execution behavior rather than normalized away.
