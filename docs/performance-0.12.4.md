# Performance refactor in 0.12.4

QuantJourney Backtester 0.12.4 reduces Python and pandas overhead without changing the accounting model or public API. The work followed profiling of lightweight weight-based backtests and order-based simulations; it did not replace the engine with a native-language implementation or skip validation.

## Measured result

The native comparison suite gives every engine the same adjusted OHLCV panel and research contract. Each engine independently computes its indicators, signals, decisions, and portfolio path.

The values below are median core seconds from three consecutive complete runs on an Apple M5 Max. Data download, process startup, report rendering, and exports are excluded. Lower is better.

| Strategy | QJ 0.12.4 | VectorBT | pm/bt | Zipline | Backtrader | LEAN |
|---|---:|---:|---:|---:|---:|---:|
| SMA 50/200 | **0.73 s** | 0.16 s | 1.35 s | 6.59 s | 1.23 s | 1.94 s |
| RSI Reversion | **2.21 s** | 1.99 s | 3.28 s | 1.88 s | 0.88 s | 1.20 s |
| Monthly Equal Weight | **0.16 s** | 0.11 s | 0.58 s | 1.06 s | 0.74 s | 0.94 s |
| Momentum + Vol Target | **0.36 s** | 0.18 s | 0.53 s | 1.45 s | 0.72 s | 0.81 s |
| Dual Momentum | **0.23 s** | 0.16 s | 0.44 s | 1.20 s | 0.71 s | 0.78 s |

VectorBT uses rolling RSI semantics in this fixture; the other implementations use Wilder-smoothed RSI. The resulting RSI path difference is expected and is attributed separately from timing.

### QJ before and after

Percentages compare the previously published values with the new unrounded three-run medians.

| Strategy | Before | 0.12.4 median | Reduction | Speedup |
|---|---:|---:|---:|---:|
| SMA 50/200 | 0.85 s | 0.734 s | **13.6%** | **1.16x** |
| RSI Reversion | 2.58 s | 2.214 s | **14.2%** | **1.17x** |
| Monthly Equal Weight | 0.46 s | 0.163 s | **64.5%** | **2.82x** |
| Momentum + Vol Target | 0.67 s | 0.363 s | **45.8%** | **1.84x** |
| Dual Momentum | 0.55 s | 0.225 s | **59.0%** | **2.44x** |
| **Sum of medians** | **5.11 s** | **3.700 s** | **27.6%** | **1.38x** |

The comparison is a compact workflow benchmark, not a universal engine ranking. Universe size, execution mode, order count, risk models, and reporting requirements can change the dominant cost.

## What changed

### Contract specifications are resolved once

The previous lookup used an eager default expression:

```python
spec = contract_specs.get(key, get_contract_spec(key))
```

Python evaluates `get_contract_spec(key)` even when `key` already exists. Repeated portfolio operations therefore reconstructed immutable specifications that were immediately discarded.

Version 0.12.4 uses an explicit miss path, normalizes symbols before a bounded `lru_cache`, and retains validated specifications in each backtester and ledger. Multi-asset PnL, margin, and notional calculations also avoid eager fallback evaluation.

### The simulator aligns pandas data once

The order simulator used scalar `.loc[date, instrument]` calls for each OHLCV field inside its date-by-instrument loop. It now performs one label-aware alignment and uses positional NumPy access inside the loop:

```python
close_values = close.loc[all_dates, instruments].to_numpy(copy=False)

for row, date in enumerate(all_dates):
    for column, instrument in enumerate(instruments):
        close_value = close_values[row, column]
```

Bulk `.loc` preserves the previous missing-label behavior. Optional volume data is aligned separately and missing instruments retain the existing `NaN` fallback.

### Ledger history is preallocated

An execution simulation knows its date index and instrument list before the first bar. The simulator now preallocates NumPy buffers for NAV, cash, positions, values, exposures, average entry prices, and margin. Pandas objects are constructed once at the result boundary instead of growing lists of dictionaries on every bar.

The standalone append-based ledger path remains available, so this is not an API change.

### Optional reporting imports are lazy

Importing a lightweight execution component previously initialized optional performance, plotting, PDF, narrative, and report-builder modules through `backtester.engines.__init__`.

Those public exports are now resolved through module-level `__getattr__` only when requested. This reduces cold-start work for lightweight backtests while retaining the historical optional-dependency fallback.

Cold-import savings are not included in the timing table because process startup is deliberately outside `core_seconds`.

### Risk and tracking-error paths reuse complete arrays

- Inverse volatility computes one shifted rolling-volatility matrix instead of slicing and recomputing a pandas window for every date. The shift preserves the rule that a decision at t uses returns strictly before t.
- Risk parity converts weights and rescore flags to arrays once, solves only when required by the existing rescore/signature rules, and propagates the last solution between those events.
- The rebalance engine stores each portfolio bar return as it updates NAV. Tracking-error windows slice that history instead of reconstructing the same returns from weights and asset returns.

## Correctness checks

Performance-only changes received regression tests against the previous implementations. The checks cover:

- lazy import boundaries;
- inverse-volatility parity, including warm-up, signs, zero exposure, and missing data;
- risk-parity parity;
- complete equality between preallocated and append-based ledger results;
- one immutable contract resolution per instrument;
- identical rolling tracking-error values.

The native publication benchmark then ran three complete 5 x 6 matrices: 90 engine-strategy runs in total. Every run contained all 30 results, used one canonical data hash and calendar, and passed the divergence gate. Final NAV, CAGR, Sharpe, and maximum drawdown did not change between repetitions; the maximum repeated-run Final NAV delta was $0.00.

## Why this stayed in Python

The profile was dominated by repeated object construction, scalar pandas indexing, temporary dictionaries, eager imports, and recomputed rolling inputs. These costs can be removed without introducing a compiled extension boundary.

NumPy buffers and pandas vectorization already execute dense arithmetic in compiled code while keeping the accounting rules inspectable and the package easy to install. A Rust kernel remains an option if future profiles show a stable, arithmetic-heavy loop dominating after Python-level overhead has been removed.

