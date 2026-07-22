# Compare

Public code for cross-engine comparisons.

**[`native/`](native/README.md)** contains the six-engine native strategy
benchmark behind <https://backtester.quantjourney.cloud/compare>: five
strategies implemented independently in QJ, VectorBT, pmorissette/bt, Zipline,
Backtrader and QuantConnect LEAN, with a runner that enforces invariant gates
(complete result matrix, one data hash, one calendar, attributed decision
divergences only). Every adapter computes its own indicators, signals, state
and rebalances — nothing is replayed from a shared decision matrix.

These files are intentionally small and strategy-focused. They exist so a reader
can inspect the exact logic behind each comparison, and so the
same idea can be run on different engines under identical assumptions.

## The comparison contract

A fair cross-engine comparison holds everything except the engine constant:

- same data
- same rebalance calendar
- same cost assumptions (commissions, slippage)
- same signal timing (when a signal is observed vs when it trades)
- same target-weight semantics

Only then does a difference in results say something about the engine rather than
about mismatched inputs.

## When engines disagree

Differences almost always trace back to one of these, not to "one engine being
better":

- **Execution timing** — same-bar vs next-bar; when a signal is allowed to trade.
- **Ranking windows** — off-by-one lookbacks in cross-sectional selection.
- **Cash handling** — how uninvested cash and drift are treated.
- **Fee accounting** — per-trade vs per-share, turnover cost vs per-fill cost.
- **Rounding** — fractional vs whole shares, lot sizes.
- **Calendar alignment** — trading-day calendars, holidays, timezones.
- **Order lifecycle** — how pending stop/limit/OCO orders are triggered and cancelled.

The value of a comparison is not the leaderboard — it is making these assumptions
explicit so a result can be trusted or reproduced.

See the [engine reference](https://backtester.quantjourney.cloud/engine) for the
QuantJourney timing rules and failure modes, and the
[strategy catalog](../strategies/README.md) for the runnable examples.
