# Compare

Public strategy code for cross-engine comparisons.

These files are intentionally small and strategy-focused. They are designed to be linked from the QuantJourney Compare page so readers can inspect the exact QJ-style logic behind each comparison row.

The public comparison target is:

- same data
- same rebalance calendar
- same cost assumptions
- same signal timing
- same target-weight semantics

When engines disagree, the forensic question is usually execution timing, ranking windows, cash handling, fee accounting, rounding, calendar alignment, or order lifecycle behavior.
