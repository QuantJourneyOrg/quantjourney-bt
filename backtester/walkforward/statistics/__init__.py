"""
Walk-Forward Statistics — reusable overfitting diagnostics.

All functions work on arrays/Series of Sharpe ratios or returns,
and can be used independently of the walk-forward engine.

Usage::

    from backtester.walkforward.statistics import (
        overfit_ratio,
        efficiency,
        sharpe_decay,
        deflated_sharpe,
        probability_of_backtest_overfitting,
        aggregate_oos_returns,
        compute_composite_metrics,
        interpret_metrics,
    )

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from backtester.walkforward.statistics.overfit import (
    overfit_ratio,
    efficiency,
    sharpe_decay,
)
from backtester.walkforward.statistics.deflated_sharpe import (
    deflated_sharpe,
    probabilistic_sharpe,
)
from backtester.walkforward.statistics.pbo import (
    probability_of_backtest_overfitting,
    pbo_logit_distribution,
    pbo_from_selected_ranks,
    selected_trial_logit,
)
from backtester.walkforward.statistics.aggregation import (
    aggregate_oos_returns,
    compute_composite_metrics,
)
from backtester.walkforward.statistics.interpretation import interpret_metrics

__all__ = [
    "overfit_ratio",
    "efficiency",
    "sharpe_decay",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "probability_of_backtest_overfitting",
    "pbo_logit_distribution",
    "pbo_from_selected_ranks",
    "selected_trial_logit",
    "aggregate_oos_returns",
    "compute_composite_metrics",
    "interpret_metrics",
]
