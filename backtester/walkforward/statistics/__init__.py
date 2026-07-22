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

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from backtester.walkforward.statistics.aggregation import (
    aggregate_oos_returns,
    bootstrap_sharpe_ci,
    compute_composite_metrics,
)
from backtester.walkforward.statistics.deflated_sharpe import (
    deflated_sharpe,
    probabilistic_sharpe,
)
from backtester.walkforward.statistics.interpretation import interpret_metrics
from backtester.walkforward.statistics.overfit import (
    efficiency,
    overfit_ratio,
    sharpe_decay,
)
from backtester.walkforward.statistics.pbo import (
    is_oos_transfer_distribution,
    pbo_from_selected_ranks,
    pbo_logit_distribution,
    probability_of_backtest_overfitting,
    selected_trial_logit,
    selected_trial_rank_logit,
    walk_forward_top_k_rank_failure_rate,
)

__all__ = [
    "overfit_ratio",
    "efficiency",
    "sharpe_decay",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "probability_of_backtest_overfitting",
    "is_oos_transfer_distribution",
    "selected_trial_rank_logit",
    "walk_forward_top_k_rank_failure_rate",
    # Deprecated 0.12.x aliases:
    "pbo_logit_distribution",
    "pbo_from_selected_ranks",
    "selected_trial_logit",
    "aggregate_oos_returns",
    "bootstrap_sharpe_ci",
    "compute_composite_metrics",
    "interpret_metrics",
]
