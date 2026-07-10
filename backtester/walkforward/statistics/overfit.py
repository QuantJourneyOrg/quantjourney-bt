"""
Overfit diagnostics — overfit ratio, efficiency, sharpe decay.

These are pure functions operating on scalars or lists of per-fold
metrics. No dependency on the WF engine or fold geometry.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def overfit_ratio(is_sharpe: float, oos_sharpe: float) -> float:
    """
    IS Sharpe / OOS Sharpe.

    > 1.5 → caution,  > 2.5 → likely overfit.
    Returns inf if OOS Sharpe ≤ 0 and IS Sharpe > 0.
    Returns 0.0 if both are ≤ 0.

    NOTE: the value is NOT interpretable when OOS Sharpe ≤ 0 (0.0 does
    not mean "robust" — it means "losing everywhere") or when IS Sharpe
    < 0 (negative ratio). ``interpret_metrics`` gates its verdict on the
    aggregate OOS Sharpe for exactly this reason — pass it as
    ``composite_sharpe`` in the metrics dict.
    """
    if oos_sharpe <= 0:
        return float("inf") if is_sharpe > 0 else 0.0
    return is_sharpe / oos_sharpe


def efficiency(is_cagr: float, oos_cagr: float) -> float:
    """
    OOS CAGR / IS CAGR.

    1.0 = perfect transfer,  < 0.4 → red flag.
    Returns 0.0 if IS CAGR ≤ 0.
    """
    if is_cagr <= 0:
        return 0.0
    return oos_cagr / is_cagr


def sharpe_decay(oos_sharpes: Sequence[float]) -> float:
    """
    Slope of OOS Sharpe across folds (linear regression).

    Negative slope → alpha is decaying over time.
    Returns 0.0 if fewer than 2 folds.

    NOTE: a positive slope is NOT evidence of health when the strategy
    loses money — e.g. sharpe_decay([-2, -1.5, -1, -0.5]) = +0.5
    ("improving") for an always-losing strategy. ``interpret_metrics``
    gates its verdict on the aggregate OOS Sharpe (``composite_sharpe``).
    """
    n = len(oos_sharpes)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=float)
    y = np.array(oos_sharpes, dtype=float)

    # Simple linear regression slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def aggregate_overfit_ratio(
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
) -> float:
    """Mean IS Sharpe / mean OOS Sharpe across folds."""
    mean_is = np.mean(is_sharpes)
    mean_oos = np.mean(oos_sharpes)
    return overfit_ratio(float(mean_is), float(mean_oos))


def aggregate_efficiency(
    is_cagrs: Sequence[float],
    oos_cagrs: Sequence[float],
) -> float:
    """Mean OOS CAGR / mean IS CAGR across folds."""
    mean_is = np.mean(is_cagrs)
    mean_oos = np.mean(oos_cagrs)
    return efficiency(float(mean_is), float(mean_oos))
