"""
OOS aggregation — concatenate per-fold OOS returns into a single equity curve
and compute composite metrics.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtester.utils.logger import logger


def aggregate_oos_returns(
    fold_oos_returns: List[pd.Series],
) -> Tuple[pd.Series, pd.Series]:
    """
    Concatenate per-fold OOS returns into a single time series.

    If OOS windows overlap (step < test), overlapping dates are averaged.

    Returns:
        (oos_returns, oos_nav) — daily returns and equity curve rebased to 1.0.
    """
    if not fold_oos_returns:
        empty = pd.Series(dtype=float)
        return empty, empty

    combined = pd.concat(fold_oos_returns)

    # Handle overlapping dates by averaging
    if combined.index.duplicated().any():
        logger.warning(
            "Overlapping OOS windows detected (step < test): duplicated dates are "
            "averaged across folds. With per-fold refit this blends differently "
            "parameterized strategies and biases the composite Sharpe upward; "
            "prefer step_months >= test_months for a realized equity curve."
        )
        combined = combined.groupby(combined.index).mean()

    combined = combined.sort_index()
    nav = (1.0 + combined).cumprod()

    return combined, nav


def compute_composite_metrics(
    oos_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Compute aggregate OOS metrics from concatenated returns.

    Returns dict with keys: sharpe, cagr, max_dd, volatility.
    """
    if oos_returns.empty:
        return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0, "volatility": 0.0}

    n_days = len(oos_returns)
    years = n_days / trading_days

    # Annualised return
    total_return = (1.0 + oos_returns).prod() - 1.0
    cagr = (1.0 + total_return) ** (1.0 / max(years, 1e-9)) - 1.0

    # Annualised volatility
    vol = oos_returns.std() * np.sqrt(trading_days)

    # Sharpe
    excess = oos_returns.mean() - risk_free_rate / trading_days
    sharpe = (excess / oos_returns.std() * np.sqrt(trading_days)) if oos_returns.std() > 0 else 0.0

    # Max drawdown
    nav = (1.0 + oos_returns).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "max_dd": max_dd,
        "volatility": float(vol),
    }
