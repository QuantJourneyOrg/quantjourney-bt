"""
OOS aggregation — concatenate per-fold OOS returns into a single equity curve
and compute composite metrics.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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


def bootstrap_sharpe_ci(
    returns: pd.Series,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> Optional[Tuple[float, float]]:
    """
    Stationary block bootstrap 5%/95% CI for the annualized Sharpe ratio.

    Politis & Romano (1994): resamples wrap-around blocks whose lengths
    are geometric with expected length ≈ √T, preserving short-range
    autocorrelation that an i.i.d. bootstrap would destroy.

    Deliberately self-contained (numpy only): the walkforward package
    syncs to the public repo and must NOT import
    ``backtester.portfolio.calc.montecarlo`` (private-only).

    Args:
        returns: daily return series (e.g. composite OOS returns).
        n_resamples: number of bootstrap resamples.
        seed: RNG seed — pass ``WalkForwardConfig.seed`` for reproducibility.
        risk_free_rate: annual risk-free rate.
        trading_days: annualization factor.

    Returns:
        (sharpe_5pct, sharpe_95pct), or None when the series is too short
        (< 20 observations) or has zero variance.
    """
    r = returns.to_numpy(dtype=float)
    t = r.size
    if t < 20 or not np.isfinite(r).all() or float(np.std(r, ddof=1)) == 0.0:
        return None

    rng = np.random.default_rng(seed)
    expected_block = max(1, int(round(math.sqrt(t))))
    p_restart = 1.0 / expected_block
    rfr_daily = risk_free_rate / trading_days
    ann = math.sqrt(trading_days)
    pos = np.arange(t)

    sharpes = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        restarts = rng.random(t) < p_restart
        restarts[0] = True
        starts = rng.integers(0, t, size=t)
        # Index of the most recent block start for each position, then
        # walk forward from that block's random anchor (wrap-around).
        last_restart = np.maximum.accumulate(np.where(restarts, pos, -1))
        idx = (starts[last_restart] + (pos - last_restart)) % t
        sample = r[idx]
        sd = sample.std(ddof=1)
        sharpes[i] = ((sample.mean() - rfr_daily) / sd * ann) if sd > 0 else 0.0

    lo, hi = np.percentile(sharpes, [5.0, 95.0])
    return float(lo), float(hi)
