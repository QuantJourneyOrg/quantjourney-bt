"""
Liquidity Analytics - Amihud, ADR, Zero-Return Days

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def amihud_illiquidity(returns: pd.Series, volume_proxy: pd.Series) -> float:
    if returns is None or volume_proxy is None or len(returns) == 0:
        return float("nan")
    r, v = returns.align(volume_proxy, join="inner")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (r.abs() / v.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return float(ratio.mean()) if len(ratio) else float("nan")


def average_daily_range(high: Optional[pd.Series], low: Optional[pd.Series], nav: Optional[pd.Series]) -> Optional[float]:
    if high is None or low is None or nav is None:
        return None
    h, l = high.align(low, join="inner")
    h, n = h.align(nav, join="inner")
    rng = (h - l).replace([np.inf, -np.inf], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        adr = (rng / n.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return float(adr.mean()) if len(adr) else None


def zero_return_days_ratio(returns: pd.Series) -> float:
    if returns is None or len(returns) == 0:
        return float("nan")
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r == 0).sum() / len(r))


def compute_liquidity_summary(
    *, returns: pd.Series, volume_proxy: pd.Series,
    high: Optional[pd.Series] = None, low: Optional[pd.Series] = None,
    nav: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    return {
        "amihud_illiquidity": amihud_illiquidity(returns, volume_proxy),
        "avg_daily_range": average_daily_range(high, low, nav),
        "zero_return_days": zero_return_days_ratio(returns),
    }
