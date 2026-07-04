"""
Risk calculations for QuantJourney portfolio reports.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdowns(returns: pd.DataFrame) -> pd.DataFrame:
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    return (nav - peak) / peak


def compute_max_drawdown(returns: pd.DataFrame) -> pd.Series:
    return compute_drawdowns(returns).min()


def sharpe_ratio(
    returns: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    days_per_year: int = 252,
    annualize: bool = True,
) -> pd.Series:
    rf_daily = risk_free_rate / float(days_per_year)
    excess = returns - rf_daily
    ratio = excess.mean() / excess.std()
    if annualize:
        ratio = ratio * np.sqrt(days_per_year)
    return ratio
