"""
Return calculations for QuantJourney portfolio reports.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd


def compute_total_returns(returns: pd.DataFrame) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, index=returns.columns)
    return (1.0 + returns).prod() - 1.0


def compute_annualized_returns(
    returns: pd.DataFrame,
    *,
    days_per_year: int = 252,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, index=getattr(returns, "columns", None))
    total = compute_total_returns(returns)
    n = max(len(returns), 1)
    years = max(n / float(days_per_year), 1e-9)
    return (1.0 + total) ** (1.0 / years) - 1.0
