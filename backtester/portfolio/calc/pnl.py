"""
PnL Calculations - Returns x Units, Cumulative Contributions

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd


def compute_pnl(returns: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    return (returns * units.shift(1)).fillna(0.0)


def compute_cumulative_pnl(returns: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    return compute_pnl(returns, units).cumsum().fillna(0.0)


def compute_cumulative_returns_from_units(returns: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    return compute_cumulative_pnl(returns, units)
