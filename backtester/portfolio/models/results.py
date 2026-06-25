"""
Result Models - Typed Bundles for Summaries

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ReturnsSummary:
    """Basic returns summary per instrument."""
    annualized_return: pd.Series
    total_return: pd.Series
    num_years: float


@dataclass(frozen=True)
class RiskSummary:
    """Risk statistics per instrument at a glance."""
    volatility_annualized: pd.Series
    var: pd.Series
    cvar: pd.Series
    max_drawdown: pd.Series
    downside_deviation: pd.Series


@dataclass(frozen=True)
class RollingBundle:
    """Common rolling windows packaged together."""
    rolling_mean: pd.DataFrame
    rolling_volatility: pd.DataFrame
    rolling_sharpe: pd.DataFrame
