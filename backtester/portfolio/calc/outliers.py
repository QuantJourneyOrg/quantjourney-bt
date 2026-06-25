"""
Outlier Utilities

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd

from backtester.portfolio._compat import (
    outliers as qs_outliers,
    remove_outliers as qs_remove_outliers,
)


def outliers(returns: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    return qs_outliers(returns, quantile)


def remove_outliers(returns: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    return qs_remove_outliers(returns, quantile)
