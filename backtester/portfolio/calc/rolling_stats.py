"""
Public rolling calculations for QuantJourney instrument reports.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd


def rolling_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window=window).std()
