"""
Metric helpers for QuantJourney instrument reports.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()
