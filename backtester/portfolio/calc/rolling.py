"""
Rolling Operations - Windowed Reductions (Numba Optional)

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:
    njit = None  # type: ignore

USE_NUMBA = os.getenv("QJ_USE_NUMBA", "").lower() in ("1", "true", "yes") and njit is not None


def _rolling_std_numpy(a: np.ndarray, window: int) -> np.ndarray:
    n, m = a.shape
    out = np.full_like(a, np.nan, dtype=float)
    for j in range(m):
        col = a[:, j]
        for i in range(window - 1, n):
            w = col[i - window + 1 : i + 1]
            out[i, j] = np.nanstd(w, ddof=1)  # sample std, consistent with pandas
    return out


if USE_NUMBA:
    @njit(cache=True)  # type: ignore
    def _rolling_std_numba(a: np.ndarray, window: int) -> np.ndarray:
        n, m = a.shape
        out = np.empty((n, m))
        out[:] = np.nan
        for j in range(m):
            for i in range(window - 1, n):
                s = 0.0
                ss = 0.0
                count = 0
                for k in range(i - window + 1, i + 1):
                    v = a[k, j]
                    if np.isnan(v):
                        continue
                    s += v
                    ss += v * v
                    count += 1
                if count > 1:
                    mean = s / count
                    var = max((ss - s * s / count) / (count - 1), 0.0)  # ddof=1
                    out[i, j] = np.sqrt(var)
                else:
                    out[i, j] = np.nan
        return out


def rolling_std_df(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    a = df.to_numpy(dtype=float)
    if USE_NUMBA:
        out = _rolling_std_numba(a, window)
    else:
        out = _rolling_std_numpy(a, window)
    return pd.DataFrame(out, index=df.index, columns=df.columns)
