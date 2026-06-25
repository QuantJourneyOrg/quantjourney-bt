"""
Rolling Statistics - Mean, Volatility, Sharpe, Beta/Alpha, Correlation

When QJ_USE_NUMBA=1, Sharpe / Max-Drawdown / Beta use Numba @njit kernels
that are 5-20x faster than pandas `.rolling()` on large DataFrames.

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
except Exception:  # pragma: no cover
    njit = None  # type: ignore

USE_NUMBA = os.getenv("QJ_USE_NUMBA", "").lower() in ("1", "true", "yes") and njit is not None


# ── Numba kernels ────────────────────────────────────────────────────

def _rolling_sharpe_numpy(
    a: np.ndarray, window: int, rf_daily: float,
) -> np.ndarray:
    n, m = a.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        for i in range(window - 1, n):
            w = a[i - window + 1 : i + 1, j]
            valid = w[~np.isnan(w)]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 0.0:
                out[i, j] = (mu - rf_daily) / sigma
    return out


def _rolling_max_dd_numpy(a: np.ndarray, window: int) -> np.ndarray:
    n, m = a.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        for i in range(window - 1, n):
            w = a[i - window + 1 : i + 1, j]
            peak = w[0]
            dd = 0.0
            for k in range(len(w)):
                if w[k] > peak:
                    peak = w[k]
                drawdown = (w[k] - peak) / peak if peak != 0.0 else 0.0
                if drawdown < dd:
                    dd = drawdown
            out[i, j] = dd
    return out


def _rolling_beta_numpy(
    a: np.ndarray, bench: np.ndarray, window: int,
) -> np.ndarray:
    n, m = a.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        for i in range(window - 1, n):
            start = i - window + 1
            w_a = a[start : i + 1, j]
            w_b = bench[start : i + 1]
            mask = ~(np.isnan(w_a) | np.isnan(w_b))
            w_a = w_a[mask]
            w_b = w_b[mask]
            if len(w_a) < 2:
                continue
            b_var = np.var(w_b, ddof=1)
            if b_var > 0.0:
                cov = np.cov(w_a, w_b, ddof=1)[0, 1]
                out[i, j] = cov / b_var
    return out


if USE_NUMBA:
    @njit(cache=True)
    def _rolling_sharpe_numba(  # type: ignore
        a: np.ndarray, window: int, rf_daily: float,
    ) -> np.ndarray:
        n, m = a.shape
        out = np.empty((n, m))
        out[:] = np.nan
        for j in range(m):
            for i in range(window - 1, n):
                s = 0.0
                ss = 0.0
                cnt = 0
                for k in range(i - window + 1, i + 1):
                    v = a[k, j]
                    if np.isnan(v):
                        continue
                    s += v
                    ss += v * v
                    cnt += 1
                if cnt > 1:
                    mu = s / cnt
                    var = max((ss - s * s / cnt) / (cnt - 1), 0.0)
                    sigma = np.sqrt(var)
                    if sigma > 0.0:
                        out[i, j] = (mu - rf_daily) / sigma
        return out

    @njit(cache=True)
    def _rolling_max_dd_numba(a: np.ndarray, window: int) -> np.ndarray:  # type: ignore
        n, m = a.shape
        out = np.empty((n, m))
        out[:] = np.nan
        for j in range(m):
            for i in range(window - 1, n):
                peak = a[i - window + 1, j]
                dd = 0.0
                for k in range(i - window + 1, i + 1):
                    v = a[k, j]
                    if v > peak:
                        peak = v
                    if peak != 0.0:
                        cur = (v - peak) / peak
                        if cur < dd:
                            dd = cur
                out[i, j] = dd
        return out

    @njit(cache=True)
    def _rolling_beta_numba(  # type: ignore
        a: np.ndarray, bench: np.ndarray, window: int,
    ) -> np.ndarray:
        n, m = a.shape
        out = np.empty((n, m))
        out[:] = np.nan
        for j in range(m):
            for i in range(window - 1, n):
                sx = 0.0
                sy = 0.0
                sxx = 0.0
                sxy = 0.0
                cnt = 0
                for k in range(i - window + 1, i + 1):
                    x = a[k, j]
                    y = bench[k]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    sx += x
                    sy += y
                    sxx += x * x
                    sxy += x * y
                    cnt += 1
                if cnt > 1:
                    mx = sx / cnt
                    my = sy / cnt
                    var_y = (sy * sy - cnt * my * my) / (cnt - 1)
                    # more stable: Welford-style not needed for this size
                    var_y_alt = 0.0
                    cov_xy = 0.0
                    for k in range(i - window + 1, i + 1):
                        x = a[k, j]
                        y = bench[k]
                        if np.isnan(x) or np.isnan(y):
                            continue
                        cov_xy += (x - mx) * (y - my)
                        var_y_alt += (y - my) * (y - my)
                    if var_y_alt > 0.0:
                        out[i, j] = cov_xy / var_y_alt
        return out


# ── Public API ───────────────────────────────────────────────────────

def rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window).mean()


def rolling_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window=window).std()


def rolling_sharpe_ratio(
    returns: pd.DataFrame, *, risk_free_rate: float = 0.0,
    window: int, days_per_year: int = 252,
) -> pd.DataFrame:
    rf_daily = risk_free_rate / float(days_per_year)

    if USE_NUMBA:
        a = returns.to_numpy(dtype=np.float64)
        out = _rolling_sharpe_numba(a, window, rf_daily)
    else:
        a = returns.to_numpy(dtype=np.float64)
        out = _rolling_sharpe_numpy(a, window, rf_daily)

    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def rolling_max_drawdown(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    if USE_NUMBA:
        a = prices.to_numpy(dtype=np.float64)
        out = _rolling_max_dd_numba(a, window)
    else:
        a = prices.to_numpy(dtype=np.float64)
        out = _rolling_max_dd_numpy(a, window)

    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def rolling_beta(returns: pd.DataFrame, benchmark: pd.Series, window: int) -> pd.DataFrame:
    bench = benchmark.reindex(returns.index).to_numpy(dtype=np.float64)

    if USE_NUMBA:
        a = returns.to_numpy(dtype=np.float64)
        out = _rolling_beta_numba(a, bench, window)
    else:
        a = returns.to_numpy(dtype=np.float64)
        out = _rolling_beta_numpy(a, bench, window)

    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def rolling_alpha(
    returns: pd.DataFrame, benchmark: pd.Series, window: int,
    *, risk_free_rate: float = 0.0, days_per_year: int = 252,
) -> pd.DataFrame:
    beta = rolling_beta(returns, benchmark, window)
    bench = benchmark.reindex(returns.index)
    bench_df = pd.concat([bench] * returns.shape[1], axis=1)
    bench_df.columns = returns.columns
    alpha = (returns - beta * bench_df).rolling(window=window).mean() - risk_free_rate / float(days_per_year)
    return alpha


def rolling_calmar_ratio(
    returns: pd.DataFrame, *, window: int = 252,
    days_per_year: int = 252,
) -> pd.DataFrame:
    """Rolling Calmar ratio = annualised return / |rolling max drawdown|.

    Uses the same rolling-max-DD kernel already in this module, so it
    inherits Numba acceleration when ``QJ_USE_NUMBA=1``.
    """
    # Annualised mean return over window
    ann_return = returns.rolling(window=window).mean() * days_per_year

    # Rolling max-drawdown (negative values)
    cum = (1 + returns).cumprod()
    dd = rolling_max_drawdown(cum, window=window)   # negative or zero
    abs_dd = dd.abs().replace(0.0, np.nan)           # avoid div-by-zero

    calmar = ann_return / abs_dd
    return calmar


def rolling_correlation(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window=window).corr()
