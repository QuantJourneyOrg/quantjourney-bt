"""
Metrics Utilities - Convenience Metrics and Small Aggregations

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def excess_returns(returns: pd.DataFrame, benchmark_returns: pd.DataFrame | pd.Series) -> pd.DataFrame:
    bench = benchmark_returns.reindex(returns.index)
    return returns.subtract(bench, axis=0)


def annualized_volatility(returns: pd.DataFrame, *, days_per_year: int = 252) -> pd.Series:
    return returns.std() * np.sqrt(days_per_year)


def active_return(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    bench = benchmark_returns.reindex(returns.index)
    return returns.mean() - bench.mean()


def information_coefficient(
    returns: pd.DataFrame | pd.Series, predicted_returns: pd.DataFrame | pd.Series,
) -> float:
    """Correlation between realised and predicted returns.

    NaN policy: pairwise drop missing aligned observations. Missing
    predictions are not treated as zero. DataFrame inputs are aligned by both
    date index and column labels before flattening.
    """
    if isinstance(returns, pd.DataFrame) and isinstance(predicted_returns, pd.DataFrame):
        r_al, p_al = returns.align(predicted_returns, join="inner", axis=None)
        r_vals = r_al.to_numpy(dtype=float).ravel()
        p_vals = p_al.to_numpy(dtype=float).ravel()
    elif isinstance(returns, pd.Series) and isinstance(predicted_returns, pd.Series):
        r_al, p_al = returns.align(predicted_returns, join="inner")
        r_vals = r_al.to_numpy(dtype=float)
        p_vals = p_al.to_numpy(dtype=float)
    else:
        r_df = returns if isinstance(returns, pd.DataFrame) else returns.to_frame(name="ret")
        p_df = predicted_returns if isinstance(predicted_returns, pd.DataFrame) else predicted_returns.to_frame(name="pred")
        r_ser = r_df.mean(axis=1)
        p_ser = p_df.mean(axis=1)
        r_al, p_al = r_ser.align(p_ser, join="inner")
        r_vals = r_al.to_numpy(dtype=float)
        p_vals = p_al.to_numpy(dtype=float)

    valid = np.isfinite(r_vals) & np.isfinite(p_vals)
    r_vals = r_vals[valid]
    p_vals = p_vals[valid]

    if len(r_vals) < 2:
        return float("nan")
    if np.std(r_vals) == 0 or np.std(p_vals) == 0:
        return float("nan")
    return float(np.corrcoef(r_vals, p_vals)[0, 1])


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def return_persistence(returns: pd.Series, *, ignore_zero: bool = True) -> float:
    """Fraction of consecutive non-missing returns with the same sign."""
    if returns is None or len(returns) == 0:
        return float("nan")
    r = returns.dropna()
    if ignore_zero:
        r = r[r != 0]
    if len(r) < 2:
        return float("nan")
    signs = pd.Series(np.sign(r.to_numpy(dtype=float)), index=r.index)
    same_direction = signs.iloc[1:] == signs.shift(1).iloc[1:]
    if same_direction.empty:
        return float("nan")
    return float(same_direction.mean())


def lag_sensitivity_summary(returns: pd.Series, window: int = 252) -> dict[str, float]:
    """Rolling lag-1 return correlation summary.

    This is not market beta. Use ``market_beta_summary`` or
    ``rolling_stats.rolling_beta`` for benchmark-relative beta.
    """
    if returns is None or len(returns) == 0:
        return {"avg_lag_corr": float("nan"), "lag_corr_vol": float("nan")}
    r = returns.dropna()
    if len(r) < window:
        return {"avg_lag_corr": float("nan"), "lag_corr_vol": float("nan")}
    lag_corr = r.rolling(window=window).corr(r.shift(1))
    return {
        "avg_lag_corr": float(lag_corr.mean()),
        "lag_corr_vol": float(lag_corr.std()),
    }


def market_sensitivity_summary(returns: pd.Series, window: int = 252) -> dict[str, float]:
    """Deprecated alias for lag-sensitivity summary.

    The returned keys are kept for backwards compatibility, but the values are
    lag correlations, not benchmark beta.
    """
    warnings.warn(
        "market_sensitivity_summary() is deprecated; use "
        "lag_sensitivity_summary(). This function is not benchmark beta.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = lag_sensitivity_summary(returns, window=window)
    return {
        "avg_beta": result["avg_lag_corr"],
        "beta_vol": result["lag_corr_vol"],
    }


def market_beta_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> dict[str, float]:
    """Rolling benchmark-beta summary."""
    if returns is None or benchmark_returns is None:
        return {"avg_beta": float("nan"), "beta_vol": float("nan")}

    r, b = returns.align(benchmark_returns, join="inner")
    valid = r.notna() & b.notna()
    r = r.loc[valid]
    b = b.loc[valid]
    if len(r) < window:
        return {"avg_beta": float("nan"), "beta_vol": float("nan")}

    rolling_cov = r.rolling(window=window).cov(b)
    rolling_var = b.rolling(window=window).var()
    beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan).dropna()
    if beta.empty:
        return {"avg_beta": float("nan"), "beta_vol": float("nan")}

    return {
        "avg_beta": float(beta.mean()),
        "beta_vol": float(beta.std()),
    }


def left_tail_summary(returns: pd.Series, quantile: float = 0.05) -> dict[str, float]:
    """Single-series left-tail return diagnostics."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if returns is None or len(returns) == 0:
        return {
            "tail_mean": float("nan"),
            "tail_vol": float("nan"),
            "tail_min": float("nan"),
            "tail_count": 0,
            "tail_threshold": float("nan"),
        }
    r = returns.dropna()
    if len(r) < 3:
        return {
            "tail_mean": float("nan"),
            "tail_vol": float("nan"),
            "tail_min": float("nan"),
            "tail_count": 0,
            "tail_threshold": float("nan"),
        }

    threshold = r.quantile(quantile)
    tail = r[r <= threshold]
    return {
        "tail_mean": float(tail.mean()) if len(tail) else float("nan"),
        "tail_vol": float(tail.std()) if len(tail) else float("nan"),
        "tail_min": float(tail.min()) if len(tail) else float("nan"),
        "tail_count": int(len(tail)),
        "tail_threshold": float(threshold),
    }


def tail_dependence(
    returns: pd.Series,
    reference_returns: pd.Series | None = None,
    quantile: float = 0.05,
) -> dict[str, float]:
    """Left-tail dependence of returns relative to a reference series.

    If ``reference_returns`` is omitted, this falls back to ``left_tail_summary``
    for backwards compatibility and returns NaN dependence fields.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")

    if reference_returns is None:
        warnings.warn(
            "tail_dependence() without reference_returns is deprecated; use "
            "left_tail_summary() for single-series diagnostics or pass a "
            "reference series for true tail dependence.",
            DeprecationWarning,
            stacklevel=2,
        )
        summary = left_tail_summary(returns, quantile=quantile)
        return {
            "tail_beta": float("nan"),
            "tail_correlation": float("nan"),
            **summary,
        }

    if returns is None or reference_returns is None:
        return {
            "tail_beta": float("nan"),
            "tail_correlation": float("nan"),
            "tail_count": 0,
            "tail_threshold": float("nan"),
        }

    r, ref = returns.align(reference_returns, join="inner")
    valid = r.notna() & ref.notna()
    r = r.loc[valid]
    ref = ref.loc[valid]

    if len(r) < 3:
        return {
            "tail_beta": float("nan"),
            "tail_correlation": float("nan"),
            "tail_count": 0,
            "tail_threshold": float("nan"),
        }

    threshold = ref.quantile(quantile)
    mask = ref <= threshold
    tail_r = r.loc[mask]
    tail_ref = ref.loc[mask]

    if len(tail_r) < 2 or tail_ref.var() == 0 or tail_r.std() == 0:
        return {
            "tail_beta": float("nan"),
            "tail_correlation": float("nan"),
            "tail_count": int(len(tail_r)),
            "tail_threshold": float(threshold),
        }

    tail_beta = tail_r.cov(tail_ref) / tail_ref.var()
    tail_corr = tail_r.corr(tail_ref)
    return {
        "tail_beta": float(tail_beta),
        "tail_correlation": float(tail_corr),
        "tail_count": int(len(tail_r)),
        "tail_threshold": float(threshold),
    }
