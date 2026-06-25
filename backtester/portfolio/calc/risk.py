"""
Risk Analytics - Volatility, Drawdowns, VaR/CVaR/ES, Ratios

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backtester.portfolio._compat import (
    ulcer_index as qs_ulcer_index,
    serenity_index as qs_serenity_index,
    risk_of_ruin as qs_risk_of_ruin,
)


def compute_volatility(
    returns: pd.DataFrame, window: int = 252, *, days_per_year: int = 252,
) -> pd.DataFrame:
    return returns.rolling(window=window).std() * np.sqrt(days_per_year)


def compute_drawdowns(returns: pd.DataFrame) -> pd.DataFrame:
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    return (nav - peak) / peak


def compute_max_drawdown(returns: pd.DataFrame) -> pd.Series:
    dd = compute_drawdowns(returns)
    return dd.min()


def compute_var(returns: pd.DataFrame, confidence: float = 0.95) -> pd.Series:
    return returns.quantile(1 - confidence)


def compute_cvar(returns: pd.DataFrame, confidence: float = 0.95) -> pd.Series:
    var = compute_var(returns, confidence)
    return returns[returns.le(var)].mean()


def compute_expected_shortfall(returns: pd.DataFrame, confidence: float = 0.95) -> pd.Series:
    return compute_cvar(returns, confidence)


def downside_deviation(returns: pd.DataFrame, target_return: float = 0.0) -> pd.Series:
    downside = returns.where(returns < target_return)
    downside = downside.fillna(0.0)
    return np.sqrt((downside.pow(2)).mean())


def sharpe_ratio(
    returns: pd.DataFrame, *, risk_free_rate: float = 0.0,
    days_per_year: int = 252, annualize: bool = True,
) -> pd.Series:
    rf_daily = risk_free_rate / float(days_per_year)
    excess = returns - rf_daily
    base = excess.mean() / excess.std()
    if annualize:
        base = base * np.sqrt(days_per_year)
    return base


def sortino_ratio(
    returns: pd.DataFrame, *, risk_free_rate: float = 0.0,
    target_return: float = 0.0, days_per_year: int = 252, annualize: bool = True,
) -> pd.Series:
    rf_daily = risk_free_rate / float(days_per_year)
    excess = returns - rf_daily
    dd = downside_deviation(returns, target_return=target_return)
    ratio = excess.mean() / dd.replace(0.0, np.nan)
    if annualize:
        ratio = ratio * np.sqrt(days_per_year)
    return ratio


def information_ratio(
    returns: pd.DataFrame, benchmark_returns: pd.Series, *, days_per_year: int = 252,
) -> pd.Series:
    bench = benchmark_returns.reindex(returns.index)
    active = returns.sub(bench, axis=0)
    return active.mean() / active.std() * np.sqrt(days_per_year)


def calmar_ratio(returns: pd.DataFrame, *, days_per_year: int = 252) -> pd.Series:
    from .returns import compute_annualized_returns
    ann = compute_annualized_returns(returns, days_per_year=days_per_year)
    mdd = compute_max_drawdown(returns).abs().replace(0.0, np.nan)
    return ann / mdd


def omega_ratio(returns: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    above = returns.where(returns > threshold, 0.0).sum()
    below = returns.where(returns <= threshold, 0.0).sum().abs().replace(0.0, np.nan)
    return above / below


def ulcer_index(returns: pd.DataFrame) -> pd.Series:
    return returns.apply(qs_ulcer_index)


def serenity_index(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    return returns.apply(lambda x: qs_serenity_index(x, rf))


def gain_to_pain_ratio(returns: pd.DataFrame) -> pd.Series:
    gains = returns.where(returns > 0, 0.0).sum()
    pain = returns.where(returns < 0, 0.0).sum().abs().replace(0.0, np.nan)
    return gains / pain


def upside_potential_ratio(returns: pd.DataFrame, target_return: float = 0.0) -> pd.Series:
    upside = returns.where(returns > target_return, 0.0)
    up_mean = upside.mean()
    dd = downside_deviation(returns, target_return=target_return)
    safe_dd = dd.replace(0.0, np.nan)
    return up_mean / safe_dd


def risk_of_ruin(returns: pd.DataFrame) -> pd.Series:
    return qs_risk_of_ruin(returns)


def smart_sharpe_ratio(
    returns: pd.DataFrame, *, risk_free_rate: float = 0.0,
    days_per_year: int = 252, trim_frac: float = 0.02,
) -> pd.Series:
    rf_daily = risk_free_rate / float(days_per_year)
    excess = returns - rf_daily
    ex = excess.stack().sort_values()
    n = len(ex)
    if n == 0:
        return pd.Series(dtype=float)
    k = int(n * trim_frac)
    ex_t = ex.iloc[k : n - k] if n - 2 * k > 0 else ex
    ex_df = ex_t.unstack()
    mu = ex_df.mean()
    sd = ex_df.std().replace(0.0, np.nan)
    return mu / sd * np.sqrt(days_per_year)


def smart_sortino_ratio(
    returns: pd.DataFrame, *, risk_free_rate: float = 0.0,
    target_return: float = 0.0, days_per_year: int = 252, trim_frac: float = 0.02,
) -> pd.Series:
    rf_daily = risk_free_rate / float(days_per_year)
    excess = returns - rf_daily
    downside = excess.where(excess < target_return).stack().sort_values()
    n = len(downside)
    k = int(n * trim_frac)
    d_t = downside.iloc[k : n - k] if n - 2 * k > 0 else downside
    d_df = d_t.unstack().fillna(0.0)
    dd = np.sqrt((d_df.pow(2)).mean()).replace(0.0, np.nan)
    mu = excess.mean()
    return mu / dd * np.sqrt(days_per_year)


def smart_calmar_ratio(
    returns: pd.DataFrame, *, days_per_year: int = 252, trim_frac: float = 0.02,
) -> pd.Series:
    from .returns import compute_annualized_returns
    ann = compute_annualized_returns(returns, days_per_year=days_per_year)
    dd = compute_drawdowns(returns).stack().sort_values()
    n = len(dd)
    k = int(n * trim_frac)
    dd_t = dd.iloc[k : n - k] if n - 2 * k > 0 else dd
    dd_min = dd_t.unstack().min().abs().replace(0.0, np.nan)
    return ann / dd_min


def pain_index(returns: pd.DataFrame) -> pd.Series:
    dd = compute_drawdowns(returns)
    return dd.abs().mean()


def max_drawdown_duration(returns: pd.DataFrame) -> pd.Series:
    dd = compute_drawdowns(returns)
    durations = pd.Series(index=dd.columns, dtype=float)
    for col in dd.columns:
        series = dd[col]
        end_dd = series.idxmin()
        start_dd = series[:end_dd][series == 0].index[-1] if any(series[:end_dd] == 0) else series.index[0]
        durations[col] = (end_dd - start_dd).days
    return durations


def conditional_drawdown_at_risk(returns: pd.DataFrame, confidence: float = 0.95) -> float:
    dd = compute_drawdowns(returns)
    vals = dd.to_numpy().ravel()
    if vals.size == 0 or np.all(np.isnan(vals)):
        return 0.0
    threshold = np.nanpercentile(vals, 100 * (1 - confidence))
    mask = dd <= threshold
    if mask.sum().sum() == 0:
        return 0.0
    return float(dd[mask].mean().mean())


def sampled_volatility(
    returns: pd.DataFrame, *, freq_vol: str = "M",
    freq_return: str | None = None, days_per_year: int = 252,
) -> pd.DataFrame:
    r = returns.copy()
    freq_vol_norm = {"M": "ME", "Q": "QE"}.get(freq_vol, freq_vol)
    freq_ret_norm = {"M": "ME", "Q": "QE"}.get(freq_return, freq_return) if freq_return else None
    if freq_ret_norm:
        r = (1 + r).resample(freq_ret_norm).prod() - 1
    vol_samples = r.groupby(pd.Grouper(freq=freq_vol_norm)).std()
    return vol_samples * np.sqrt(days_per_year)
