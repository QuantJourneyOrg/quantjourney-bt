"""
Return Analytics - Periodic/Log/Relative, NAV, Annualization

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

from backtester.portfolio._compat import ReturnTypes, DataFrequency


def compute_periodic_returns(
    prices: pd.DataFrame,
    *,
    is_log_returns: bool = False,
    return_type: ReturnTypes = ReturnTypes.RELATIVE,
    freq: Optional[str] = None,
    include_start_date: bool = False,
    include_end_date: bool = False,
    ffill_nans: bool = True,
    drop_first: bool = False,
    is_first_zero: bool = False,
) -> pd.DataFrame:
    df = prices
    if freq is not None:
        fill_na_method = "ffill" if ffill_nans else None
        df = DataFrequency.resample_to_frequency(
            df=df, freq=freq,
            include_start_date=include_start_date,
            include_end_date=include_end_date,
            fill_na_method=fill_na_method,
        )
    else:
        if ffill_nans:
            df = df.ffill()

    if return_type == ReturnTypes.LOG or is_log_returns:
        rets = np.log(df) - np.log(df.shift(1))
    elif return_type == ReturnTypes.RELATIVE:
        rets = df.divide(df.shift(1)).subtract(1.0)
    elif return_type == ReturnTypes.DIFFERENCE:
        rets = df.subtract(df.shift(1))
    elif return_type == ReturnTypes.LEVEL:
        rets = df
    elif return_type == ReturnTypes.LEVEL0:
        rets = df.shift(1)
    else:
        raise NotImplementedError(f"Unsupported return type: {return_type}")

    if is_first_zero and not rets.empty:
        rets.iloc[0, :] = 0.0
    elif drop_first:
        rets = rets.iloc[1:, :]

    return rets


def compute_total_returns(returns: pd.DataFrame) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, index=returns.columns)
    return (1.0 + returns).prod() - 1.0


def compute_annualized_returns(
    returns: pd.DataFrame, *, days_per_year: int = 252,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, index=getattr(returns, "columns", None))
    total = compute_total_returns(returns)
    # Use number of trading observations, not calendar days.
    # Calendar days / trading days (e.g. 365/252) overstates the holding
    # period and understates annualised returns by ~30%.
    n = max(len(returns), 1)
    years = max(n / float(days_per_year), 1e-9)
    return (1.0 + total) ** (1.0 / years) - 1.0


def convert_returns_to_nav(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    *,
    init_period: Optional[int] = 1,
    terminal_value: Optional[np.ndarray] = None,
    init_value: Optional[np.ndarray | float] = None,
    freq: Optional[str] = None,
    ffill_between_nans: bool = True,
    constant_trade_level: bool = False,
) -> pd.DataFrame:
    df = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(returns)
    if init_period is not None:
        df = df.copy()
        df.iloc[:init_period] = 0.0

    if constant_trade_level:
        nav = df.cumsum(skipna=True, axis=0).add(1.0)
    else:
        nav = (1.0 + df).cumprod(skipna=True, axis=0)

    if terminal_value is not None:
        last = nav.ffill().iloc[-1]
        nav = nav * (terminal_value / last)
    elif init_value is not None:
        first = nav.ffill().iloc[0]
        nav = nav * (init_value / first)

    if freq is not None:
        nav = nav.asfreq(freq, method="ffill").ffill()
    if ffill_between_nans:
        nav = nav.ffill()

    return nav


def convert_log_returns_to_nav(
    log_returns: pd.DataFrame | pd.Series | np.ndarray,
    *,
    init_period: Optional[int] = None,
    terminal_value: Optional[np.ndarray] = None,
    init_value: Optional[np.ndarray | float] = None,
) -> pd.DataFrame:
    df = log_returns if isinstance(log_returns, pd.DataFrame) else pd.DataFrame(log_returns)
    if init_period is not None:
        df = df.copy()
        df.iloc[:init_period] = 0.0
    nav = np.exp(np.nancumsum(df, axis=0))
    if isinstance(nav, np.ndarray):
        nav = pd.DataFrame(nav, index=df.index, columns=df.columns)
    if terminal_value is not None:
        nav = nav * (terminal_value / nav.iloc[-1])
    elif init_value is not None:
        nav = nav * (init_value / nav.iloc[0])
    return nav
