"""
Sampling Utilities - Resampling, FFill, First Non-NaN Zero

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from backtester.portfolio._compat import DataFrequency, DataDFS


def resample_prices_at_frequency(
    prices: pd.DataFrame, *, freq: Optional[str] = None,
    include_start_date: bool = False, include_end_date: bool = False,
    ffill_nans: bool = True,
) -> pd.DataFrame:
    df = prices
    if freq is not None:
        freq_norm = {"M": "ME", "Q": "QE"}.get(freq, freq)
        fill_na_method = "ffill" if ffill_nans else None
        df = DataFrequency.resample_to_frequency(
            df=df, freq=freq_norm,
            include_start_date=include_start_date,
            include_end_date=include_end_date,
            fill_na_method=fill_na_method,
        )
    else:
        if ffill_nans:
            df = df.ffill()
    return df


def ffill_prices_between_nans(prices: pd.DataFrame, method: Optional[str] = "ffill") -> pd.DataFrame:
    first_date = DataDFS.get_first_or_last_non_nan_index(df=prices, get_first=True)
    last_date = DataDFS.get_first_or_last_non_nan_index(df=prices, get_first=False)
    parts = []
    for idx, column in enumerate(prices.columns):
        good = prices.loc[first_date[idx] : last_date[idx], column]
        if method is not None:
            if method == "ffill":
                good = good.ffill()
            elif method == "bfill":
                good = good.bfill()
            else:
                good = good.fillna(method=method)
        parts.append(good)
    filled = pd.concat(parts, axis=1)
    return filled.reindex(index=prices.index)


def set_first_non_nan_returns_to_zero(
    returns: pd.DataFrame | pd.Series, init_period: int | None = 1
) -> pd.DataFrame | pd.Series:
    if init_period is not None:
        if init_period == 1:
            r = returns.copy()
            first_before = DataDFS.get_index_before_first_non_nan(df=r)
            first_date = r.index[0]
            for idx, column in zip(first_before, r.columns):
                if idx >= first_date:
                    r.loc[idx, column] = 0.0
            return r
        else:
            return returns
    return returns
