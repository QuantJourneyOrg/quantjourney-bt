"""
Exposure Analytics - Market Value, Long/Short, Turnover, Participation

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd


def compute_exposures(prices_adj_close: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    units_aligned = units.reindex(columns=prices_adj_close.columns)
    values = units_aligned.to_numpy() * prices_adj_close.to_numpy()
    return pd.DataFrame(values, index=prices_adj_close.index, columns=prices_adj_close.columns)


def compute_short_long_exposure(prices_adj_close: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    long_exposure = (units.where(units > 0, 0.0)) * prices_adj_close
    short_exposure = (units.where(units < 0, 0.0)) * prices_adj_close
    return pd.DataFrame({"Long": long_exposure.sum(axis=1), "Short": short_exposure.sum(axis=1)})


def compute_turnover(
    units: pd.DataFrame,
    instruments: list[str],
    add_total: bool = False,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute daily turnover per instrument.

    When *prices* is provided the result is **dollar turnover** per day
    (``abs(delta_units) × price``), which is the standard institutional
    definition.  Without prices the function falls back to a binary
    indicator (position changed yes/no) for backward compatibility.
    """
    if isinstance(units.columns, pd.MultiIndex):
        instrument_cols = [col for col in units.columns if col[0] in instruments]
        units = units[instrument_cols]
    numeric_units = units.select_dtypes(include=[float, int])
    position_changes = numeric_units.diff().abs()

    if prices is not None:
        # Dollar turnover: |delta_units| × price
        price_al = prices.reindex(columns=numeric_units.columns, index=numeric_units.index)
        turnover = position_changes * price_al.ffill()
    else:
        # Legacy binary indicator
        turnover = (position_changes != 0).astype(float)

    if add_total:
        turnover["Total"] = turnover.sum(axis=1)
    return turnover


def market_cap_participation(
    units: pd.DataFrame, prices: pd.DataFrame, market_cap: pd.DataFrame,
    *, trade_value: float = 100_000_000,
) -> pd.DataFrame:
    exposure = units * prices
    market_cap_al = market_cap.reindex(exposure.index)
    return trade_value * exposure.divide(market_cap_al)


def volume_participation(
    traded_units: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    trade_value: float | None = None,
) -> pd.DataFrame:
    """Trade participation rate: traded shares/contracts divided by volume."""
    to = traded_units.apply(pd.to_numeric, errors="coerce")
    vol = volumes.apply(pd.to_numeric, errors="coerce")
    return to / vol
