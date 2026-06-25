"""
Multi-Asset PnL Calculations.

Provides asset-class-aware PnL computation that correctly handles:
  - Equities:  shares × Δprice
  - Futures:   contracts × Δprice × multiplier
  - FX:        lots × Δprice × lot_size
  - Crypto:    units × Δprice  (fractional lots, optional inverse)

All functions accept a ContractSpec (or dict of specs keyed by symbol)
so the backtester can compute PnL for mixed portfolios.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from backtester.execution.contract_spec import AssetClass, ContractSpec, get_contract_spec


def compute_position_pnl(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    specs: Optional[Dict[str, ContractSpec]] = None,
) -> pd.DataFrame:
    """
    Daily mark-to-market PnL for each instrument.

    PnL_t = position_{t-1} × (price_t - price_{t-1}) × multiplier × lot_size

    Parameters
    ----------
    positions : DataFrame (dates × instruments)
        Number of contracts/shares held at each date.
    prices : DataFrame (dates × instruments)
        Close prices for each date.
    specs : dict mapping symbol → ContractSpec
        If None, all instruments treated as equity (multiplier=1).

    Returns
    -------
    DataFrame (dates × instruments) of daily PnL in quote currency.
    """
    if specs is None:
        specs = {}

    price_change = prices.diff().fillna(0.0)
    lagged_pos = positions.shift(1).fillna(0.0)

    pnl = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)

    for col in positions.columns:
        spec = specs.get(col, get_contract_spec(col))

        if spec.inverse:
            # Inverse contract: PnL = position × multiplier × (1/p_{t-1} - 1/p_t)
            p_prev = prices[col].shift(1)
            p_curr = prices[col]
            inv_diff = (1.0 / p_prev - 1.0 / p_curr).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            pnl[col] = lagged_pos[col] * spec.multiplier * inv_diff
        else:
            pnl[col] = lagged_pos[col] * price_change[col] * spec.multiplier * spec.lot_size

    return pnl


def compute_portfolio_pnl(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    specs: Optional[Dict[str, ContractSpec]] = None,
) -> pd.Series:
    """
    Aggregate daily portfolio PnL across all instruments.

    Returns a Series indexed by date.
    """
    instrument_pnl = compute_position_pnl(positions, prices, specs)
    return instrument_pnl.sum(axis=1)


def compute_margin_usage(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    specs: Optional[Dict[str, ContractSpec]] = None,
) -> pd.DataFrame:
    """
    Daily margin requirement per instrument.

    For margin-based instruments (futures, FX): |quantity| × margin_per_contract
    For fully-funded (equity, crypto):          |quantity| × price × multiplier
    """
    if specs is None:
        specs = {}

    margin = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)

    for col in positions.columns:
        spec = specs.get(col, get_contract_spec(col))
        margin[col] = positions[col].apply(
            lambda q, s=spec, p=prices[col]: s.margin_required(q, p.loc[q.name] if hasattr(q, 'name') else 0)
            if callable(getattr(spec, 'margin_required', None))
            else abs(q) * spec.margin
        )
        # Vectorized version
        if spec.margin > 0:
            margin[col] = positions[col].abs() * spec.margin
        else:
            margin[col] = positions[col].abs() * prices[col] * spec.multiplier * spec.lot_size

    return margin


def compute_total_margin(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    specs: Optional[Dict[str, ContractSpec]] = None,
) -> pd.Series:
    """Total margin requirement across all instruments. Returns Series indexed by date."""
    return compute_margin_usage(positions, prices, specs).sum(axis=1)


def compute_notional_exposure(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    specs: Optional[Dict[str, ContractSpec]] = None,
) -> pd.DataFrame:
    """
    Notional exposure for each instrument.

    notional = |quantity| × price × multiplier × lot_size
    """
    if specs is None:
        specs = {}

    exposure = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)

    for col in positions.columns:
        spec = specs.get(col, get_contract_spec(col))
        if spec.inverse:
            exposure[col] = positions[col].abs() * spec.multiplier / prices[col].replace(0, np.nan)
        else:
            exposure[col] = positions[col].abs() * prices[col] * spec.multiplier * spec.lot_size

    return exposure.fillna(0.0)


def compute_returns_from_pnl(
    pnl: pd.Series,
    capital: float,
) -> pd.Series:
    """
    Convert PnL series to returns based on starting capital.

    For futures / multi-asset, returns = PnL / capital is more appropriate
    than nav.pct_change() because margin ≠ notional.
    """
    return pnl / capital


def compute_nav_from_pnl(
    pnl: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Cumulative NAV from PnL series: NAV_t = initial_capital + Σ PnL."""
    return initial_capital + pnl.cumsum()
