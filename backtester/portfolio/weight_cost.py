"""
Weight-mode transaction cost models.

Weight-based strategies do not submit explicit orders, but a rebalance still
implies trades.  This module converts target portfolio weights into implied
share deltas, then computes costs from the resulting trade values.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightCostBreakdown:
    """Detailed transaction-cost output for weight-mode portfolio accounting."""

    quantity_deltas: pd.DataFrame
    trade_values: pd.DataFrame
    transaction_costs: pd.DataFrame
    total_cost: pd.Series
    total_cost_pct: pd.Series


@runtime_checkable
class WeightCostModel(Protocol):
    """Protocol for weight-mode cost models."""

    def compute(
        self,
        *,
        actual_weights: pd.DataFrame,
        prices: pd.DataFrame,
        nav: pd.Series,
        rebalance_flags: pd.Series,
    ) -> WeightCostBreakdown:
        """Return transaction costs implied by target weights and prices."""


@dataclass(frozen=True, slots=True)
class FixedBpsWeightCostModel:
    """
    Fixed-bps cost model for implied weight-mode trades.

    Parameters
    ----------
    total_bps:
        Total round-trip-independent cost in basis points applied to each
        implied trade value.  ``1.0`` means 1 bp per buy/sell notional.
    min_trade_value:
        Optional dust filter.  Implied trades below this absolute value are
        ignored before costs are computed.
    """

    total_bps: float = 1.0
    min_trade_value: float = 0.0

    def compute(
        self,
        *,
        actual_weights: pd.DataFrame,
        prices: pd.DataFrame,
        nav: pd.Series,
        rebalance_flags: pd.Series,
    ) -> WeightCostBreakdown:
        weights = actual_weights.copy().astype(float)
        px = prices.reindex(index=weights.index, columns=weights.columns).astype(float)
        nav_aligned = nav.reindex(weights.index).ffill().fillna(0.0).astype(float)
        flags = rebalance_flags.reindex(weights.index).fillna(False).astype(bool)

        # Implied quantities are a reporting approximation, not executable
        # contracts. Preserve the last quantity through data gaps and use an
        # absolute mark so legal negative futures prices cannot create a
        # negative trade value (and therefore a transaction-cost credit).
        safe_prices = px.abs().replace(0.0, np.nan).ffill()
        target_values = weights.multiply(nav_aligned, axis=0)
        target_quantities = (
            target_values.divide(safe_prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        quantity_deltas = target_quantities.diff().fillna(target_quantities)
        quantity_deltas.loc[~flags, :] = 0.0
        # A missing raw mark is not a tradeable bar. Keeping the inferred
        # quantity above prevents a phantom full re-entry cost on resume.
        quantity_deltas = quantity_deltas.mask(px.isna(), 0.0)

        trade_values = (
            quantity_deltas.abs().multiply(px.abs()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        if self.min_trade_value > 0:
            trade_values = trade_values.mask(trade_values < self.min_trade_value, 0.0)
            quantity_deltas = quantity_deltas.mask(trade_values == 0.0, 0.0)

        transaction_costs = trade_values * (float(self.total_bps) / 10_000.0)
        total_cost = transaction_costs.sum(axis=1)
        total_cost.name = "transaction_cost"

        nav_safe = nav_aligned.replace(0.0, np.nan)
        total_cost_pct = total_cost.divide(nav_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        total_cost_pct.name = "transaction_cost_pct"

        return WeightCostBreakdown(
            quantity_deltas=quantity_deltas,
            trade_values=trade_values,
            transaction_costs=transaction_costs,
            total_cost=total_cost,
            total_cost_pct=total_cost_pct,
        )
