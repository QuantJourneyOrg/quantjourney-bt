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
        Optional cost-materiality filter. Implied trades below this absolute
        value remain in the quantity/trade audit but incur no modeled cost.
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
        trade_unit_values: pd.DataFrame | None = None,
    ) -> WeightCostBreakdown:
        weights = actual_weights.copy().astype(float)
        px = prices.reindex(index=weights.index, columns=weights.columns).astype(float)
        unit_values = (
            px.abs()
            if trade_unit_values is None
            else trade_unit_values.reindex(index=weights.index, columns=weights.columns).astype(
                float
            )
        )
        nav_aligned = nav.reindex(weights.index).ffill().fillna(0.0).astype(float)
        flags = rebalance_flags.reindex(weights.index).fillna(False).astype(bool)

        # Implied quantities are a reporting approximation, not executable
        # contracts. Preserve the last quantity through data gaps and use an
        # absolute mark so legal negative futures prices cannot create a
        # negative trade value (and therefore a transaction-cost credit).
        # ``trade_unit_values`` lets the common ledger supply contract-aware
        # unit notionals while direct callers retain the historical share-like
        # price convention.
        safe_unit_values = unit_values.abs().replace(0.0, np.nan).ffill()
        target_values = weights.multiply(nav_aligned, axis=0)
        target_quantities = (
            target_values.divide(safe_unit_values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        quantity_deltas = target_quantities.diff().fillna(target_quantities)
        quantity_deltas.loc[~flags, :] = 0.0
        # A missing raw mark is not a tradeable bar. Keeping the inferred
        # quantity above prevents a phantom full re-entry cost on resume.
        quantity_deltas = quantity_deltas.mask(px.isna(), 0.0)

        trade_values = (
            quantity_deltas.abs()
            .multiply(unit_values.abs())
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        costable_trade_values = trade_values
        if self.min_trade_value > 0:
            costable_trade_values = trade_values.mask(
                trade_values < self.min_trade_value,
                0.0,
            )

        transaction_costs = costable_trade_values * (float(self.total_bps) / 10_000.0)
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


def solve_recursive_weight_costs(
    *,
    actual_weights: pd.DataFrame,
    prices: pd.DataFrame,
    gross_returns: pd.Series,
    initial_capital: float,
    rebalance_flags: pd.Series,
    cost_model: WeightCostModel,
    trade_unit_values: pd.DataFrame | None = None,
    max_iterations: int = 100,
    rtol: float = 1e-12,
    atol: float = 1e-8,
) -> tuple[pd.Series, pd.Series, WeightCostBreakdown]:
    """Solve a self-financing post-cost NAV and implied-trade path.

    Target quantities depend on post-cost NAV, while costs depend on the
    resulting quantity deltas.  The fixed point is solved over the complete
    path so the returned NAV, quantities, trade values and dollar costs all
    describe one capital trajectory.
    """
    if not np.isfinite(float(initial_capital)) or float(initial_capital) <= 0.0:
        raise ValueError("initial_capital must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    weights = actual_weights.astype(float)
    index = weights.index
    aligned_prices = prices.reindex(index=index, columns=weights.columns).astype(float)
    aligned_returns = gross_returns.reindex(index).astype(float)
    if aligned_returns.isna().any() or not np.isfinite(aligned_returns.to_numpy()).all():
        raise ValueError("gross_returns must be finite on the complete weight index")
    flags = rebalance_flags.reindex(index).fillna(False).astype(bool)
    aligned_unit_values = (
        None
        if trade_unit_values is None
        else trade_unit_values.reindex(index=index, columns=weights.columns).astype(float)
    )

    candidate_nav = float(initial_capital) * (1.0 + aligned_returns).cumprod()
    candidate_nav.name = "nav"
    if len(candidate_nav) == 0:
        empty_breakdown = _compute_cost_breakdown(
            cost_model=cost_model,
            actual_weights=weights,
            prices=aligned_prices,
            nav=candidate_nav,
            rebalance_flags=flags,
            trade_unit_values=aligned_unit_values,
        )
        return candidate_nav, candidate_nav.rename("returns"), empty_breakdown

    for _ in range(max_iterations):
        breakdown = _compute_cost_breakdown(
            cost_model=cost_model,
            actual_weights=weights,
            prices=aligned_prices,
            nav=candidate_nav,
            rebalance_flags=flags,
            trade_unit_values=aligned_unit_values,
        )
        total_cost = _validated_total_cost(breakdown.total_cost, index)
        next_nav = _recursive_nav_after_costs(
            gross_returns=aligned_returns,
            total_cost=total_cost,
            initial_capital=float(initial_capital),
        )
        if np.allclose(
            next_nav.to_numpy(),
            candidate_nav.to_numpy(),
            rtol=rtol,
            atol=atol,
        ):
            candidate_nav = next_nav
            break
        candidate_nav = next_nav
    else:
        raise RuntimeError(
            "weight-cost accounting did not converge; check leverage and the cost model"
        )

    # Re-price once on the converged capital path and use those exact dollars
    # for the final NAV recurrence.
    breakdown = _compute_cost_breakdown(
        cost_model=cost_model,
        actual_weights=weights,
        prices=aligned_prices,
        nav=candidate_nav,
        rebalance_flags=flags,
        trade_unit_values=aligned_unit_values,
    )
    total_cost = _validated_total_cost(breakdown.total_cost, index)
    nav = _recursive_nav_after_costs(
        gross_returns=aligned_returns,
        total_cost=total_cost,
        initial_capital=float(initial_capital),
    ).rename("nav")
    if not np.allclose(nav.to_numpy(), candidate_nav.to_numpy(), rtol=rtol, atol=atol):
        raise RuntimeError("weight-cost accounting failed its final fixed-point reconciliation")

    previous_nav = nav.shift(1, fill_value=float(initial_capital))
    net_returns = nav.divide(previous_nav).subtract(1.0).rename("returns")
    return nav, net_returns, breakdown


def _compute_cost_breakdown(
    *,
    cost_model: WeightCostModel,
    actual_weights: pd.DataFrame,
    prices: pd.DataFrame,
    nav: pd.Series,
    rebalance_flags: pd.Series,
    trade_unit_values: pd.DataFrame | None,
) -> WeightCostBreakdown:
    if isinstance(cost_model, FixedBpsWeightCostModel):
        return cost_model.compute(
            actual_weights=actual_weights,
            prices=prices,
            nav=nav,
            rebalance_flags=rebalance_flags,
            trade_unit_values=trade_unit_values,
        )
    return cost_model.compute(
        actual_weights=actual_weights,
        prices=prices,
        nav=nav,
        rebalance_flags=rebalance_flags,
    )


def _validated_total_cost(costs: pd.Series, index: pd.Index) -> pd.Series:
    aligned = costs.reindex(index).astype(float)
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("weight cost model must return finite, non-negative total costs")
    return aligned


def _recursive_nav_after_costs(
    *,
    gross_returns: pd.Series,
    total_cost: pd.Series,
    initial_capital: float,
) -> pd.Series:
    nav_values = np.empty(len(gross_returns), dtype=float)
    previous_nav = float(initial_capital)
    for position, (gross_return, cost) in enumerate(
        zip(gross_returns.to_numpy(dtype=float), total_cost.to_numpy(dtype=float), strict=True)
    ):
        nav_before_cost = previous_nav * (1.0 + gross_return)
        nav_after_cost = nav_before_cost - cost
        if not np.isfinite(nav_after_cost) or nav_after_cost <= 0.0:
            raise ValueError(
                "weight-mode NAV became non-positive; check leverage, returns, and costs"
            )
        nav_values[position] = nav_after_cost
        previous_nav = nav_after_cost
    return pd.Series(nav_values, index=gross_returns.index, dtype=float, name="nav")
