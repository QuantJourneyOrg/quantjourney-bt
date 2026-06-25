"""
Slippage Models — pluggable fill-price adjustments.

All models implement the ``SlippageModel`` protocol:

    def compute(self, price: float, quantity: float, side: OrderSide,
                bar: BarData) -> float:
        '''Return the slipped fill price.'''

Available models:
    NoSlippage         — fill at theoretical price (default)
    FixedBpsSlippage   — constant basis-point spread
    VolatilitySlippage — spread proportional to bar range / close
    MarketImpactSlippage — Almgren-style sqrt(participation) model

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backtester.execution.order_types import BarData, OrderSide


# ── Protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class SlippageModel(Protocol):
    """Protocol for slippage models."""

    def compute(
        self,
        price: float,
        quantity: float,
        side: OrderSide,
        bar: BarData,
    ) -> float:
        """Return adjusted fill price after slippage."""
        ...


# ── Implementations ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class NoSlippage:
    """No slippage — fill at theoretical price."""

    def compute(
        self, price: float, quantity: float, side: OrderSide, bar: BarData,
    ) -> float:
        return price


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage:
    """
    Constant basis-point slippage.

    Args:
        bps: half-spread in basis points (e.g. 5.0 = 0.05% one-way).
             Buys pay price * (1 + bps/10_000), sells receive price * (1 - bps/10_000).
    """

    bps: float = 5.0

    def compute(
        self, price: float, quantity: float, side: OrderSide, bar: BarData,
    ) -> float:
        spread = price * self.bps / 10_000
        return price + spread if side == OrderSide.BUY else price - spread


@dataclass(frozen=True, slots=True)
class VolatilitySlippage:
    """
    Slippage proportional to intra-bar volatility.

    spread = price × vol_factor × (high - low) / close

    Useful for illiquid instruments where spread widens with volatility.
    """

    vol_factor: float = 0.1

    def compute(
        self, price: float, quantity: float, side: OrderSide, bar: BarData,
    ) -> float:
        if bar.close == 0:
            return price
        bar_range = (bar.high - bar.low) / bar.close
        spread = price * self.vol_factor * bar_range
        return price + spread if side == OrderSide.BUY else price - spread


@dataclass(frozen=True, slots=True)
class MarketImpactSlippage:
    """
    Almgren-style market impact: cost ∝ σ × √(Q / ADV).

    Args:
        sigma_daily: estimated daily volatility (e.g. 0.02 = 2%)
        adv:         average daily volume in shares
        eta:         temporary impact coefficient (default 0.1)

    Impact = price × eta × sigma × √(quantity / adv)
    """

    sigma_daily: float = 0.02
    adv: float = 1_000_000.0
    eta: float = 0.1

    def compute(
        self, price: float, quantity: float, side: OrderSide, bar: BarData,
    ) -> float:
        participation = abs(quantity) / max(self.adv, 1.0)
        impact = price * self.eta * self.sigma_daily * math.sqrt(participation)
        return price + impact if side == OrderSide.BUY else price - impact
