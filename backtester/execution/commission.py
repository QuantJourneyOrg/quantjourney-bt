"""
Commission Schemes — pluggable fee models.

All schemes implement the ``CommissionScheme`` protocol:

    def compute(self, price: float, quantity: float, notional: float) -> float:
        '''Return total commission for this fill.'''

Available schemes:
    ZeroCommission     — no fees
    PerShareCommission — fixed cost per share (e.g. $0.005/share, IB-style)
    FixedBpsCommission — percentage of notional (e.g. 1 bp)
    TieredCommission   — volume-based tiered pricing

CommissionConfig is a Pydantic model for clean YAML/JSON/env configuration.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field


# ── Protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class CommissionScheme(Protocol):
    """Protocol for commission schemes."""

    def compute(self, price: float, quantity: float, notional: float) -> float:
        """Return total commission cost for a single fill."""
        ...


# ── Implementations ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ZeroCommission:
    """No commission."""

    def compute(self, price: float, quantity: float, notional: float) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class PerShareCommission:
    """
    Fixed cost per share, with optional minimum and maximum.

    Example (Interactive Brokers US equities):
        PerShareCommission(cost_per_share=0.005, min_per_order=1.0, max_pct=0.005)
    """

    cost_per_share: float = 0.005
    min_per_order: float = 1.0
    max_pct: float = 0.005  # max as fraction of notional (0.5%)

    def compute(self, price: float, quantity: float, notional: float) -> float:
        raw = abs(quantity) * self.cost_per_share
        capped = min(raw, abs(notional) * self.max_pct) if self.max_pct else raw
        return max(capped, self.min_per_order)


@dataclass(frozen=True, slots=True)
class FixedBpsCommission:
    """
    Commission as fixed basis points of notional value.

    Example (1 bp = 0.01%):
        FixedBpsCommission(bps=1.0)
    """

    bps: float = 1.0
    min_per_order: float = 0.0

    def compute(self, price: float, quantity: float, notional: float) -> float:
        raw = abs(notional) * self.bps / 10_000
        return max(raw, self.min_per_order)


@dataclass(frozen=True, slots=True)
class TieredCommission:
    """
    Volume-based tiered commission (e.g. exchange fee schedules).

    Tiers are defined as (volume_threshold, cost_per_share) pairs,
    sorted ascending by threshold. Quantity above the last tier uses
    the last tier's rate.

    Example (US equities tiered):
        TieredCommission(tiers=[
            (300,       0.0035),   # first 300 shares @ $0.0035
            (3_000,     0.0020),   # next 2700 shares @ $0.0020
            (20_000,    0.0015),   # next 17000 @ $0.0015
            (float('inf'), 0.0010),
        ])
    """

    tiers: List[Tuple[float, float]]
    min_per_order: float = 0.35

    def compute(self, price: float, quantity: float, notional: float) -> float:
        qty_remaining = abs(quantity)
        total = 0.0
        prev_threshold = 0.0

        for threshold, rate in sorted(self.tiers, key=lambda t: t[0]):
            tier_qty = min(qty_remaining, threshold - prev_threshold)
            if tier_qty <= 0:
                break
            total += tier_qty * rate
            qty_remaining -= tier_qty
            prev_threshold = threshold

        return max(total, self.min_per_order)


# ── Pydantic Config ───────────────────────────────────────────────────

class CommissionConfig(BaseModel):
    """
    Beautiful config for commission schemes — supports YAML/JSON/env.

    Examples:
        # Zero commission
        CommissionConfig(scheme="zero")

        # IB-style per-share
        CommissionConfig(
            scheme="per_share",
            cost_per_share=0.005,
            min_per_order=1.0,
            max_pct=0.005,
        )

        # Fixed bps
        CommissionConfig(scheme="fixed_bps", bps=1.0)

        # Tiered
        CommissionConfig(
            scheme="tiered",
            tiers=[(300, 0.0035), (3000, 0.002), (20000, 0.0015)],
        )
    """

    scheme: str = Field("fixed_bps", description="One of: zero, per_share, fixed_bps, tiered")
    bps: float = Field(1.0, description="Basis points for fixed_bps scheme")
    cost_per_share: float = Field(0.005, description="Cost per share for per_share scheme")
    min_per_order: float = Field(1.0, description="Minimum commission per order")
    max_pct: float = Field(0.005, description="Max commission as fraction of notional")
    tiers: Optional[List[Tuple[float, float]]] = Field(
        None, description="Volume tiers: [(threshold, rate), ...]"
    )

    def build(self) -> CommissionScheme:
        """Construct the CommissionScheme from config."""
        if self.scheme == "zero":
            return ZeroCommission()
        elif self.scheme == "per_share":
            return PerShareCommission(
                cost_per_share=self.cost_per_share,
                min_per_order=self.min_per_order,
                max_pct=self.max_pct,
            )
        elif self.scheme == "fixed_bps":
            return FixedBpsCommission(
                bps=self.bps,
                min_per_order=self.min_per_order,
            )
        elif self.scheme == "tiered":
            return TieredCommission(
                tiers=self.tiers or [(float("inf"), 0.001)],
                min_per_order=self.min_per_order,
            )
        else:
            raise ValueError(
                f"Unknown commission scheme: {self.scheme}. "
                f"Options: zero, per_share, fixed_bps, tiered"
            )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"scheme": "zero"},
                {"scheme": "per_share", "cost_per_share": 0.005, "min_per_order": 1.0},
                {"scheme": "fixed_bps", "bps": 1.0},
            ]
        }
    }
