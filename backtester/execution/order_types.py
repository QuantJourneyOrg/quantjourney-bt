"""
Order types — Market, Limit, Stop, StopLimit, StopTrail, StopTrailLimit,
OCO, Bracket.

All types are pure dataclasses with no side-effects.
The FillEngine is responsible for matching orders against bars.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ──────────────────────────────────────────────────────────────

class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    STOP_TRAIL = "stop_trail"
    STOP_TRAIL_LIMIT = "stop_trail_limit"
    OCO = "oco"          # one-cancels-other (pair reference)
    BRACKET = "bracket"  # entry + TP + SL


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Order validity policy."""
    GTC = "GTC"  # good till cancelled
    DAY = "DAY"  # first eligible bar only in bar-based simulation
    GTD = "GTD"  # good till date/time


# ── Bar Data ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BarData:
    """Single OHLCV bar for fill simulation."""
    timestamp: object  # pd.Timestamp or datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# ── Fill ───────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Fill:
    """Result of an order execution."""
    order_id: str
    instrument: str
    side: OrderSide
    quantity: float
    fill_price: float          # price after slippage
    slippage: float = 0.0      # slippage cost (absolute per share)
    commission: float = 0.0    # commission cost (total)
    timestamp: object = None
    theoretical_price: Optional[float] = None
    remaining_qty: float = 0.0
    order_status: OrderStatus = OrderStatus.FILLED


# ── Bracket Spec ───────────────────────────────────────────────────────

@dataclass(slots=True)
class BracketSpec:
    """Bracket order specification: entry + take-profit + stop-loss."""
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_type: OrderType = OrderType.LIMIT
    stop_loss_type: OrderType = OrderType.STOP
    stop_limit_price: Optional[float] = None
    stop_limit_offset: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None


# ── Order ──────────────────────────────────────────────────────────────

@dataclass
class Order:
    """
    Unified order object supporting all order types.

    Usage:
        # Market order
        Order(instrument="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        # Limit order
        Order(instrument="AAPL", side=OrderSide.BUY, quantity=100,
              order_type=OrderType.LIMIT, limit_price=150.0)

        # Stop order
        Order(instrument="AAPL", side=OrderSide.SELL, quantity=100,
              order_type=OrderType.STOP, stop_price=140.0)

        # Stop-limit order
        Order(instrument="AAPL", side=OrderSide.SELL, quantity=100,
              order_type=OrderType.STOP_LIMIT, stop_price=140.0, limit_price=139.5)

        # Stop-trail (trailing stop, $5 distance)
        Order(instrument="AAPL", side=OrderSide.SELL, quantity=100,
              order_type=OrderType.STOP_TRAIL, trail_amount=5.0)

        # Stop-trail-limit with $5 trailing stop and $0.25 limit offset
        Order(instrument="AAPL", side=OrderSide.SELL, quantity=100,
              order_type=OrderType.STOP_TRAIL_LIMIT, trail_amount=5.0,
              limit_offset=0.25)

        # Bracket (entry + TP + SL)
        Order(instrument="AAPL", side=OrderSide.BUY, quantity=100,
              order_type=OrderType.BRACKET, limit_price=150.0,
              bracket=BracketSpec(take_profit_price=170.0, stop_loss_price=140.0))
    """

    # ── Required ──
    instrument: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET

    # ── Price parameters ──
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_offset: Optional[float] = None  # used by STOP_TRAIL_LIMIT

    # ── Trailing stop ──
    trail_amount: Optional[float] = None    # absolute $ distance
    trail_percent: Optional[float] = None   # percentage distance (0.02 = 2%)
    _trail_anchor: Optional[float] = None   # internal: current anchor price
    _limit_activated: bool = False          # internal: stop-limit was triggered
    _activated_limit_price: Optional[float] = None

    # ── Bracket ──
    bracket: Optional[BracketSpec] = None

    # ── OCO ──
    oco_pair_id: Optional[str] = None  # links two orders as OCO pair

    # ── Metadata ──
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: object = None   # timestamp
    expire_at: object = None    # GTD timestamp/date
    time_in_force: TimeInForce | str = TimeInForce.GTC
    expires_after_bars: Optional[int] = None
    _bars_live: int = 0          # internal: bars processed since submission
    tag: str = ""               # user-defined label

    @property
    def remaining_qty(self) -> float:
        return self.quantity - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    def cancel(self) -> None:
        if self.is_active:
            self.status = OrderStatus.CANCELLED

    def __repr__(self) -> str:
        return (
            f"Order({self.order_type.value} {self.side.value} "
            f"{self.remaining_qty:.1f} {self.instrument} "
            f"[{self.status.value}] id={self.order_id[:8]})"
        )
