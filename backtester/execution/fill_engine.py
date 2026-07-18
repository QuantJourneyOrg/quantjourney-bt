"""
Fill Engine — processes pending orders against OHLCV bars.

Supports: Market, Limit, Stop, StopLimit, StopTrail, StopTrailLimit, OCO,
Bracket orders.
Applies slippage and commission per fill.

Usage:
    from backtester.execution import (
        FillEngine, Order, OrderType, OrderSide, BarData,
        FixedBpsSlippage, PerShareCommission,
    )

    engine = FillEngine(
        slippage=FixedBpsSlippage(bps=5.0),
        commission=PerShareCommission(),
    )
    engine.submit(Order(
        instrument="AAPL", side=OrderSide.BUY, quantity=100,
        order_type=OrderType.LIMIT, limit_price=150.0,
    ))

    bar = BarData(timestamp=..., open=149.5, high=152.0, low=148.0, close=151.0, volume=1e6)
    fills = engine.process_bar("AAPL", bar)

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict, deque
from collections.abc import Callable

from backtester.execution.commission import CommissionScheme, ZeroCommission
from backtester.execution.order_types import (
    BarData,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from backtester.execution.slippage import NoSlippage, SlippageModel

logger = logging.getLogger(__name__)


class FillEngine:
    """
    Stateful order book that processes pending orders against OHLCV bars.

    Processing priority per bar:
        1. Stop / StopLimit / StopTrail / StopTrailLimit orders
        2. Limit orders
        3. Market orders

    Same-bar convention:
        Daily OHLC bars do not reveal the intrabar path. If a protective stop
        and take-profit are both reachable inside the same bar, this engine
        applies the priority above. For separate exit orders this is deliberately
        conservative: stops are evaluated before limits. Use intraday data for
        strategies whose edge depends on exact same-bar path ordering.

    Opening-fill observability:
        A market fill at the bar open cannot observe that bar's later high,
        low, close or volume. Slippage therefore receives the previous
        completed bar's HLCV, while volume participation uses the mean of up
        to ``volume_lookback`` prior volumes, optionally scaled by
        ``expected_open_volume_fraction``. Closing fills may use the completed
        current bar. With no lagged volume, an opening participation cap fails
        closed at zero capacity.

    Bracket orders are decomposed into an entry order + OCO pair (TP + SL).
    OCO logic: when one order in a pair fills, the other is cancelled.

    Per-run state and ``reset()``:
        A FillEngine instance accumulates per-run state — pending orders,
        order/fill histories, OCO registries, and per-order commission
        state. Re-running a backtest on the SAME engine instance without
        clearing this state changes results (e.g. an order left pending at
        the end of run 1 fills at the first bar of run 2, and audit
        histories accumulate across runs). Runners that reuse a Backtester
        / FillEngine instance MUST call ``reset()`` at the start of each
        run, before any orders are submitted.
    """

    def __init__(
        self,
        slippage: SlippageModel | None = None,
        commission: CommissionScheme | None = None,
        fill_at: str = "open",  # "open" or "close"
        max_volume_participation: float | None = None,
        volume_lookback: int = 20,
        expected_open_volume_fraction: float = 1.0,
        notional_fn: Callable[[str, float, float], float] | None = None,
        pre_submit_check: Callable[[Order], str | None] | None = None,
        price_validator: Callable[[str, float], bool] | None = None,
        pre_fill_check: Callable[[Order, float, float, float], str | None] | None = None,
    ):
        self.slippage = slippage or NoSlippage()
        self.commission = commission or ZeroCommission()
        self.fill_at = fill_at
        self.max_volume_participation = max_volume_participation
        if isinstance(volume_lookback, bool) or not isinstance(volume_lookback, int):
            raise TypeError("volume_lookback must be a positive integer")
        if volume_lookback <= 0:
            raise ValueError("volume_lookback must be a positive integer")
        open_fraction = float(expected_open_volume_fraction)
        if not math.isfinite(open_fraction) or not 0.0 <= open_fraction <= 1.0:
            raise ValueError("expected_open_volume_fraction must be between 0 and 1")
        self.volume_lookback = volume_lookback
        self.expected_open_volume_fraction = open_fraction
        self.notional_fn = notional_fn
        self.pre_submit_check = pre_submit_check
        self.price_validator = price_validator
        self.pre_fill_check = pre_fill_check

        # Active orders keyed by instrument
        self._orders: dict[str, list[Order]] = defaultdict(list)
        # OCO pair tracking: oco_pair_id → [order_id, order_id]
        self._oco_pairs: dict[str, list[str]] = {}
        # All orders ever submitted (for audit trail)
        self._order_history: list[Order] = []
        # All fills emitted by this engine (for strategy-side metadata)
        self._fill_history: list[Fill] = []
        self._last_fill_by_instrument: dict[str, Fill] = {}
        self._last_fill_by_order: dict[str, Fill] = {}
        self._commission_state_by_order: dict[str, dict[str, float]] = {}
        self._last_bar_by_instrument: dict[str, BarData] = {}
        self._volume_history: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.volume_lookback)
        )

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear all per-run state so the engine can start a fresh backtest run.

        Clears pending orders, OCO pair registries, order/fill histories,
        last-fill caches, and per-order commission state. Configuration
        (slippage, commission, fill_at, max_volume_participation,
        volume_lookback, expected_open_volume_fraction, notional_fn) is
        preserved.

        Runners that reuse a Backtester / FillEngine instance across runs
        MUST call this at run start — otherwise orders left pending by the
        previous run fill on the new run's first bar and audit histories
        accumulate across runs.
        """
        self._orders = defaultdict(list)
        self._oco_pairs = {}
        self._order_history = []
        self._fill_history = []
        self._last_fill_by_instrument = {}
        self._last_fill_by_order = {}
        self._commission_state_by_order = {}
        self._last_bar_by_instrument = {}
        self._volume_history = defaultdict(lambda: deque(maxlen=self.volume_lookback))

    # ── Submit ─────────────────────────────────────────────────────────

    def submit(self, order: Order, *, bypass_pre_submit: bool = False) -> str:
        """Submit an order. Returns order_id."""
        self._validate_order(order)
        if not bypass_pre_submit and self.pre_submit_check is not None:
            rejection_reason = self.pre_submit_check(order)
            if rejection_reason:
                return self.reject(order, str(rejection_reason))
        if order.order_type == OrderType.BRACKET:
            return self._submit_bracket(order)

        self._orders[order.instrument].append(order)
        self._order_history.append(order)

        # Register OCO pair tracking for user-submitted OCO orders
        if order.oco_pair_id:
            pair = self._oco_pairs.setdefault(order.oco_pair_id, [])
            pair.append(order.order_id)

        return order.order_id

    def reject(self, order: Order, reason: str) -> str:
        """Record a pre-trade rejection without adding a pending order."""
        order.status = OrderStatus.REJECTED
        order.rejection_reason = str(reason)
        self._order_history.append(order)
        return order.order_id

    def _submit_bracket(self, order: Order) -> str:
        """Decompose bracket into entry + OCO(TP, SL)."""
        bracket = order.bracket
        if bracket is None:
            raise ValueError("Bracket order requires a BracketSpec")
        self._validate_bracket(bracket)

        # Entry order (market or limit)
        entry_type = OrderType.LIMIT if order.limit_price else OrderType.MARKET
        entry = Order(
            instrument=order.instrument,
            side=order.side,
            quantity=order.quantity,
            order_type=entry_type,
            limit_price=order.limit_price,
            time_in_force=order.time_in_force,
            expire_at=order.expire_at,
            expires_after_bars=order.expires_after_bars,
            order_id=order.order_id,
            created_at=order.created_at,
            tag=f"bracket_entry:{order.order_id[:8]}",
        )

        # Take-profit and stop-loss as OCO pair
        exit_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        oco_id = str(uuid.uuid4())

        tp = Order(
            instrument=order.instrument,
            side=exit_side,
            quantity=order.quantity,
            order_type=bracket.take_profit_type,
            limit_price=bracket.take_profit_price,
            oco_pair_id=oco_id,
            time_in_force=order.time_in_force,
            expire_at=order.expire_at,
            expires_after_bars=order.expires_after_bars,
            tag=f"bracket_tp:{order.order_id[:8]}",
        )
        sl = Order(
            instrument=order.instrument,
            side=exit_side,
            quantity=order.quantity,
            order_type=bracket.stop_loss_type,
            stop_price=bracket.stop_loss_price,
            limit_price=bracket.stop_limit_price,
            limit_offset=bracket.stop_limit_offset,
            trail_amount=bracket.trail_amount,
            trail_percent=bracket.trail_percent,
            oco_pair_id=oco_id,
            time_in_force=order.time_in_force,
            expire_at=order.expire_at,
            expires_after_bars=order.expires_after_bars,
            tag=f"bracket_sl:{order.order_id[:8]}",
        )

        self._oco_pairs[oco_id] = [tp.order_id, sl.order_id]
        tp._bracket_parent_order_id = entry.order_id  # type: ignore[attr-defined]
        sl._bracket_parent_order_id = entry.order_id  # type: ignore[attr-defined]

        # Submit entry immediately; TP + SL are parked until entry fills
        self._orders[order.instrument].append(entry)
        self._order_history.extend([entry, tp, sl])

        # Store bracket children for post-entry activation
        entry._bracket_children = [tp, sl]  # type: ignore[attr-defined]
        entry._bracket_spec = bracket  # type: ignore[attr-defined]
        entry._bracket_parent_side = order.side  # type: ignore[attr-defined]
        entry._bracket_children_activated = False  # type: ignore[attr-defined]

        return entry.order_id

    # ── Cancel ─────────────────────────────────────────────────────────

    def cancel(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if found and cancelled."""
        for instrument_orders in self._orders.values():
            for order in instrument_orders:
                if order.order_id == order_id and order.is_active:
                    order.cancel()
                    return True
        return False

    def cancel_all(self, instrument: str | None = None) -> int:
        """Cancel all active orders, optionally filtered by instrument."""
        count = 0
        targets = [instrument] if instrument else list(self._orders.keys())
        for inst in targets:
            for order in self._orders.get(inst, []):
                if order.is_active:
                    order.cancel()
                    count += 1
        return count

    # ── Process Bar ────────────────────────────────────────────────────

    def process_bar(self, instrument: str, bar: BarData) -> list[Fill]:
        """
        Process all pending orders for an instrument against one bar.
        Returns list of fills generated.
        """
        fills: list[Fill] = []
        previous_bar = self._last_bar_by_instrument.get(instrument)
        orders = self._orders.get(instrument, [])
        if not orders:
            self._observe_bar(instrument, bar)
            return fills

        remaining_bar_capacity = self._bar_fill_capacity(instrument, bar)
        slippage_bar = self._slippage_reference_bar(bar, previous_bar)

        # Process in priority: stops → limits → markets
        for order in sorted(orders, key=lambda o: _ORDER_PRIORITY.get(o.order_type, 99)):
            if not order.is_active:
                continue

            if self._expire_before_bar(order, bar):
                continue

            fill = self._try_fill(
                order,
                bar,
                max_quantity=remaining_bar_capacity,
                slippage_bar=slippage_bar,
            )
            if fill:
                fills.append(fill)
                self._record_fill(fill)
                if remaining_bar_capacity is not None:
                    remaining_bar_capacity = max(0.0, remaining_bar_capacity - fill.quantity)

                # Activate bracket children on the FIRST entry fill (partial
                # or full) so the position is never left unprotected. Children
                # are sized to the actual filled quantity (the engine does not
                # clamp sells to held quantity) and resized on later partials.
                if hasattr(order, "_bracket_children"):
                    for child in order._bracket_children:  # type: ignore[attr-defined]
                        if child.is_active:
                            child.quantity = order.filled_qty
                    if not order._bracket_children_activated:  # type: ignore[attr-defined]
                        order._bracket_children_activated = True  # type: ignore[attr-defined]
                        self._prepare_bracket_children(order, fill)
                        for child in order._bracket_children:  # type: ignore[attr-defined]
                            self._orders[instrument].append(child)

                # OCO: any fill (partial or full) cancels the sibling leg.
                if order.oco_pair_id:
                    self._cancel_oco_pair(order.oco_pair_id, except_id=order.order_id)
                parent_id = getattr(order, "_bracket_parent_order_id", None)
                if parent_id is not None:
                    self.cancel(parent_id)

                if order.is_active:
                    self._expire_after_bar(order, bar, filled_this_bar=True)
            else:
                self._expire_after_bar(order, bar, filled_this_bar=False)

        # Clean up filled/cancelled orders
        self._orders[instrument] = [o for o in orders if o.is_active]
        self._observe_bar(instrument, bar)
        return fills

    # ── Fill Logic ─────────────────────────────────────────────────────

    def _try_fill(
        self,
        order: Order,
        bar: BarData,
        max_quantity: float | None = None,
        slippage_bar: BarData | None = None,
    ) -> Fill | None:
        """Attempt to fill a single order against a bar."""
        if not self._bar_supports_order(order, bar):
            return None
        # Ratchet the trailing anchor first so the stop is live from the
        # first bar and keeps trailing even when capacity blocks a fill.
        if order.order_type in (OrderType.STOP_TRAIL, OrderType.STOP_TRAIL_LIMIT):
            self._update_trail_anchor(order, bar)

        theoretical_price = self._get_theoretical_price(order, bar)
        if theoretical_price is None or not self._valid_price(order.instrument, theoretical_price):
            return None

        fill_qty = self._fill_quantity(order, max_quantity)
        if fill_qty <= 0:
            return None

        # Apply slippage. Limit-like orders are clamped back to price-or-better
        # after slippage so a simulated fill can never violate the limit.
        slipped_price = self.slippage.compute(
            price=theoretical_price,
            quantity=fill_qty,
            side=order.side,
            bar=bar if slippage_bar is None else slippage_bar,
        )
        if not self._valid_price(order.instrument, slipped_price):
            return None
        fill_price = self._apply_limit_price_constraint(order, slipped_price, theoretical_price)
        if not self._valid_price(order.instrument, fill_price):
            return None

        notional = self._trade_notional(order.instrument, fill_price, fill_qty)
        commission_cost = self._compute_incremental_commission(
            order, fill_price, fill_qty, notional, commit=False
        )
        if self.pre_fill_check is not None:
            rejection_reason = self.pre_fill_check(
                order,
                float(fill_price),
                float(fill_qty),
                float(commission_cost),
            )
            if rejection_reason:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = str(rejection_reason)
                return None
        commission_cost = self._compute_incremental_commission(
            order, fill_price, fill_qty, notional, commit=True
        )

        slippage_per_share = abs(fill_price - theoretical_price)

        previous_filled_qty = order.filled_qty
        order.filled_qty += fill_qty
        if order.avg_fill_price is None or previous_filled_qty <= 0:
            order.avg_fill_price = fill_price
        else:
            order.avg_fill_price = (
                (order.avg_fill_price * previous_filled_qty) + (fill_price * fill_qty)
            ) / order.filled_qty
        order.status = OrderStatus.FILLED if order.remaining_qty <= 1e-12 else OrderStatus.PARTIAL

        return Fill(
            order_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=fill_qty,
            fill_price=fill_price,
            slippage=slippage_per_share,
            commission=commission_cost,
            timestamp=bar.timestamp,
            theoretical_price=theoretical_price,
            remaining_qty=max(order.remaining_qty, 0.0),
            order_status=order.status,
        )

    def _valid_price(self, instrument: str, value: object) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(numeric):
            return False
        if self.price_validator is None:
            return numeric > 0.0
        return bool(self.price_validator(instrument, numeric))

    def _bar_supports_order(self, order: Order, bar: BarData) -> bool:
        """Validate only prices observable to this order's fill rule."""

        def valid(value: object) -> bool:
            return self._valid_price(order.instrument, value)

        if order.order_type == OrderType.MARKET:
            return valid(bar.open if self.fill_at == "open" else bar.close)
        if order.order_type == OrderType.LIMIT:
            trigger = bar.low if order.side == OrderSide.BUY else bar.high
            return valid(trigger) and valid(bar.open)
        if order.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            trigger = bar.high if order.side == OrderSide.BUY else bar.low
            return valid(trigger) and valid(bar.open)
        if order.order_type in {OrderType.STOP_TRAIL, OrderType.STOP_TRAIL_LIMIT}:
            return valid(bar.open) and valid(bar.high) and valid(bar.low)
        if order.order_type == OrderType.OCO:
            required = [bar.open]
            if order.stop_price is not None:
                required.append(bar.high if order.side == OrderSide.BUY else bar.low)
            if order.limit_price is not None:
                required.append(bar.low if order.side == OrderSide.BUY else bar.high)
            return all(valid(value) for value in required)
        return False

    def _get_theoretical_price(self, order: Order, bar: BarData) -> float | None:
        """Determine the theoretical (pre-slippage) fill price, or None if not triggered."""
        match order.order_type:
            case OrderType.MARKET:
                return bar.open if self.fill_at == "open" else bar.close

            case OrderType.LIMIT:
                if order.limit_price is None:
                    return None
                if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                    return min(order.limit_price, bar.open)
                if order.side == OrderSide.SELL and bar.high >= order.limit_price:
                    return max(order.limit_price, bar.open)
                return None

            case OrderType.STOP:
                if order.stop_price is None:
                    return None
                if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                    return max(order.stop_price, bar.open)
                if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                    return min(order.stop_price, bar.open)
                return None

            case OrderType.STOP_LIMIT:
                return self._get_stop_limit_price(order, bar)

            case OrderType.STOP_TRAIL:
                effective_stop = self._compute_trail_stop(order, bar)
                if effective_stop is None:
                    return None
                if order.side == OrderSide.SELL and bar.low <= effective_stop:
                    return min(effective_stop, bar.open)
                if order.side == OrderSide.BUY and bar.high >= effective_stop:
                    return max(effective_stop, bar.open)
                return None

            case OrderType.STOP_TRAIL_LIMIT:
                return self._get_trailing_stop_limit_price(order, bar)

            case OrderType.OCO:
                # OCO orders act as limit or stop depending on which price is set.
                # Stop is evaluated first per the same-bar convention above.
                if order.stop_price is not None:
                    if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                        return max(order.stop_price, bar.open)
                    if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                        return min(order.stop_price, bar.open)
                if order.limit_price is not None:
                    if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                        return min(order.limit_price, bar.open)
                    if order.side == OrderSide.SELL and bar.high >= order.limit_price:
                        return max(order.limit_price, bar.open)
                return None

            case _:
                return None

    def _active_limit_price(self, order: Order, theoretical_price: float) -> float | None:
        if order.order_type == OrderType.LIMIT:
            return order.limit_price
        if order.order_type in (OrderType.STOP_LIMIT, OrderType.STOP_TRAIL_LIMIT):
            return order._activated_limit_price or order.limit_price
        if order.order_type == OrderType.OCO and order.limit_price is not None:
            if order.side == OrderSide.BUY and theoretical_price <= order.limit_price:
                return order.limit_price
            if order.side == OrderSide.SELL and theoretical_price >= order.limit_price:
                return order.limit_price
        return None

    def _apply_limit_price_constraint(
        self,
        order: Order,
        fill_price: float,
        theoretical_price: float,
    ) -> float:
        limit_price = self._active_limit_price(order, theoretical_price)
        if limit_price is None:
            return fill_price
        if order.side == OrderSide.BUY:
            return min(fill_price, limit_price)
        return max(fill_price, limit_price)

    def _compute_incremental_commission(
        self,
        order: Order,
        fill_price: float,
        fill_qty: float,
        notional: float,
        *,
        commit: bool = True,
    ) -> float:
        state = self._commission_state_by_order.get(
            order.order_id,
            {"quantity": 0.0, "notional": 0.0, "charged": 0.0},
        )
        cumulative_qty = state["quantity"] + abs(float(fill_qty))
        cumulative_notional = state["notional"] + abs(float(notional))
        cumulative_due = self.commission.compute(
            price=fill_price,
            quantity=cumulative_qty,
            notional=cumulative_notional,
        )
        incremental = max(0.0, float(cumulative_due) - state["charged"])
        if commit:
            self._commission_state_by_order[order.order_id] = {
                "quantity": cumulative_qty,
                "notional": cumulative_notional,
                "charged": state["charged"] + incremental,
            }
        return incremental

    def _trade_notional(self, instrument: str, price: float, quantity: float) -> float:
        if self.notional_fn is not None:
            return float(self.notional_fn(instrument, price, quantity))
        return float(price) * float(quantity)

    def _compute_trail_stop(self, order: Order, bar: BarData) -> float | None:
        """Compute the effective stop price for a trailing stop."""
        anchor = order._trail_anchor
        if anchor is None:
            return None

        if order.trail_amount is not None:
            distance = order.trail_amount
        elif order.trail_percent is not None:
            distance = anchor * order.trail_percent
        else:
            return None

        if order.side == OrderSide.SELL:
            return anchor - distance
        else:
            return anchor + distance

    def _get_stop_limit_price(self, order: Order, bar: BarData) -> float | None:
        """Activate a stop-limit order, then fill only at-or-better than limit."""
        if order.stop_price is None or order.limit_price is None:
            return None

        reference_price = bar.open
        if not order._limit_activated:
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                order._limit_activated = True
                order._activated_limit_price = order.limit_price
                # Activation bar: the fill cannot reference prices from before
                # the stop triggered — floor the reference at the stop price.
                reference_price = max(order.stop_price, bar.open)
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                order._limit_activated = True
                order._activated_limit_price = order.limit_price
                reference_price = min(order.stop_price, bar.open)
            else:
                return None

        return self._get_limit_price(order, bar, order._activated_limit_price, reference_price)

    def _get_trailing_stop_limit_price(self, order: Order, bar: BarData) -> float | None:
        """Activate a trailing stop-limit, then fill only at-or-better than limit."""
        effective_stop = self._compute_trail_stop(order, bar)
        if effective_stop is None:
            return None

        reference_price = bar.open
        if not order._limit_activated:
            if order.side == OrderSide.SELL and bar.low <= effective_stop:
                order._limit_activated = True
                order._activated_limit_price = self._trail_limit_price(order, effective_stop)
                # Activation bar: bound the fill reference by the trigger price.
                reference_price = min(effective_stop, bar.open)
            elif order.side == OrderSide.BUY and bar.high >= effective_stop:
                order._limit_activated = True
                order._activated_limit_price = self._trail_limit_price(order, effective_stop)
                reference_price = max(effective_stop, bar.open)
            else:
                return None

        return self._get_limit_price(order, bar, order._activated_limit_price, reference_price)

    def _trail_limit_price(self, order: Order, effective_stop: float) -> float:
        """Resolve the limit price for a stop-trail-limit activation."""
        if order.limit_price is not None:
            return order.limit_price
        offset = float(order.limit_offset or 0.0)
        if order.side == OrderSide.SELL:
            return effective_stop - offset
        return effective_stop + offset

    def _get_limit_price(
        self,
        order: Order,
        bar: BarData,
        limit_price: float | None,
        reference_price: float | None = None,
    ) -> float | None:
        """Return fill price for a limit order if reachable.

        ``reference_price`` overrides ``bar.open`` as the price-improvement
        reference (used on stop-limit activation bars where the fill must be
        bounded by the trigger price rather than the pre-trigger open).
        """
        if limit_price is None:
            return None
        reference = bar.open if reference_price is None else reference_price
        if order.side == OrderSide.BUY and bar.low <= limit_price:
            return min(limit_price, reference)
        if order.side == OrderSide.SELL and bar.high >= limit_price:
            return max(limit_price, reference)
        return None

    def _update_trail_anchor(self, order: Order, bar: BarData) -> None:
        """Update the trailing stop anchor based on new bar data."""
        current = order._trail_anchor

        if order.side == OrderSide.SELL:
            # Selling: anchor ratchets UP with high
            new_anchor = bar.high
            if current is None or new_anchor > current:
                order._trail_anchor = new_anchor
        else:
            # Buying: anchor ratchets DOWN with low
            new_anchor = bar.low
            if current is None or new_anchor < current:
                order._trail_anchor = new_anchor

    def _cancel_oco_pair(self, oco_id: str, except_id: str) -> None:
        """Cancel the other order in an OCO pair."""
        pair_ids = self._oco_pairs.get(oco_id, [])
        for oid in pair_ids:
            if oid != except_id:
                self.cancel(oid)

    def _prepare_bracket_children(self, entry_order: Order, entry_fill: Fill) -> None:
        """Resolve bracket child prices that depend on the actual entry fill."""
        bracket = getattr(entry_order, "_bracket_spec", None)
        parent_side = getattr(entry_order, "_bracket_parent_side", entry_order.side)
        if bracket is None:
            return

        entry_price = float(entry_order.avg_fill_price or entry_fill.fill_price)
        is_long = parent_side == OrderSide.BUY
        for child in entry_order._bracket_children:  # type: ignore[attr-defined]
            if child.tag.startswith("bracket_tp") and bracket.take_profit_pct is not None:
                pct = float(bracket.take_profit_pct)
                child.limit_price = entry_price * (1.0 + pct if is_long else 1.0 - pct)

            if child.tag.startswith("bracket_sl"):
                if bracket.stop_loss_pct is not None and child.order_type in (
                    OrderType.STOP,
                    OrderType.STOP_LIMIT,
                ):
                    pct = float(bracket.stop_loss_pct)
                    child.stop_price = entry_price * (1.0 - pct if is_long else 1.0 + pct)

                if child.order_type == OrderType.STOP_LIMIT and child.limit_price is None:
                    stop = child.stop_price
                    if stop is not None and bracket.stop_limit_offset is not None:
                        offset = float(bracket.stop_limit_offset)
                        child.limit_price = (
                            stop - offset if child.side == OrderSide.SELL else stop + offset
                        )

    def _fill_quantity(self, order: Order, max_quantity: float | None) -> float:
        remaining = max(float(order.remaining_qty), 0.0)
        if max_quantity is None:
            return remaining
        return min(remaining, max(float(max_quantity), 0.0))

    def _bar_fill_capacity(self, instrument: str, bar: BarData) -> float | None:
        if self.max_volume_participation is None:
            return None
        if self.max_volume_participation <= 0:
            return 0.0
        if self.fill_at == "open":
            observations = self._volume_history.get(instrument)
            if not observations:
                logger.warning(
                    "No lagged volume for %s at %s with open participation cap; "
                    "treating capacity as 0 (no fills this bar)",
                    instrument,
                    bar.timestamp,
                )
                return 0.0
            volume = sum(observations) / len(observations)
            volume *= self.expected_open_volume_fraction
        else:
            volume = float(bar.volume)
        if not math.isfinite(volume):
            # Unknown liquidity: refuse to fill rather than fill uncapped
            # (NaN would otherwise bypass min() and poison later capacity).
            logger.warning(
                "Non-finite volume for %s at %s with max_volume_participation set; "
                "treating bar capacity as 0 (no fills this bar)",
                instrument,
                bar.timestamp,
            )
            return 0.0
        return max(volume, 0.0) * float(self.max_volume_participation)

    def _slippage_reference_bar(
        self,
        bar: BarData,
        previous_bar: BarData | None,
    ) -> BarData:
        """Expose only information available at the configured fill time."""
        if self.fill_at != "open":
            return bar
        if previous_bar is None:
            return BarData(
                timestamp=bar.timestamp,
                open=bar.open,
                high=float("nan"),
                low=float("nan"),
                close=float("nan"),
                volume=float("nan"),
            )
        return BarData(
            timestamp=bar.timestamp,
            open=bar.open,
            high=previous_bar.high,
            low=previous_bar.low,
            close=previous_bar.close,
            volume=previous_bar.volume,
        )

    def _observe_bar(self, instrument: str, bar: BarData) -> None:
        self._last_bar_by_instrument[instrument] = bar
        volume = float(bar.volume)
        if math.isfinite(volume) and volume >= 0.0:
            self._volume_history[instrument].append(volume)

    def _normalize_tif(self, order: Order) -> TimeInForce:
        tif = order.time_in_force
        if isinstance(tif, TimeInForce):
            return tif
        return TimeInForce[str(tif).upper()]

    def _expire_before_bar(self, order: Order, bar: BarData) -> bool:
        if order.created_at is None:
            order.created_at = bar.timestamp
        if order.expire_at is not None and self._timestamp_gt(bar.timestamp, order.expire_at):
            order.status = OrderStatus.EXPIRED
            return True
        return False

    def _expire_after_bar(self, order: Order, bar: BarData, filled_this_bar: bool = False) -> bool:
        if not order.is_active:
            return False

        # Bars with a (partial) fill do not count against expires_after_bars —
        # an actively-filling order must not be silently expired.
        if not filled_this_bar:
            order._bars_live += 1
        tif = self._normalize_tif(order)
        should_expire = False
        if tif == TimeInForce.DAY:
            should_expire = True
        if order.expires_after_bars is not None and order._bars_live >= order.expires_after_bars:
            should_expire = True
        if order.expire_at is not None and self._timestamp_gte(bar.timestamp, order.expire_at):
            should_expire = True

        if should_expire:
            order.status = OrderStatus.EXPIRED
            return True
        return False

    @staticmethod
    def _timestamp_gt(left, right) -> bool:
        try:
            return left > right
        except TypeError:
            import pandas as pd

            return pd.Timestamp(left) > pd.Timestamp(right)

    @staticmethod
    def _timestamp_gte(left, right) -> bool:
        try:
            return left >= right
        except TypeError:
            import pandas as pd

            return pd.Timestamp(left) >= pd.Timestamp(right)

    def _record_fill(self, fill: Fill) -> None:
        self._fill_history.append(fill)
        self._last_fill_by_instrument[fill.instrument] = fill
        self._last_fill_by_order[fill.order_id] = fill

    def _validate_order(self, order: Order) -> None:
        if not math.isfinite(float(order.quantity)) or order.quantity <= 0:
            raise ValueError("Order quantity must be finite and positive")
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if order.order_type == OrderType.STOP and order.stop_price is None:
            raise ValueError("STOP order requires stop_price")
        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                raise ValueError("STOP_LIMIT order requires stop_price and limit_price")
        if order.order_type == OrderType.STOP_TRAIL:
            if order.trail_amount is None and order.trail_percent is None:
                raise ValueError("STOP_TRAIL order requires trail_amount or trail_percent")
        if order.order_type == OrderType.STOP_TRAIL_LIMIT:
            if order.trail_amount is None and order.trail_percent is None:
                raise ValueError("STOP_TRAIL_LIMIT order requires trail_amount or trail_percent")
        self._normalize_tif(order)

    def _validate_bracket(self, bracket) -> None:
        if (
            bracket.take_profit_type == OrderType.LIMIT
            and bracket.take_profit_price is None
            and bracket.take_profit_pct is None
        ):
            raise ValueError(
                "Bracket LIMIT take-profit requires take_profit_price or take_profit_pct"
            )
        if (
            bracket.stop_loss_type == OrderType.STOP
            and bracket.stop_loss_price is None
            and bracket.stop_loss_pct is None
        ):
            raise ValueError("Bracket STOP stop-loss requires stop_loss_price or stop_loss_pct")
        if bracket.stop_loss_type == OrderType.STOP_LIMIT:
            if bracket.stop_loss_price is None and bracket.stop_loss_pct is None:
                raise ValueError(
                    "Bracket STOP_LIMIT stop-loss requires stop_loss_price or stop_loss_pct"
                )
            if bracket.stop_limit_price is None and bracket.stop_limit_offset is None:
                raise ValueError(
                    "Bracket STOP_LIMIT stop-loss requires stop_limit_price or stop_limit_offset"
                )
        if bracket.stop_loss_type in (OrderType.STOP_TRAIL, OrderType.STOP_TRAIL_LIMIT):
            if bracket.trail_amount is None and bracket.trail_percent is None:
                raise ValueError(
                    "Bracket trailing stop-loss requires trail_amount or trail_percent"
                )

    # ── Queries ────────────────────────────────────────────────────────

    @property
    def pending_orders(self) -> list[Order]:
        """All active pending orders across all instruments."""
        return [o for orders in self._orders.values() for o in orders if o.is_active]

    @property
    def order_history(self) -> list[Order]:
        """All orders ever submitted."""
        return list(self._order_history)

    @property
    def fill_history(self) -> list[Fill]:
        """All fills emitted by this engine."""
        return list(self._fill_history)

    def last_fill(self, instrument: str | None = None) -> Fill | None:
        """Return the last fill globally or for one instrument."""
        if instrument is not None:
            return self._last_fill_by_instrument.get(instrument)
        return self._fill_history[-1] if self._fill_history else None

    def last_fill_price(self, instrument: str | None = None) -> float | None:
        """Return the last actual fill price globally or for one instrument."""
        fill = self.last_fill(instrument)
        return None if fill is None else float(fill.fill_price)


# ── Priority map ──────────────────────────────────────────────────────
_ORDER_PRIORITY = {
    OrderType.STOP: 0,
    OrderType.STOP_LIMIT: 1,
    OrderType.STOP_TRAIL: 2,
    OrderType.STOP_TRAIL_LIMIT: 3,
    # OCO may carry a protective stop leg, so it shares the conservative
    # stop-family priority instead of trailing MARKET orders.
    OrderType.OCO: 4,
    OrderType.LIMIT: 5,
    OrderType.MARKET: 6,
    OrderType.BRACKET: 7,
}
