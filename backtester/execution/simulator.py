"""Event-driven order execution simulator.

This module owns the bar loop and delegates portfolio state transitions to
``PortfolioLedger``. Strategy code is invoked through callbacks so the
execution package remains independent of ``Backtester`` and can later be
reused by paper/live adapters.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.execution.fill_engine import FillEngine
from backtester.execution.order_types import (
    BarData,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from backtester.portfolio.accounting.ledger import LedgerResult, PortfolioLedger
from backtester.risk.pre_trade import PreTradeRisk

BarCallback = Callable[
    [object, Mapping[str, BarData], dict[str, float], float, dict[str, float | None]],
    None,
]
FillCallback = Callable[[Fill, float], None]


@dataclass(frozen=True)
class BatchSubmissionResult:
    """Audit result for one deterministic pre-trade evaluation batch."""

    submitted_order_ids: tuple[str, ...]
    rejected_order_ids: tuple[str, ...]


class ExecutionSimulator:
    """Run the order lifecycle against aligned OHLCV frames."""

    def __init__(
        self,
        *,
        fill_engine: FillEngine,
        ledger: PortfolioLedger,
        pre_trade_risk: PreTradeRisk | None = None,
    ) -> None:
        self.fill_engine = fill_engine
        self.ledger = ledger
        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()
        self._current_prices: dict[str, object] = {}
        self._current_nav = float(ledger.initial_cash)
        self._external_pre_submit_check = None
        self._external_pre_fill_check = None
        self._risk_prices: dict[str, object] = {}

    def run(
        self,
        *,
        close: pd.DataFrame,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        volume: pd.DataFrame,
        on_bar: BarCallback,
        on_fill: FillCallback | None = None,
    ) -> LedgerResult:
        """Execute pending orders, publish context, then accept next-bar orders.

        The phase order is intentionally stable:

        ``build bars -> process pending fills -> mark NAV -> strategy callback
        -> record state``.
        """
        self.fill_engine.reset()
        self.ledger.reset()
        instruments = list(close.columns)
        all_dates = close.index
        self.ledger.prepare_history(all_dates)

        # Align once, then use positional ndarray access inside the hot loop.
        # Bulk ``.loc`` preserves the previous fail-closed behavior for
        # missing OHLC labels without paying pandas scalar-indexing overhead
        # for every bar and instrument.
        close_values = close.loc[all_dates, instruments].to_numpy(copy=False)
        open_values = open_.loc[all_dates, instruments].to_numpy(copy=False)
        high_values = high.loc[all_dates, instruments].to_numpy(copy=False)
        low_values = low.loc[all_dates, instruments].to_numpy(copy=False)
        volume_values = np.full((len(all_dates), len(instruments)), np.nan, dtype=float)
        volume_columns = set(volume.columns)
        available_volume = [
            (column, instrument)
            for column, instrument in enumerate(instruments)
            if instrument in volume_columns
        ]
        if available_volume:
            selected = [instrument for _column, instrument in available_volume]
            selected_values = volume.loc[all_dates, selected].to_numpy(copy=False)
            for selected_column, (output_column, _instrument) in enumerate(available_volume):
                volume_values[:, output_column] = selected_values[:, selected_column]
        previous_check = getattr(self.fill_engine, "pre_submit_check", None)
        previous_price_validator = getattr(self.fill_engine, "price_validator", None)
        previous_fill_check = getattr(self.fill_engine, "pre_fill_check", None)
        self._external_pre_submit_check = previous_check
        self._external_pre_fill_check = previous_fill_check
        self.fill_engine.pre_submit_check = self._pre_submit_rejection_reason
        self.fill_engine.price_validator = lambda instrument, value: self.ledger.is_valid_price(
            value, instrument
        )
        self.fill_engine.pre_fill_check = self._pre_fill_rejection_reason

        try:
            for row, date in enumerate(all_dates):
                bars: dict[str, BarData] = {}
                self._current_prices = {}
                for column, instrument in enumerate(instruments):
                    close_value = close_values[row, column]
                    self.ledger.observe_mark(instrument, close_value)
                    self._current_prices[instrument] = close_value
                    bars[instrument] = BarData(
                        timestamp=date,
                        open=float(open_values[row, column]),
                        high=float(high_values[row, column]),
                        low=float(low_values[row, column]),
                        close=float(close_value),
                        volume=float(volume_values[row, column]),
                    )

                self._risk_prices = {
                    instrument: (
                        bars[instrument].open
                        if self.fill_engine.fill_at == "open"
                        else bars[instrument].close
                    )
                    for instrument in instruments
                }

                for instrument in instruments:
                    bar = bars[instrument]
                    for fill in self.fill_engine.process_bar(instrument, bar):
                        accounting = self.ledger.apply_fill(fill)
                        if on_fill is not None:
                            on_fill(fill, accounting.trade_notional)

                self._current_nav, position_values = self.ledger.mark_to_market(
                    self._current_prices
                )
                on_bar(
                    date,
                    dict(bars),
                    dict(self.ledger.positions),
                    float(self._current_nav),
                    dict(self.ledger.average_entry_price),
                )
                self.ledger.record(
                    date=date,
                    nav=self._current_nav,
                    position_values=position_values,
                    prices=self._current_prices,
                )
        finally:
            self.fill_engine.pre_submit_check = previous_check
            self.fill_engine.price_validator = previous_price_validator
            self.fill_engine.pre_fill_check = previous_fill_check
            self._current_prices = {}
            self._risk_prices = {}
            self._external_pre_submit_check = None
            self._external_pre_fill_check = None

        return self.ledger.result()

    def _pre_submit_rejection_reason(self, order: Order) -> str | None:
        if self._external_pre_submit_check is not None:
            reason = self._external_pre_submit_check(order)
            if reason:
                return str(reason)
        snapshot = self.ledger.snapshot(
            prices=self._current_prices,
            nav=self._current_nav,
        )
        decision = self.pre_trade_risk.evaluate(
            order,
            portfolio=snapshot,
            contract_spec_resolver=self.ledger.contract_spec,
            pending_orders=self.fill_engine.pending_orders,
        )
        return None if decision.approved else (decision.reason or "pre-trade rejected")

    def _pre_fill_rejection_reason(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
        commission: float,
    ) -> str | None:
        if self._external_pre_fill_check is not None:
            reason = self._external_pre_fill_check(order, fill_price, fill_quantity, commission)
            if reason:
                return str(reason)

        risk_prices = dict(self._risk_prices)
        risk_prices[order.instrument] = float(fill_price)
        risk_nav, _ = self.ledger.mark_to_market(risk_prices)
        risk_nav -= float(commission)
        snapshot = self.ledger.snapshot(prices=risk_prices, nav=risk_nav)
        candidate = Order(
            instrument=order.instrument,
            side=order.side,
            quantity=float(fill_quantity),
            order_type=OrderType.MARKET,
            created_at=order.created_at,
            tag=f"fill_check:{order.order_id}",
        )
        other_pending = [
            pending
            for pending in self.fill_engine.pending_orders
            if pending.order_id != order.order_id
        ]
        decision = self.pre_trade_risk.evaluate(
            candidate,
            portfolio=snapshot,
            contract_spec_resolver=self.ledger.contract_spec,
            pending_orders=other_pending,
        )
        if decision.approved:
            return None
        return "fill-time pre-trade rejected: " + (decision.reason or "portfolio limit")

    def submit_batch(self, orders: Sequence[Order]) -> BatchSubmissionResult:
        """Evaluate a deterministic trade list before adding any order."""
        candidates = []
        rejected_ids = []
        for order in orders:
            if self._external_pre_submit_check is not None:
                reason = self._external_pre_submit_check(order)
                if reason:
                    rejected_ids.append(self.fill_engine.reject(order, str(reason)))
                    continue
            candidates.append(order)

        snapshot = self.ledger.snapshot(
            prices=self._current_prices,
            nav=self._current_nav,
        )
        risk_result = self.pre_trade_risk.evaluate_batch(
            candidates,
            portfolio=snapshot,
            contract_spec_resolver=self.ledger.contract_spec,
            pending_orders=self.fill_engine.pending_orders,
            allow_cross_instrument_netting=False,
        )
        submitted_ids = []
        for order, decision in zip(candidates, risk_result.decisions, strict=True):
            if decision.approved:
                submitted_ids.append(self.fill_engine.submit(order, bypass_pre_submit=True))
            else:
                rejected_ids.append(
                    self.fill_engine.reject(order, decision.reason or "pre-trade rejected")
                )
        return BatchSubmissionResult(
            submitted_order_ids=tuple(submitted_ids),
            rejected_order_ids=tuple(rejected_ids),
        )


class TargetWeightOrderExecutor:
    """Convert stateful rebalance decisions into executable market orders."""

    REBALANCE_TAG_PREFIX = "rebalance:"

    def __init__(
        self,
        *,
        target_weights: pd.DataFrame,
        simulator: ExecutionSimulator,
        rebalance_engine: object,
    ) -> None:
        self.simulator = simulator
        self.fill_engine = simulator.fill_engine
        self.ledger = simulator.ledger
        self.rebalance_engine = rebalance_engine
        unknown = set(target_weights.columns) - set(self.ledger.instruments)
        if unknown:
            raise ValueError(
                "Target weights contain unknown instruments: "
                + ", ".join(sorted(map(str, unknown)))
            )
        aligned = target_weights.reindex(columns=self.ledger.instruments, fill_value=0.0).astype(
            float
        )
        if np.isinf(aligned.to_numpy()).any():
            raise ValueError("Target weights must not contain +/-inf")
        self.target_weights = aligned.fillna(0.0)
        self._date_positions = {date: i for i, date in enumerate(self.target_weights.index)}
        self._active_desired_weights: dict[str, float] | None = None
        self._active_reason = "policy"
        self._active_order_ids: set[str] = set()
        self._active_had_rejection = False

    def on_fill(self, fill: Fill, trade_notional: float) -> None:
        self.rebalance_engine.record_fill(
            timestamp=fill.timestamp,
            notional=trade_notional,
            nav=self.simulator._current_nav,
        )

    def on_bar(
        self,
        date: object,
        bars: Mapping[str, BarData],
        positions: Mapping[str, float],
        nav: float,
    ) -> BatchSubmissionResult:
        position = self._date_positions.get(date)
        if position is None:
            raise KeyError(f"No target-weight row for simulation date {date!r}")
        next_date = (
            self.target_weights.index[position + 1]
            if position + 1 < len(self.target_weights.index)
            else None
        )
        if not np.isfinite(float(nav)) or float(nav) <= 0.0:
            self._cancel_pending_rebalance_orders()
            self._clear_active_target()
            raise ValueError(
                f"target-weight execution requires positive finite NAV; got {nav!r} at {date!r}"
            )
        marks = {instrument: bars[instrument].close for instrument in self.ledger.instruments}
        realized_weights = self.ledger.exposure_weights(prices=marks, nav=nav)
        available = {
            instrument: self.ledger.is_valid_price(marks[instrument], instrument)
            for instrument in self.ledger.instruments
        }
        decision = self.rebalance_engine.evaluate(
            bar_index=position,
            decision_time=date,
            execution_time=next_date,
            target_weights=self.target_weights.loc[date].to_dict(),
            realized_weights=realized_weights,
            positions=positions,
            nav=nav,
            available=available,
        )
        if next_date is None:
            self._cancel_pending_rebalance_orders()
            self._clear_active_target()
            return BatchSubmissionResult((), ())

        if decision.should_rebalance:
            desired = {
                instrument: float(weight)
                for instrument, weight in decision.persistent_weights.items()
            }
            next_reason = decision.reason or "policy"
            risk_reasons = {"circuit_breaker", "post_cooldown"}
            same_intent = (
                self._active_desired_weights is not None
                and self._weights_equal(self._active_desired_weights, desired)
                and (
                    self._active_reason == next_reason
                    or (self._active_reason not in risk_reasons and next_reason not in risk_reasons)
                )
            )
            if same_intent and self._pending_rebalance_orders():
                return BatchSubmissionResult((), ())
            self._cancel_pending_rebalance_orders()
            self._active_desired_weights = desired
            self._active_reason = next_reason
            self._active_order_ids = set()
            self._active_had_rejection = False
        elif self._active_desired_weights is None:
            return BatchSubmissionResult((), ())
        elif self._pending_rebalance_orders():
            return BatchSubmissionResult((), ())
        elif self._active_intent_completed():
            self.rebalance_engine.record_target_reconciled(reason=self._active_reason)
            self._clear_active_target()
            return BatchSubmissionResult((), ())

        orders = self._build_orders(
            date=date,
            reason=self._active_reason,
            desired_weights=self._active_desired_weights,
            positions=positions,
            marks=marks,
            nav=nav,
        )
        if not orders:
            if self._target_is_reconciled(
                desired_weights=self._active_desired_weights,
                realized_weights=realized_weights,
                positions=positions,
                marks=marks,
                nav=nav,
            ):
                self.rebalance_engine.record_target_reconciled(reason=self._active_reason)
                self._clear_active_target()
            return BatchSubmissionResult((), ())
        result = self.simulator.submit_batch(orders)
        self._active_order_ids.update(result.submitted_order_ids)
        self._active_had_rejection = self._active_had_rejection or bool(result.rejected_order_ids)
        self.rebalance_engine.record_submission(
            timestamp=date,
            submitted=len(result.submitted_order_ids),
            rejected=len(result.rejected_order_ids),
            reason=self._active_reason,
        )
        return result

    @staticmethod
    def _weights_equal(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
        keys = set(left) | set(right)
        return all(
            abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) <= 1e-12 for key in keys
        )

    def _active_intent_completed(self) -> bool:
        if self._active_had_rejection or not self._active_order_ids:
            return False
        statuses = {order.order_id: order.status for order in self.fill_engine.order_history}
        return all(
            statuses.get(order_id) == OrderStatus.FILLED for order_id in self._active_order_ids
        )

    def _clear_active_target(self) -> None:
        self._active_desired_weights = None
        self._active_reason = "policy"
        self._active_order_ids = set()
        self._active_had_rejection = False

    def _target_is_reconciled(
        self,
        *,
        desired_weights: Mapping[str, float],
        realized_weights: Mapping[str, float],
        positions: Mapping[str, float],
        marks: Mapping[str, float],
        nav: float,
    ) -> bool:
        min_trade_weight = self._min_trade_weight(self._active_reason)
        for instrument in self.ledger.instruments:
            price = float(marks[instrument])
            desired_weight = float(desired_weights.get(instrument, 0.0))
            if not self.ledger.is_valid_price(price, instrument):
                if abs(desired_weight - float(realized_weights.get(instrument, 0.0))) > max(
                    min_trade_weight, 1e-12
                ):
                    return False
                continue
            spec = self.ledger.contract_spec(instrument)
            unit_notional = float(spec.notional(1.0, price))
            if not np.isfinite(unit_notional) or unit_notional <= 0.0:
                return False
            target_quantity = spec.round_quantity(float(nav) * desired_weight / unit_notional)
            current_quantity = float(positions.get(instrument, 0.0))
            delta = abs(float(target_quantity) - current_quantity)
            if delta <= 1e-12:
                continue
            if delta * unit_notional / abs(float(nav)) >= min_trade_weight:
                return False
        return True

    def _pending_rebalance_orders(self) -> list[Order]:
        return [
            order
            for order in self.fill_engine.pending_orders
            if order.is_active and order.tag.startswith(self.REBALANCE_TAG_PREFIX)
        ]

    def _cancel_pending_rebalance_orders(self) -> None:
        for order in list(self.fill_engine.pending_orders):
            if order.tag.startswith(self.REBALANCE_TAG_PREFIX):
                self.fill_engine.cancel(order.order_id)

    def _min_trade_weight(self, reason: str) -> float:
        if reason in {"circuit_breaker", "post_cooldown"}:
            return 0.0
        return max(
            float(getattr(self.rebalance_engine.policy, "min_trade_size", 0.0)),
            0.0,
        )

    def _build_orders(
        self,
        *,
        date: object,
        reason: str,
        desired_weights: Mapping[str, float],
        positions: Mapping[str, float],
        marks: Mapping[str, float],
        nav: float,
    ) -> list[Order]:
        reductions = []
        increases = []
        min_trade_weight = self._min_trade_weight(reason)
        for instrument in self.ledger.instruments:
            price = float(marks[instrument])
            if not self.ledger.is_valid_price(price, instrument):
                continue
            spec = self.ledger.contract_spec(instrument)
            unit_notional = float(spec.notional(1.0, price))
            if not np.isfinite(unit_notional) or unit_notional <= 0.0:
                continue
            target_value = float(nav) * float(desired_weights.get(instrument, 0.0))
            target_quantity = spec.round_quantity(target_value / unit_notional)
            current_quantity = float(positions.get(instrument, 0.0))
            if (
                current_quantity * target_quantity < 0.0
                and abs(current_quantity) > 1e-12
                and abs(target_quantity) > 1e-12
            ):
                close_order = self._order_from_delta(
                    instrument=instrument,
                    quantity_delta=-current_quantity,
                    date=date,
                    reason=reason,
                    phase="close",
                )
                open_order = self._order_from_delta(
                    instrument=instrument,
                    quantity_delta=target_quantity,
                    date=date,
                    reason=reason,
                    phase="open",
                )
                reductions.append(close_order)
                increases.append(open_order)
                continue

            delta = float(target_quantity - current_quantity)
            if abs(delta) <= 1e-12:
                continue
            delta_weight = abs(delta) * unit_notional / max(abs(float(nav)), 1e-12)
            if delta_weight + 1e-12 < min_trade_weight:
                continue
            order = self._order_from_delta(
                instrument=instrument,
                quantity_delta=delta,
                date=date,
                reason=reason,
                phase="rebalance",
            )
            if abs(target_quantity) < abs(current_quantity) - 1e-12:
                reductions.append(order)
            else:
                increases.append(order)
        return reductions + increases

    def _order_from_delta(
        self,
        *,
        instrument: str,
        quantity_delta: float,
        date: object,
        reason: str,
        phase: str,
    ) -> Order:
        return Order(
            instrument=instrument,
            side=OrderSide.BUY if quantity_delta > 0 else OrderSide.SELL,
            quantity=abs(float(quantity_delta)),
            order_type=OrderType.MARKET,
            created_at=date,
            tag=f"{self.REBALANCE_TAG_PREFIX}{reason}:{phase}",
        )


__all__ = [
    "BatchSubmissionResult",
    "ExecutionSimulator",
    "TargetWeightOrderExecutor",
]
