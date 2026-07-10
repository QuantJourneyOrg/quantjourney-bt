"""Pre-trade order validation and portfolio-limit checks.

The default configuration is deliberately pass-through for backward
compatibility. Institutional limits become active only when explicitly
configured because leverage and margin rules depend on account and venue.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from backtester.execution.contract_spec import AssetClass, ContractSpec
from backtester.execution.order_types import Order, OrderSide
from backtester.portfolio.accounting.ledger import PortfolioSnapshot

ContractSpecResolver = Callable[[str], ContractSpec]


@dataclass(frozen=True)
class PreTradeDecision:
    """Decision and projected portfolio state for one proposed order."""

    approved: bool
    reason: str | None
    projected_margin: float
    projected_buying_power: float
    projected_gross_leverage: float


@dataclass(frozen=True)
class PreTradeResult:
    """Batch pre-trade result preserving input order and audit reasons."""

    approved_orders: tuple[Order, ...]
    rejected_orders: tuple[Order, ...]
    decisions: tuple[PreTradeDecision, ...]


class PreTradeRisk:
    """Configurable checks applied immediately before order submission.

    Parameters are opt-in. With default values the class validates projections
    but does not impose a universal account leverage policy.
    """

    def __init__(
        self,
        *,
        max_margin_utilization: float | None = None,
        max_gross_leverage: float | None = None,
        allow_short: bool = True,
        reserve_pending_orders: bool = False,
        tolerance: float = 1e-9,
    ) -> None:
        if max_margin_utilization is not None and max_margin_utilization < 0:
            raise ValueError("max_margin_utilization must be non-negative")
        if max_gross_leverage is not None and max_gross_leverage < 0:
            raise ValueError("max_gross_leverage must be non-negative")
        self.max_margin_utilization = max_margin_utilization
        self.max_gross_leverage = max_gross_leverage
        self.allow_short = bool(allow_short)
        self.reserve_pending_orders = bool(reserve_pending_orders)
        self.tolerance = float(tolerance)

    @property
    def is_passthrough(self) -> bool:
        """Whether no portfolio limit can reject a finite order."""
        return (
            self.max_margin_utilization is None
            and self.max_gross_leverage is None
            and self.allow_short
        )

    def evaluate(
        self,
        order: Order,
        *,
        portfolio: PortfolioSnapshot,
        contract_spec_resolver: ContractSpecResolver,
        pending_orders: Sequence[Order] = (),
    ) -> PreTradeDecision:
        """Evaluate one order against projected positions and collateral."""
        quantity = float(order.quantity)
        if not np.isfinite(quantity) or quantity <= 0.0:
            return self._reject("order quantity must be finite and positive")

        positions = {
            str(instrument): float(value) for instrument, value in portfolio.positions.items()
        }
        if self.reserve_pending_orders:
            for pending in pending_orders:
                if pending.is_active:
                    self._apply_order_delta(positions, pending)

        current_positions = dict(positions)
        self._apply_order_delta(positions, order)

        projected_quantity = positions.get(order.instrument, 0.0)
        if not self.allow_short and projected_quantity < -self.tolerance:
            return self._decision(
                approved=False,
                reason=f"short positions are disabled for {order.instrument}",
                positions=positions,
                portfolio=portfolio,
                contract_spec_resolver=contract_spec_resolver,
            )

        current_margin, _ = self._projected_requirements(
            current_positions,
            portfolio.prices,
            portfolio.nav,
            contract_spec_resolver,
        )
        projected_margin, projected_gross = self._projected_requirements(
            positions,
            portfolio.prices,
            portfolio.nav,
            contract_spec_resolver,
        )
        projected_buying_power = float(portfolio.nav) - projected_margin

        # Risk-reducing orders remain executable even when the book is already
        # outside an opening limit. This is required for closes and deleveraging.
        risk_reducing = projected_margin <= current_margin + self.tolerance
        if not risk_reducing and self.max_margin_utilization is not None:
            allowed_margin = max(float(portfolio.nav), 0.0) * float(self.max_margin_utilization)
            if projected_margin > allowed_margin + self.tolerance:
                return PreTradeDecision(
                    approved=False,
                    reason=(
                        f"projected margin {projected_margin:.6g} exceeds "
                        f"limit {allowed_margin:.6g}"
                    ),
                    projected_margin=projected_margin,
                    projected_buying_power=projected_buying_power,
                    projected_gross_leverage=projected_gross,
                )

        if (
            not risk_reducing
            and self.max_gross_leverage is not None
            and projected_gross > float(self.max_gross_leverage) + self.tolerance
        ):
            return PreTradeDecision(
                approved=False,
                reason=(
                    f"projected gross leverage {projected_gross:.6g} exceeds "
                    f"limit {float(self.max_gross_leverage):.6g}"
                ),
                projected_margin=projected_margin,
                projected_buying_power=projected_buying_power,
                projected_gross_leverage=projected_gross,
            )

        return PreTradeDecision(
            approved=True,
            reason=None,
            projected_margin=projected_margin,
            projected_buying_power=projected_buying_power,
            projected_gross_leverage=projected_gross,
        )

    def evaluate_batch(
        self,
        orders: Iterable[Order],
        *,
        portfolio: PortfolioSnapshot,
        contract_spec_resolver: ContractSpecResolver,
        pending_orders: Sequence[Order] = (),
        allow_cross_instrument_netting: bool = True,
    ) -> PreTradeResult:
        """Evaluate a trade list using approved deltas.

        When ``allow_cross_instrument_netting`` is false, approved orders only
        affect later checks for the same instrument. A planned sale of one
        asset therefore cannot finance a purchase of another asset before the
        sale has actually filled. This is the conservative convention used by
        target-weight execution on bar data, where multi-leg atomicity cannot
        be guaranteed.
        """
        approved = []
        rejected = []
        decisions = []
        working_positions: dict[str, float] = {
            str(instrument): float(value) for instrument, value in portfolio.positions.items()
        }
        if self.reserve_pending_orders and allow_cross_instrument_netting:
            for pending in pending_orders:
                if pending.is_active:
                    self._apply_order_delta(working_positions, pending)
        conservative_positions = dict(working_positions)

        for order in orders:
            snapshot_positions = dict(working_positions)
            if not allow_cross_instrument_netting:
                snapshot_positions = dict(conservative_positions)
                snapshot_positions[order.instrument] = working_positions.get(
                    order.instrument,
                    snapshot_positions.get(order.instrument, 0.0),
                )
            working_snapshot = PortfolioSnapshot(
                cash=portfolio.cash,
                nav=portfolio.nav,
                positions=snapshot_positions,
                prices=portfolio.prices,
                margin_used=portfolio.margin_used,
                buying_power=portfolio.buying_power,
            )
            decision = self.evaluate(
                order,
                portfolio=working_snapshot,
                contract_spec_resolver=contract_spec_resolver,
                pending_orders=(),
            )
            decisions.append(decision)
            if decision.approved:
                approved.append(order)
                self._apply_order_delta(working_positions, order)
                if not allow_cross_instrument_netting:
                    instrument = order.instrument
                    planned = working_positions.get(instrument, 0.0)
                    conservative = conservative_positions.get(instrument, 0.0)
                    if abs(planned) > abs(conservative) + self.tolerance:
                        conservative_positions[instrument] = planned
            else:
                rejected.append(order)

        return PreTradeResult(
            approved_orders=tuple(approved),
            rejected_orders=tuple(rejected),
            decisions=tuple(decisions),
        )

    @staticmethod
    def _apply_order_delta(positions: dict[str, float], order: Order) -> None:
        signed = (
            float(order.remaining_qty)
            if order.side == OrderSide.BUY
            else -float(order.remaining_qty)
        )
        positions[order.instrument] = positions.get(order.instrument, 0.0) + signed

    def _projected_requirements(
        self,
        positions: Mapping[str, float],
        prices: Mapping[str, float | None],
        nav: float,
        contract_spec_resolver: ContractSpecResolver,
    ) -> tuple[float, float]:
        margin = 0.0
        gross_notional = 0.0
        for instrument, quantity in positions.items():
            if abs(quantity) <= self.tolerance:
                continue
            price = prices.get(instrument)
            if price is None or not np.isfinite(float(price)):
                # A position without a usable mark cannot safely increase.
                return float("inf"), float("inf")
            spec = contract_spec_resolver(instrument)
            numeric_price = float(price)
            if numeric_price <= 0.0 and not (
                spec.asset_class == AssetClass.FUTURE and not spec.inverse
            ):
                return float("inf"), float("inf")
            margin += float(spec.margin_required(quantity, numeric_price))
            gross_notional += float(spec.notional(quantity, numeric_price))
        gross_leverage = (
            gross_notional / abs(float(nav))
            if abs(float(nav)) > self.tolerance
            else (0.0 if gross_notional <= self.tolerance else float("inf"))
        )
        return margin, gross_leverage

    def _decision(
        self,
        *,
        approved: bool,
        reason: str | None,
        positions: Mapping[str, float],
        portfolio: PortfolioSnapshot,
        contract_spec_resolver: ContractSpecResolver,
    ) -> PreTradeDecision:
        margin, gross = self._projected_requirements(
            positions,
            portfolio.prices,
            portfolio.nav,
            contract_spec_resolver,
        )
        return PreTradeDecision(
            approved=approved,
            reason=reason,
            projected_margin=margin,
            projected_buying_power=float(portfolio.nav) - margin,
            projected_gross_leverage=gross,
        )

    @staticmethod
    def _reject(reason: str) -> PreTradeDecision:
        return PreTradeDecision(
            approved=False,
            reason=reason,
            projected_margin=float("nan"),
            projected_buying_power=float("nan"),
            projected_gross_leverage=float("nan"),
        )


__all__ = ["PreTradeDecision", "PreTradeResult", "PreTradeRisk"]
