"""Deterministic portfolio accounting for simulated fills and target weights.

The stateful :class:`PortfolioLedger` is used by event-driven order
simulation. ``build_weight_ledger`` preserves the existing vectorized
target-weight accounting convention while publishing the same result schema.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtester.execution.contract_spec import AssetClass, ContractSpec
from backtester.execution.order_types import Fill, OrderSide

ContractSpecResolver = Callable[[str], ContractSpec]


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Immutable pre-trade view of the simulated portfolio."""

    cash: float
    nav: float
    positions: Mapping[str, float]
    prices: Mapping[str, float | None]
    margin_used: float
    buying_power: float


@dataclass(frozen=True)
class FillAccounting:
    """Accounting values produced while applying one fill."""

    trade_notional: float
    transaction_cash_value: float
    signed_quantity: float
    previous_position: float
    new_position: float


@dataclass(frozen=True)
class LedgerResult:
    """Common accounting result returned by every simulation mode."""

    nav: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    position_values: pd.DataFrame
    weights: pd.DataFrame
    book_weights: pd.DataFrame
    exposure_values: pd.DataFrame
    exposure_weights: pd.DataFrame
    returns: pd.Series
    margin_by_instrument: pd.DataFrame
    margin_used: pd.Series
    buying_power: pd.Series
    average_entry_price: Mapping[str, float | None] = field(default_factory=dict)
    average_entry_price_history: pd.DataFrame | None = None


class PortfolioLedger:
    """Single owner of cash, positions, marks and average entry prices."""

    def __init__(
        self,
        *,
        initial_cash: float,
        instruments: Sequence[str],
        contract_spec_resolver: ContractSpecResolver,
        settlement_currency: str = "USD",
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.instruments = list(instruments)
        self.settlement_currency = settlement_currency

        def _validated_contract_spec(instrument: str) -> ContractSpec:
            spec = contract_spec_resolver(instrument)
            spec.validate_settlement_currency(self.settlement_currency)
            return spec

        self._contract_spec = _validated_contract_spec
        # Validate the complete universe before any fill mutates cash.
        for instrument in self.instruments:
            self._contract_spec(instrument)
        self.reset()

    def reset(self) -> None:
        """Reset all per-run state while preserving configuration."""
        self.cash = self.initial_cash
        self.positions: dict[str, float] = {instrument: 0.0 for instrument in self.instruments}
        self.average_entry_price: dict[str, float | None] = {
            instrument: None for instrument in self.instruments
        }
        self.last_valid_price: dict[str, float | None] = {
            instrument: None for instrument in self.instruments
        }
        self._dates = []
        self._nav_rows = []
        self._cash_rows = []
        self._position_rows = []
        self._position_value_rows = []
        self._exposure_value_rows = []
        self._average_entry_rows = []
        self._margin_rows = []
        self._last_nav = self.initial_cash

    def contract_spec(self, instrument: str) -> ContractSpec:
        """Resolve the immutable contract specification for an instrument."""
        return self._contract_spec(instrument)

    def is_valid_price(self, value: object, instrument: str | None = None) -> bool:
        """Return whether a mark can be used for the instrument.

        Finite zero and negative prices are legal for non-inverse futures.
        Other assets, including inverse contracts, require a positive price.
        """
        try:
            numeric = float(value)
            if not np.isfinite(numeric):
                return False
            if instrument is None:
                return numeric > 0.0
            spec = self._contract_spec(instrument)
            if spec.asset_class == AssetClass.FUTURE and not spec.inverse:
                return True
            return numeric > 0.0
        except (TypeError, ValueError, OverflowError):
            return False

    def observe_mark(self, instrument: str, price: object) -> None:
        """Remember the latest usable market price, including futures zero."""
        if self.is_valid_price(price, instrument):
            self.last_valid_price[instrument] = float(price)

    def price_for_valuation(self, instrument: str, price: object) -> float | None:
        """Resolve current, last-valid, then entry price for valuation."""
        if self.is_valid_price(price, instrument):
            return float(price)
        last = self.last_valid_price.get(instrument)
        if last is not None and self.is_valid_price(last, instrument):
            return float(last)
        entry = self.average_entry_price.get(instrument)
        if entry is not None and self.is_valid_price(entry, instrument):
            return float(entry)
        return None

    def trade_notional(self, instrument: str, quantity: float, price: float) -> float:
        """Absolute exposure notional for commissions and audit."""
        return self._contract_spec(instrument).notional(quantity, price)

    def trade_cash_value(self, instrument: str, quantity: float, price: float) -> float:
        """Unsigned-quantity transaction value with the market price sign."""
        spec = self._contract_spec(instrument)
        if spec.inverse:
            return spec.notional(quantity, price)
        return abs(float(quantity)) * float(price) * float(spec.multiplier) * float(spec.lot_size)

    def signed_position_value(self, instrument: str, quantity: float, price: float) -> float:
        """Mark a signed position using its instrument contract convention."""
        spec = self._contract_spec(instrument)
        if spec.inverse:
            return float(-np.sign(quantity) * spec.notional(quantity, price))
        return float(quantity) * float(price) * float(spec.multiplier) * float(spec.lot_size)

    def apply_fill(self, fill: Fill) -> FillAccounting:
        """Apply one fill atomically to positions, average entry and cash."""
        previous_position = float(self.positions.get(fill.instrument, 0.0))
        signed_quantity = (
            float(fill.quantity) if fill.side == OrderSide.BUY else -float(fill.quantity)
        )
        new_position = previous_position + signed_quantity
        self.positions[fill.instrument] = new_position
        self.average_entry_price[fill.instrument] = self.updated_average_entry_price(
            previous_position=previous_position,
            previous_avg_price=self.average_entry_price.get(fill.instrument),
            signed_fill_qty=signed_quantity,
            fill_price=float(fill.fill_price),
        )

        notional = self.trade_notional(
            fill.instrument, float(fill.quantity), float(fill.fill_price)
        )
        cash_value = self.trade_cash_value(
            fill.instrument, float(fill.quantity), float(fill.fill_price)
        )
        spec = self._contract_spec(fill.instrument)
        signed_cost = -notional if spec.inverse else cash_value
        if fill.side == OrderSide.BUY:
            self.cash -= signed_cost + float(fill.commission)
        else:
            self.cash += signed_cost - float(fill.commission)

        return FillAccounting(
            trade_notional=float(notional),
            transaction_cash_value=float(cash_value),
            signed_quantity=signed_quantity,
            previous_position=previous_position,
            new_position=new_position,
        )

    def mark_to_market(self, prices: Mapping[str, object]) -> tuple[float, dict[str, float]]:
        """Return NAV and per-instrument values at current/frozen marks."""
        position_values: dict[str, float] = {}
        total_position_value = 0.0
        for instrument in self.instruments:
            quantity = float(self.positions.get(instrument, 0.0))
            if abs(quantity) <= 1e-12:
                position_values[instrument] = 0.0
                continue
            price = self.price_for_valuation(instrument, prices.get(instrument))
            if price is None:
                position_values[instrument] = 0.0
                continue
            value = self.signed_position_value(instrument, quantity, price)
            position_values[instrument] = value
            total_position_value += value
        nav = float(self.cash + total_position_value)
        self._last_nav = nav
        return nav, position_values

    def margin_by_instrument(self, prices: Mapping[str, object] | None = None) -> dict[str, float]:
        """Compute current collateral requirement per instrument."""
        prices = prices or {}
        result: dict[str, float] = {}
        for instrument in self.instruments:
            quantity = float(self.positions.get(instrument, 0.0))
            if abs(quantity) <= 1e-12:
                result[instrument] = 0.0
                continue
            price = self.price_for_valuation(instrument, prices.get(instrument))
            if price is None:
                result[instrument] = 0.0
                continue
            result[instrument] = float(
                self._contract_spec(instrument).margin_required(quantity, price)
            )
        return result

    def exposure_values(self, prices: Mapping[str, object] | None = None) -> dict[str, float]:
        """Return signed economic exposure, independent of accounting marks."""
        prices = prices or {}
        result: dict[str, float] = {}
        for instrument in self.instruments:
            quantity = float(self.positions.get(instrument, 0.0))
            if abs(quantity) <= 1e-12:
                result[instrument] = 0.0
                continue
            price = self.price_for_valuation(instrument, prices.get(instrument))
            if price is None:
                result[instrument] = 0.0
                continue
            result[instrument] = float(np.sign(quantity)) * self._contract_spec(
                instrument
            ).notional(quantity, price)
        return result

    def exposure_weights(
        self,
        *,
        prices: Mapping[str, object] | None = None,
        nav: float | None = None,
    ) -> dict[str, float]:
        """Return signed notional exposure divided by current NAV."""
        current_nav = self._last_nav if nav is None else float(nav)
        if abs(current_nav) <= 1e-12:
            return {instrument: 0.0 for instrument in self.instruments}
        return {
            instrument: value / current_nav
            for instrument, value in self.exposure_values(prices).items()
        }

    def snapshot(
        self,
        *,
        prices: Mapping[str, object] | None = None,
        nav: float | None = None,
    ) -> PortfolioSnapshot:
        """Publish an immutable view for pre-trade checks."""
        resolved_prices = {
            instrument: self.price_for_valuation(instrument, (prices or {}).get(instrument))
            for instrument in self.instruments
        }
        margin = self.margin_by_instrument(prices)
        current_nav = self._last_nav if nav is None else float(nav)
        margin_used = float(sum(margin.values()))
        return PortfolioSnapshot(
            cash=float(self.cash),
            nav=current_nav,
            positions=dict(self.positions),
            prices=resolved_prices,
            margin_used=margin_used,
            buying_power=current_nav - margin_used,
        )

    def record(
        self,
        *,
        date: object,
        nav: float,
        position_values: Mapping[str, float],
        prices: Mapping[str, object],
    ) -> None:
        """Append one end-of-bar accounting snapshot."""
        margin = self.margin_by_instrument(prices)
        self._dates.append(date)
        self._nav_rows.append(float(nav))
        self._cash_rows.append(float(self.cash))
        self._position_rows.append(dict(self.positions))
        self._position_value_rows.append(dict(position_values))
        self._exposure_value_rows.append(self.exposure_values(prices))
        self._average_entry_rows.append(dict(self.average_entry_price))
        self._margin_rows.append(margin)

    def result(self) -> LedgerResult:
        """Build immutable pandas outputs and validate their identity."""
        index = pd.Index(self._dates)
        nav = pd.Series(self._nav_rows, index=index, dtype=float, name="nav")
        cash = pd.Series(self._cash_rows, index=index, dtype=float, name="cash")
        positions = pd.DataFrame(
            self._position_rows, index=index, columns=self.instruments, dtype=float
        ).fillna(0.0)
        position_values = pd.DataFrame(
            self._position_value_rows,
            index=index,
            columns=self.instruments,
            dtype=float,
        ).fillna(0.0)
        exposure_values = pd.DataFrame(
            self._exposure_value_rows,
            index=index,
            columns=self.instruments,
            dtype=float,
        ).fillna(0.0)
        average_entry = pd.DataFrame(
            self._average_entry_rows,
            index=index,
            columns=self.instruments,
            dtype=float,
        )
        margin_by_instrument = pd.DataFrame(
            self._margin_rows, index=index, columns=self.instruments, dtype=float
        ).fillna(0.0)
        margin_used = margin_by_instrument.sum(axis=1).rename("margin_used")
        buying_power = (nav - margin_used).rename("buying_power")
        book_weights = weights_from_position_values(position_values, nav)
        exposure_weights = weights_from_position_values(exposure_values, nav)
        returns = nav.pct_change().fillna(0.0).rename("returns")

        _assert_accounting_identity(nav, cash, position_values)
        return LedgerResult(
            nav=nav,
            cash=cash,
            positions=positions,
            position_values=position_values,
            weights=exposure_weights,
            book_weights=book_weights,
            exposure_values=exposure_values,
            exposure_weights=exposure_weights,
            returns=returns,
            margin_by_instrument=margin_by_instrument,
            margin_used=margin_used,
            buying_power=buying_power,
            average_entry_price=dict(self.average_entry_price),
            average_entry_price_history=average_entry,
        )

    @staticmethod
    def updated_average_entry_price(
        *,
        previous_position: float,
        previous_avg_price: float | None,
        signed_fill_qty: float,
        fill_price: float,
    ) -> float | None:
        """Update average entry after scale-in, reduction, close or reversal."""
        new_position = previous_position + signed_fill_qty
        if abs(new_position) <= 1e-12:
            return None
        if abs(previous_position) <= 1e-12:
            return fill_price

        same_direction = (previous_position > 0 and signed_fill_qty > 0) or (
            previous_position < 0 and signed_fill_qty < 0
        )
        if same_direction:
            previous_abs = abs(previous_position)
            fill_abs = abs(signed_fill_qty)
            previous_avg = fill_price if previous_avg_price is None else previous_avg_price
            return (previous_avg * previous_abs + fill_price * fill_abs) / (previous_abs + fill_abs)

        if (previous_position > 0 and new_position > 0) or (
            previous_position < 0 and new_position < 0
        ):
            return previous_avg_price
        return fill_price


def weights_from_position_values(position_values: pd.DataFrame, nav: pd.Series) -> pd.DataFrame:
    """Return finite realized weights, including when NAV is zero."""
    return position_values.divide(nav, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _margin_history(
    *,
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    contract_spec_resolver: ContractSpecResolver,
) -> pd.DataFrame:
    marked_prices = prices.reindex(index=positions.index, columns=positions.columns).ffill()
    margin = pd.DataFrame(0.0, index=positions.index, columns=positions.columns, dtype=float)
    for instrument in positions.columns:
        spec = contract_spec_resolver(instrument)
        for date in positions.index:
            quantity = float(positions.loc[date, instrument])
            price = marked_prices.loc[date, instrument]
            if abs(quantity) <= 1e-12 or pd.isna(price) or not np.isfinite(float(price)):
                continue
            margin.loc[date, instrument] = spec.margin_required(quantity, float(price))
    return margin


def build_weight_ledger(
    *,
    actual_weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    prices: pd.DataFrame,
    initial_capital: float,
    rebalance_flags: pd.Series,
    cost_model: object,
    contract_spec_resolver: ContractSpecResolver,
    settlement_currency: str = "USD",
) -> tuple[LedgerResult, object, pd.DataFrame]:
    """Build the compatibility ledger for vectorized target-weight mode.

    This intentionally preserves the existing close-to-close weight accounting
    and implied-trade cost model. It does not claim that implied trades are
    broker orders or fills.
    """
    gross_nav = float(initial_capital) * (1.0 + portfolio_returns).cumprod()
    if len(gross_nav) > 0:
        gross_nav.iloc[0] = float(initial_capital)
    cost_breakdown = cost_model.compute(
        actual_weights=actual_weights,
        prices=prices,
        nav=gross_nav,
        rebalance_flags=rebalance_flags,
    )
    net_returns = portfolio_returns - cost_breakdown.total_cost_pct
    nav = float(initial_capital) * (1.0 + net_returns).cumprod()

    marked_prices = prices.reindex(
        index=actual_weights.index, columns=actual_weights.columns
    ).ffill()
    desired_exposure = actual_weights.multiply(nav, axis=0)
    positions = pd.DataFrame(
        0.0,
        index=actual_weights.index,
        columns=actual_weights.columns,
        dtype=float,
    )
    position_values = positions.copy()
    exposure_values = positions.copy()
    for instrument in actual_weights.columns:
        spec = contract_spec_resolver(instrument)
        spec.validate_settlement_currency(settlement_currency)
        for date in actual_weights.index:
            price = marked_prices.loc[date, instrument]
            target_value = float(desired_exposure.loc[date, instrument])
            if pd.isna(price) or not np.isfinite(float(price)):
                continue
            unit_notional = float(spec.notional(1.0, float(price)))
            if unit_notional <= 0.0 or not np.isfinite(unit_notional):
                continue
            quantity = target_value / unit_notional
            positions.loc[date, instrument] = quantity
            exposure_values.loc[date, instrument] = float(np.sign(quantity)) * spec.notional(
                quantity, float(price)
            )
            if spec.inverse:
                position_values.loc[date, instrument] = -float(np.sign(quantity)) * spec.notional(
                    quantity, float(price)
                )
            else:
                position_values.loc[date, instrument] = (
                    quantity * float(price) * float(spec.multiplier) * float(spec.lot_size)
                )
    cash = (nav - position_values.sum(axis=1)).rename("cash")
    realized_weights = weights_from_position_values(exposure_values, nav)
    book_weights = weights_from_position_values(position_values, nav)
    margin_by_instrument = _margin_history(
        positions=positions,
        prices=marked_prices,
        contract_spec_resolver=contract_spec_resolver,
    )
    margin_used = margin_by_instrument.sum(axis=1).rename("margin_used")
    buying_power = (nav - margin_used).rename("buying_power")
    returns = net_returns.rename("returns")
    position_changes = positions.diff().fillna(positions)
    position_changes.loc[~rebalance_flags.reindex(positions.index).fillna(False), :] = 0.0

    _assert_accounting_identity(nav, cash, position_values)
    return (
        LedgerResult(
            nav=nav,
            cash=cash,
            positions=positions,
            position_values=position_values,
            weights=realized_weights,
            book_weights=book_weights,
            exposure_values=exposure_values,
            exposure_weights=realized_weights,
            returns=returns,
            margin_by_instrument=margin_by_instrument,
            margin_used=margin_used,
            buying_power=buying_power,
        ),
        cost_breakdown,
        position_changes,
    )


def _assert_accounting_identity(
    nav: pd.Series,
    cash: pd.Series,
    position_values: pd.DataFrame,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-8,
) -> None:
    """Fail fast when common ledger outputs do not reconcile."""
    rhs = cash.reindex(nav.index) + position_values.sum(axis=1).reindex(nav.index)
    if nav.isna().any() or rhs.isna().any():
        raise ValueError("Accounting identity inputs contain missing values")
    if not np.allclose(nav.to_numpy(), rhs.to_numpy(), rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(nav.to_numpy() - rhs.to_numpy())))
        raise AssertionError(f"NAV accounting identity failed; max_abs_diff={max_abs:.12g}")
