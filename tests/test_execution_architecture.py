"""Regression tests for execution simulation, accounting and pre-trade risk.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from backtester.execution import (
    BarData,
    ExecutionSimulator,
    Fill,
    FillEngine,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from backtester.execution.contract_spec import ContractSpec
from backtester.portfolio.accounting import (
    PortfolioLedger,
    PortfolioSnapshot,
    build_weight_ledger,
)
from backtester.portfolio.rebalance import RebalanceEngine, RebalancePolicy
from backtester.portfolio.weight_cost import FixedBpsWeightCostModel
from backtester.risk import PreTradeRisk


class _Data:
    def __init__(self, frames):
        self.frames = frames

    def get_feature(self, name):
        return self.frames[name]


class _Portfolio:
    cash_buffer = 0.0

    def update_net_asset_value(self, nav):
        self.net_asset_value = nav

    def update_positions(self, positions):
        self.positions = positions

    def update_weights(self, weights):
        self.weights = weights

    def update_cash(self, cash):
        self.cash = cash

    def assert_accounting_identity(self):
        expected = self.cash + self.position_values.sum(axis=1)
        pd.testing.assert_series_equal(self.net_asset_value, expected, check_names=False)


def test_public_execution_architecture_imports() -> None:
    assert ExecutionSimulator is not None
    assert PortfolioLedger is not None
    assert PreTradeRisk is not None


def test_accounting_can_be_imported_first_in_fresh_interpreter() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from backtester.portfolio.accounting import PortfolioLedger; "
                "from backtester.execution import ExecutionSimulator; "
                "assert PortfolioLedger and ExecutionSimulator"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_market_fill_validates_only_configured_execution_price() -> None:
    open_engine = FillEngine(fill_at="open")
    open_engine.submit(Order("AAPL", OrderSide.BUY, 1.0))
    open_fills = open_engine.process_bar(
        "AAPL",
        BarData(
            timestamp=pd.Timestamp("2024-01-02"),
            open=100.0,
            high=float("nan"),
            low=float("nan"),
            close=float("nan"),
            volume=1_000.0,
        ),
    )
    assert len(open_fills) == 1
    assert open_fills[0].fill_price == pytest.approx(100.0)

    close_engine = FillEngine(fill_at="close")
    close_engine.submit(Order("AAPL", OrderSide.BUY, 1.0))
    close_fills = close_engine.process_bar(
        "AAPL",
        BarData(
            timestamp=pd.Timestamp("2024-01-02"),
            open=float("nan"),
            high=float("nan"),
            low=float("nan"),
            close=101.0,
            volume=1_000.0,
        ),
    )
    assert len(close_fills) == 1
    assert close_fills[0].fill_price == pytest.approx(101.0)


def test_unobservable_stop_still_ages_day_order() -> None:
    engine = FillEngine()
    order = Order(
        "AAPL",
        OrderSide.BUY,
        1.0,
        order_type=OrderType.STOP,
        stop_price=105.0,
        time_in_force=TimeInForce.DAY,
    )
    engine.submit(order)
    fills = engine.process_bar(
        "AAPL",
        BarData(
            timestamp=pd.Timestamp("2024-01-02"),
            open=100.0,
            high=float("nan"),
            low=99.0,
            close=100.0,
            volume=1_000.0,
        ),
    )
    assert fills == []
    assert order.status == OrderStatus.EXPIRED


def test_conservative_batch_accumulates_all_risk_increases() -> None:
    risk = PreTradeRisk(max_margin_utilization=1.0)
    snapshot = PortfolioSnapshot(
        cash=100.0,
        nav=100.0,
        positions={"A": 0.0, "B": 0.0},
        prices={"A": 1.0, "B": 1.0},
        margin_used=0.0,
        buying_power=100.0,
    )
    result = risk.evaluate_batch(
        [
            Order("A", OrderSide.BUY, 60.0),
            Order("B", OrderSide.BUY, 60.0),
        ],
        portfolio=snapshot,
        contract_spec_resolver=ContractSpec.equity,
        allow_cross_instrument_netting=False,
    )
    assert [decision.approved for decision in result.decisions] == [True, False]


@pytest.mark.parametrize("quantity", [np.nan, np.inf, -np.inf])
def test_fill_engine_rejects_non_finite_direct_quantity(quantity: float) -> None:
    engine = FillEngine()
    with pytest.raises(ValueError, match="finite and positive"):
        engine.submit(Order("AAPL", OrderSide.BUY, quantity))


def test_rejected_pre_trade_order_is_audited_but_not_pending() -> None:
    engine = FillEngine(pre_submit_check=lambda order: "account limit")
    order = Order("AAPL", OrderSide.BUY, 10.0)

    order_id = engine.submit(order)

    assert order_id == order.order_id
    assert order.status == OrderStatus.REJECTED
    assert order.rejection_reason == "account limit"
    assert engine.pending_orders == []
    assert engine.order_history == [order]


def test_ledger_preserves_zero_future_mark_through_gap() -> None:
    spec = ContractSpec.future("CL", multiplier=1_000.0, margin=5_000.0)
    ledger = PortfolioLedger(
        initial_cash=100_000.0,
        instruments=["CL"],
        contract_spec_resolver=lambda instrument: spec,
    )
    dates = pd.date_range("2020-04-17", periods=3, freq="B")

    ledger.observe_mark("CL", 10.0)
    ledger.apply_fill(
        Fill(
            order_id="entry",
            instrument="CL",
            side=OrderSide.BUY,
            quantity=1.0,
            fill_price=10.0,
            timestamp=dates[0],
        )
    )
    nav, values = ledger.mark_to_market({"CL": 10.0})
    ledger.record(date=dates[0], nav=nav, position_values=values, prices={"CL": 10.0})

    ledger.observe_mark("CL", 0.0)
    nav, values = ledger.mark_to_market({"CL": 0.0})
    ledger.record(date=dates[1], nav=nav, position_values=values, prices={"CL": 0.0})

    nav, values = ledger.mark_to_market({"CL": np.nan})
    ledger.record(date=dates[2], nav=nav, position_values=values, prices={"CL": np.nan})
    result = ledger.result()

    assert result.nav.iloc[0] == pytest.approx(100_000.0)
    assert result.nav.iloc[1] == pytest.approx(90_000.0)
    assert result.nav.iloc[2] == pytest.approx(90_000.0)
    assert result.position_values.iloc[2, 0] == pytest.approx(0.0)
    assert result.margin_used.iloc[2] == pytest.approx(5_000.0)
    assert result.buying_power.iloc[2] == pytest.approx(85_000.0)
    pd.testing.assert_series_equal(
        result.nav,
        result.cash + result.position_values.sum(axis=1),
        check_names=False,
    )


def test_backtester_preserves_zero_future_mark_through_gap() -> None:
    class BuyOnce(Backtester):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.submitted = False

        def _compute_orders(self, date, bars, current_positions, nav):
            if not self.submitted:
                self.order_market("CL", 1.0)
                self.submitted = True

    dates = pd.date_range("2020-04-17", periods=4, freq="B")
    prices = pd.DataFrame({"CL": [10.0, 10.0, 0.0, np.nan]}, index=dates)
    frames = {
        "adj_close": prices,
        "open": prices,
        "high": prices,
        "low": prices,
        "volume": pd.DataFrame({"CL": [1_000.0] * 4}, index=dates),
    }
    spec = ContractSpec.future("CL", multiplier=1_000.0, margin=5_000.0)
    strategy = BuyOnce(
        api_key="test",
        instruments=["CL"],
        backtest_period={"start": "2020-04-17", "end": "2020-04-24"},
        execution_mode="orders",
        contract_specs={"CL": spec},
    )
    strategy.instruments_data = _Data(frames)
    strategy.portfolio_data = _Portfolio()

    strategy._compute_performance_order_based()

    assert strategy.portfolio_data.net_asset_value.iloc[2] == pytest.approx(90_000.0)
    assert strategy.portfolio_data.net_asset_value.iloc[3] == pytest.approx(90_000.0)
    assert strategy.portfolio_data.position_values.iloc[3, 0] == pytest.approx(0.0)
    assert strategy.portfolio_data.margin_used.iloc[3] == pytest.approx(5_000.0)


def test_pre_trade_enforces_fixed_future_margin_and_allows_close() -> None:
    spec = ContractSpec.future("ES", multiplier=50.0, margin=60_000.0)
    risk = PreTradeRisk(max_margin_utilization=1.0)

    def resolver(instrument: str) -> ContractSpec:
        return spec

    flat = PortfolioSnapshot(
        cash=100_000.0,
        nav=100_000.0,
        positions={"ES": 0.0},
        prices={"ES": 5_000.0},
        margin_used=0.0,
        buying_power=100_000.0,
    )

    approved = risk.evaluate(
        Order("ES", OrderSide.BUY, 1.0),
        portfolio=flat,
        contract_spec_resolver=resolver,
    )
    rejected = risk.evaluate(
        Order("ES", OrderSide.BUY, 2.0),
        portfolio=flat,
        contract_spec_resolver=resolver,
    )

    assert approved.approved
    assert approved.projected_margin == pytest.approx(60_000.0)
    assert not rejected.approved
    assert "projected margin" in (rejected.reason or "")

    over_limit = PortfolioSnapshot(
        cash=-400_000.0,
        nav=50_000.0,
        positions={"ES": 2.0},
        prices={"ES": 5_000.0},
        margin_used=120_000.0,
        buying_power=-70_000.0,
    )
    close = risk.evaluate(
        Order("ES", OrderSide.SELL, 2.0),
        portfolio=over_limit,
        contract_spec_resolver=resolver,
    )
    assert close.approved
    assert close.projected_margin == pytest.approx(0.0)


def test_execution_simulator_routes_direct_submit_through_pre_trade() -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    prices = pd.DataFrame({"ES": [5_000.0, 5_000.0]}, index=dates)
    volume = pd.DataFrame({"ES": [1_000.0, 1_000.0]}, index=dates)
    spec = ContractSpec.future("ES", multiplier=50.0, margin=60_000.0)
    fill_engine = FillEngine()
    ledger = PortfolioLedger(
        initial_cash=100_000.0,
        instruments=["ES"],
        contract_spec_resolver=lambda instrument: spec,
    )
    simulator = ExecutionSimulator(
        fill_engine=fill_engine,
        ledger=ledger,
        pre_trade_risk=PreTradeRisk(max_margin_utilization=1.0),
    )
    submitted = False

    def on_bar(date, bars, positions, nav, average_entry):
        nonlocal submitted
        if not submitted:
            fill_engine.submit(Order("ES", OrderSide.BUY, 2.0))
            submitted = True

    result = simulator.run(
        close=prices,
        open_=prices,
        high=prices,
        low=prices,
        volume=volume,
        on_bar=on_bar,
    )

    assert result.positions.iloc[-1, 0] == pytest.approx(0.0)
    assert len(fill_engine.order_history) == 1
    assert fill_engine.order_history[0].status == OrderStatus.REJECTED
    assert "projected margin" in (fill_engine.order_history[0].rejection_reason or "")


def test_weight_cost_never_credits_negative_price_trade() -> None:
    dates = pd.date_range("2020-04-17", periods=2, freq="B")
    weights = pd.DataFrame({"CL": [0.0, 1.0]}, index=dates)
    prices = pd.DataFrame({"CL": [10.0, -37.0]}, index=dates)
    nav = pd.Series(100_000.0, index=dates)
    flags = pd.Series(True, index=dates)

    result = FixedBpsWeightCostModel(total_bps=1.0).compute(
        actual_weights=weights,
        prices=prices,
        nav=nav,
        rebalance_flags=flags,
    )

    assert (result.trade_values.to_numpy() >= 0.0).all()
    assert (result.transaction_costs.to_numpy() >= 0.0).all()
    assert result.total_cost.iloc[-1] == pytest.approx(10.0)


def test_fast_weight_initial_trade_cost_is_not_deferred_or_dropped() -> None:
    dates = pd.date_range("2024-01-01", periods=1, freq="B")
    weights = pd.DataFrame({"AAPL": [1.0]}, index=dates)
    prices = pd.DataFrame({"AAPL": [100.0]}, index=dates)
    result, costs, _ = build_weight_ledger(
        actual_weights=weights,
        portfolio_returns=pd.Series(0.0, index=dates),
        prices=prices,
        initial_capital=100_000.0,
        rebalance_flags=pd.Series(True, index=dates),
        cost_model=FixedBpsWeightCostModel(total_bps=100.0),
        contract_spec_resolver=ContractSpec.equity,
    )
    assert costs.total_cost.iloc[0] == pytest.approx(1_000.0)
    assert result.nav.iloc[0] == pytest.approx(99_000.0)
    assert result.returns.iloc[0] == pytest.approx(-0.01)


def test_weight_cost_does_not_reenter_unchanged_position_after_gap() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    weights = pd.DataFrame({"AAPL": [1.0] * 4}, index=dates)
    prices = pd.DataFrame({"AAPL": [10.0, np.nan, np.nan, 10.0]}, index=dates)
    nav = pd.Series(100_000.0, index=dates)
    flags = pd.Series(True, index=dates)

    result = FixedBpsWeightCostModel(total_bps=1.0).compute(
        actual_weights=weights,
        prices=prices,
        nav=nav,
        rebalance_flags=flags,
    )

    assert result.total_cost.iloc[0] == pytest.approx(10.0)
    assert result.total_cost.iloc[1:].sum() == pytest.approx(0.0)
    assert math.isfinite(float(result.total_cost.sum()))


def test_tax_aware_holding_age_tracks_short_increases_not_covers() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    targets = pd.DataFrame({"PAIR": [-0.20, -0.50, -0.10]}, index=dates)
    returns = pd.DataFrame({"PAIR": [0.0, 0.0, 0.0]}, index=dates)
    engine = RebalanceEngine(
        RebalancePolicy(
            frequency="D",
            avoid_short_term_gains=True,
            short_term_days=2,
        )
    )

    actual, _ = engine.run(targets, returns)

    assert actual.iloc[0, 0] == pytest.approx(-0.20)
    assert actual.iloc[1, 0] == pytest.approx(-0.50)
    # Cover one bar after increasing the short is blocked as a young lot.
    assert actual.iloc[2, 0] == pytest.approx(-0.50)
