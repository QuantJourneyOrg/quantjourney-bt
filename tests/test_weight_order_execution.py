# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Integration tests for execution-aware target-weight backtests."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester import Backtester
from backtester.execution import (
    FixedBpsCommission,
    FixedBpsSlippage,
    OrderStatus,
)
from backtester.execution.contract_spec import ContractSpec
from backtester.portfolio.rebalance import (
    ExecutionRebalanceEngine,
    RebalancePolicy,
)
from backtester.risk import PreTradeRisk


class _Data:
    def __init__(self, frames):
        self.frames = frames

    def get_feature(self, *args):
        key = args if len(args) > 1 else args[0]
        if key not in self.frames:
            raise KeyError(key)
        return self.frames[key]


class _Portfolio:
    def __init__(self, *, cash_buffer=0.0):
        self.cash_buffer = cash_buffer
        self.periods_per_year = 252

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


class _WeightsStrategy(Backtester):
    def _compute_orders(self, date, bars, current_positions, nav):
        raise AssertionError("weight execution must not invoke order strategy code")


def _run(
    *,
    close,
    weights,
    open_=None,
    volume=None,
    initial_capital=100_000.0,
    contract_specs=None,
    **kwargs,
):
    instruments = list(close.columns)
    strategy = _WeightsStrategy(
        api_key="test",
        instruments=instruments,
        backtest_period={
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
        },
        initial_capital=initial_capital,
        execution_mode="weights",
        weight_execution="orders",
        contract_specs=contract_specs,
        **kwargs,
    )
    open_ = close if open_ is None else open_
    volume = (
        pd.DataFrame(1_000_000.0, index=close.index, columns=instruments)
        if volume is None
        else volume
    )
    strategy.instruments_data = _Data(
        {
            "adj_close": close,
            "close": close,
            "open": open_,
            "high": pd.DataFrame(
                pd.concat([open_, close]).groupby(level=0).max(),
                index=close.index,
                columns=instruments,
            ),
            "low": pd.DataFrame(
                pd.concat([open_, close]).groupby(level=0).min(),
                index=close.index,
                columns=instruments,
            ),
            "volume": volume,
            ("strategies", "cloud_strategy", "weights"): weights,
        }
    )
    strategy.portfolio_data = _Portfolio(cash_buffer=0.0)
    strategy._compute_strategy_performance()
    return strategy


def test_weight_execution_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="weight_execution"):
        _WeightsStrategy(
            api_key="test",
            instruments=["AAPL"],
            backtest_period={"start": "2024-01-01", "end": "2024-01-31"},
            weight_execution="vector-ish",
        )

    legacy = _WeightsStrategy(
        api_key="test",
        instruments=["AAPL"],
        backtest_period={"start": "2024-01-01", "end": "2024-01-31"},
    )
    assert legacy.weight_execution == "fast"
    assert legacy.fill_engine is None


def test_target_from_close_t_fills_at_next_close_not_same_bar() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"AAPL": [100.0, 110.0, 120.0]}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=dates)

    strategy = _run(
        close=close,
        weights=weights,
        pre_trade_risk=PreTradeRisk(),
    )

    first_order = strategy.fill_engine.order_history[0]
    first_fill = strategy.fill_engine.fill_history[0]
    assert first_order.created_at == dates[0]
    assert first_fill.timestamp == dates[1]
    assert first_fill.fill_price == pytest.approx(110.0)
    assert strategy.portfolio_data.positions.loc[dates[0], "AAPL"] == 0.0
    assert strategy.portfolio_data.positions.loc[dates[1], "AAPL"] == pytest.approx(1_000.0)
    assert strategy.portfolio_data.rebalance_decision_flags.loc[dates[0]]
    assert strategy.portfolio_data.rebalance_fill_flags.loc[dates[1]]


def test_explicit_open_fill_uses_next_bar_open() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"AAPL": [100.0, 110.0, 120.0]}, index=dates)
    open_ = pd.DataFrame({"AAPL": [95.0, 80.0, 115.0]}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=dates)

    strategy = _run(close=close, open_=open_, weights=weights, fill_at="open")

    assert strategy.fill_engine.fill_history[0].timestamp == dates[1]
    assert strategy.fill_engine.fill_history[0].fill_price == pytest.approx(80.0)
    assert strategy.portfolio_data.net_asset_value.loc[dates[1]] == pytest.approx(130_000.0)


def test_futures_target_uses_multiplier_and_reports_exposure_and_margin() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"ES": [5_000.0] * 3}, index=dates)
    weights = pd.DataFrame({"ES": [0.5] * 3}, index=dates)
    spec = ContractSpec.future("ES", multiplier=50.0, margin=15_840.0)

    strategy = _run(
        close=close,
        weights=weights,
        initial_capital=1_000_000.0,
        contract_specs={"ES": spec},
    )

    fill = strategy.fill_engine.fill_history[0]
    assert fill.quantity == pytest.approx(2.0)
    assert strategy.portfolio_data.positions.iloc[1]["ES"] == pytest.approx(2.0)
    assert strategy.portfolio_data.exposure_weights.iloc[1]["ES"] == pytest.approx(0.5)
    assert strategy.portfolio_data.margin_used.iloc[1] == pytest.approx(31_680.0)
    assert strategy.portfolio_data.buying_power.iloc[1] == pytest.approx(968_320.0)


def test_fx_target_uses_fractional_standard_lots() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"EURUSD": [1.10] * 3}, index=dates)
    weights = pd.DataFrame({"EURUSD": [0.25] * 3}, index=dates)

    strategy = _run(close=close, weights=weights, initial_capital=100_000.0)

    assert strategy.fill_engine.fill_history[0].quantity == pytest.approx(0.22)
    assert strategy.portfolio_data.positions.iloc[1]["EURUSD"] == pytest.approx(0.22)


def test_quantity_rounding_keeps_unallocatable_cash_visible() -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    close = pd.DataFrame({"AAPL": [333.0, 333.0]}, index=dates)
    weights = pd.DataFrame({"AAPL": [0.5, 0.5]}, index=dates)

    strategy = _run(close=close, weights=weights, initial_capital=1_000.0)

    assert strategy.portfolio_data.positions.iloc[1]["AAPL"] == pytest.approx(1.0)
    assert strategy.portfolio_data.cash.iloc[1] == pytest.approx(667.0)
    assert strategy.portfolio_data.exposure_weights.iloc[1]["AAPL"] == pytest.approx(0.333)


def test_batch_pre_trade_rejection_is_audited_and_leaves_book_flat() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"ES": [5_000.0] * 3}, index=dates)
    weights = pd.DataFrame({"ES": [0.5] * 3}, index=dates)
    spec = ContractSpec.future("ES", multiplier=50.0, margin=600_000.0)

    strategy = _run(
        close=close,
        weights=weights,
        initial_capital=1_000_000.0,
        contract_specs={"ES": spec},
        pre_trade_risk=PreTradeRisk(max_margin_utilization=1.0),
    )

    assert (strategy.portfolio_data.positions["ES"] == 0.0).all()
    assert strategy.fill_engine.fill_history == []
    assert strategy.fill_engine.order_history
    assert all(order.status == OrderStatus.REJECTED for order in strategy.fill_engine.order_history)
    assert all(
        "projected margin" in (order.rejection_reason or "")
        for order in strategy.fill_engine.order_history
    )


def test_fill_time_risk_rejects_next_bar_gap_beyond_margin_limit() -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    close = pd.DataFrame({"AAPL": [100.0, 200.0]}, index=dates)
    open_ = pd.DataFrame({"AAPL": [100.0, 200.0]}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)

    strategy = _run(
        close=close,
        open_=open_,
        weights=weights,
        initial_capital=10_000.0,
        fill_at="open",
    )

    assert strategy.fill_engine.fill_history == []
    assert strategy.portfolio_data.positions.iloc[-1]["AAPL"] == 0.0
    rejected = strategy.fill_engine.order_history[0]
    assert rejected.status == OrderStatus.REJECTED
    assert "fill-time pre-trade rejected" in (rejected.rejection_reason or "")
    assert strategy.portfolio_data.margin_used.iloc[-1] <= 10_000.0


def test_unchanged_daily_target_keeps_one_partial_order_without_duplicates() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    close = pd.DataFrame({"AAPL": [10.0] * 4}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0] * 4}, index=dates)
    volume = pd.DataFrame({"AAPL": [10.0] * 4}, index=dates)

    strategy = _run(
        close=close,
        weights=weights,
        initial_capital=1_000.0,
        volume=volume,
        max_volume_participation=0.10,
        commission_scheme=FixedBpsCommission(bps=1.0),
    )

    assert len(strategy.fill_engine.fill_history) == 3
    assert sum(fill.quantity for fill in strategy.fill_engine.fill_history) == pytest.approx(3.0)
    assert strategy.portfolio_data.positions.iloc[-1]["AAPL"] == pytest.approx(3.0)
    assert len(strategy.fill_engine.order_history) == 1
    assert strategy.fill_engine.order_history[0].status == OrderStatus.CANCELLED
    active = [order for order in strategy.fill_engine.pending_orders if order.is_active]
    assert active == []
    assert strategy.fill_engine.order_history[0].remaining_qty == pytest.approx(97.0)
    assert strategy.portfolio_data.total_transaction_costs.sum() > 0.0


def test_pending_target_waits_through_gap_without_ghost_pnl() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    close = pd.DataFrame({"AAPL": [100.0, float("nan"), 120.0, 130.0]}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0] * 4}, index=dates)

    strategy = _run(
        close=close,
        weights=weights,
        initial_capital=10_000.0,
        rebalance_policy=RebalancePolicy(frequency=None),
        pre_trade_risk=PreTradeRisk(),
    )

    assert strategy.fill_engine.fill_history[0].timestamp == dates[2]
    assert strategy.portfolio_data.positions["AAPL"].tolist() == [
        0.0,
        0.0,
        100.0,
        100.0,
    ]
    assert strategy.portfolio_data.net_asset_value.tolist() == pytest.approx(
        [10_000.0, 10_000.0, 10_000.0, 11_000.0]
    )


def test_unchanged_daily_target_keeps_pending_order_through_dark_bar() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    close = pd.DataFrame({"AAPL": [100.0, float("nan"), 120.0, 120.0]}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0] * 4}, index=dates)

    strategy = _run(
        close=close,
        weights=weights,
        initial_capital=10_000.0,
        pre_trade_risk=PreTradeRisk(),
    )

    assert len(strategy.fill_engine.order_history) == 2
    assert strategy.fill_engine.fill_history[0].timestamp == dates[2]
    assert strategy.fill_engine.order_history[0].created_at == dates[0]


def test_partial_fill_persistent_order_reaches_target_without_duplicates() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    close = pd.DataFrame({"AAPL": [100.0] * 5}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0] * 5}, index=dates)
    volume = pd.DataFrame({"AAPL": [100.0] * 5}, index=dates)

    strategy = _run(
        close=close,
        weights=weights,
        volume=volume,
        initial_capital=10_000.0,
        max_volume_participation=0.25,
        rebalance_policy=RebalancePolicy(frequency=None),
    )

    assert len(strategy.fill_engine.order_history) == 1
    assert [fill.quantity for fill in strategy.fill_engine.fill_history] == [25.0] * 4
    assert strategy.portfolio_data.positions.iloc[-1]["AAPL"] == pytest.approx(100.0)


def test_rebalance_does_not_spend_unfilled_cross_asset_sale_proceeds() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    close = pd.DataFrame(
        {
            "A": [100.0, 100.0, float("nan"), 100.0, 100.0],
            "B": [100.0] * 5,
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.0, 0.0, 0.0],
            "B": [0.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=dates,
    )

    strategy = _run(
        close=close,
        weights=weights,
        pre_trade_risk=PreTradeRisk(max_margin_utilization=1.0),
    )

    assert (
        strategy.portfolio_data.margin_used <= strategy.portfolio_data.net_asset_value + 1e-9
    ).all()
    assert strategy.portfolio_data.positions.loc[dates[2], "A"] == 1_000.0
    assert strategy.portfolio_data.positions.loc[dates[2], "B"] == 0.0
    assert strategy.portfolio_data.positions.iloc[-1].to_dict() == {
        "A": 0.0,
        "B": 1_000.0,
    }


def test_total_transaction_costs_include_economic_slippage() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"AAPL": [100.0] * 3}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0] * 3}, index=dates)

    strategy = _run(
        close=close,
        weights=weights,
        initial_capital=10_000.0,
        rebalance_policy=RebalancePolicy(frequency=None),
        slippage_model=FixedBpsSlippage(bps=100.0),
        commission_scheme=FixedBpsCommission(bps=100.0),
        pre_trade_risk=PreTradeRisk(),
    )

    fill = strategy.fill_engine.fill_history[0]
    economic_slippage = fill.slippage * fill.quantity
    expected = fill.commission + economic_slippage
    assert strategy.portfolio_data.net_asset_value.iloc[1] == pytest.approx(9_799.0)
    assert strategy.portfolio_data.total_transaction_costs.iloc[1] == pytest.approx(expected)


def test_turnover_budget_does_not_block_initial_portfolio_build() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"AAPL": [100.0] * 3}, index=dates)
    weights = pd.DataFrame({"AAPL": [1.0] * 3}, index=dates)

    strategy = _run(
        close=close,
        weights=weights,
        rebalance_policy=RebalancePolicy(frequency=None, max_annual_turnover=0.5),
    )

    assert strategy.portfolio_data.positions.iloc[1]["AAPL"] == pytest.approx(1_000.0)
    assert strategy._rebalance_stats["turnover_veto_count"] == 0


def test_circuit_breaker_bypasses_trade_gates_and_waits_for_reconciliation() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    planner = ExecutionRebalanceEngine(
        RebalancePolicy(
            frequency=None,
            drift_threshold=0.50,
            partial_rebalance=True,
            avoid_short_term_gains=True,
            short_term_days=252,
            max_annual_turnover=0.10,
            max_drawdown_trigger=-0.10,
            circuit_breaker_cooldown_days=1,
        ),
        dates=dates,
        instruments=["AAPL"],
    )
    planner.evaluate(
        bar_index=0,
        decision_time=dates[0],
        execution_time=dates[1],
        target_weights={"AAPL": 1.0},
        realized_weights={"AAPL": 1.0},
        positions={"AAPL": 100.0},
        nav=100_000.0,
        available={"AAPL": True},
    )
    breaker = planner.evaluate(
        bar_index=1,
        decision_time=dates[1],
        execution_time=dates[2],
        target_weights={"AAPL": 1.0},
        realized_weights={"AAPL": 1.0},
        positions={"AAPL": 100.0},
        nav=80_000.0,
        available={"AAPL": True},
    )
    assert breaker.reason == "circuit_breaker"
    assert breaker.persistent_weights == {"AAPL": 0.0}
    assert breaker.proposed_turnover == pytest.approx(1.0)

    planner.record_submission(
        timestamp=dates[1],
        submitted=1,
        rejected=0,
        reason="circuit_breaker",
    )
    while_exit_pending = planner.evaluate(
        bar_index=2,
        decision_time=dates[2],
        execution_time=dates[3],
        target_weights={"AAPL": 1.0},
        realized_weights={"AAPL": 0.5},
        positions={"AAPL": 50.0},
        nav=80_000.0,
        available={"AAPL": True},
    )
    assert not while_exit_pending.should_rebalance

    planner.record_target_reconciled(reason="circuit_breaker")
    reentry = planner.evaluate(
        bar_index=3,
        decision_time=dates[3],
        execution_time=dates[4],
        target_weights={"AAPL": 1.0},
        realized_weights={"AAPL": 0.0},
        positions={"AAPL": 0.0},
        nav=80_000.0,
        available={"AAPL": True},
    )
    assert reentry.reason == "post_cooldown"
    planner.record_submission(
        timestamp=dates[3],
        submitted=1,
        rejected=0,
        reason="post_cooldown",
    )
    assert planner._reentry_pending


def test_relative_drift_metric_is_reused_by_partial_rebalance() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    planner = ExecutionRebalanceEngine(
        RebalancePolicy(
            frequency=None,
            drift_threshold=0.50,
            drift_type="relative",
            partial_rebalance=True,
        ),
        dates=dates,
        instruments=["AAPL"],
    )
    planner.evaluate(
        bar_index=0,
        decision_time=dates[0],
        execution_time=dates[1],
        target_weights={"AAPL": 0.01},
        realized_weights={"AAPL": 0.01},
        positions={"AAPL": 1.0},
        nav=100.0,
        available={"AAPL": True},
    )
    decision = planner.evaluate(
        bar_index=1,
        decision_time=dates[1],
        execution_time=dates[2],
        target_weights={"AAPL": 0.01},
        realized_weights={"AAPL": 0.02},
        positions={"AAPL": 2.0},
        nav=100.0,
        available={"AAPL": True},
    )
    assert decision.reason == "drift"
    assert decision.desired_weights["AAPL"] == pytest.approx(0.01)
