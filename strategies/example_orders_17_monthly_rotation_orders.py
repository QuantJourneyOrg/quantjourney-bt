# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 17 - Monthly Event-Driven Rotation
=================================================

Mode: orders.
Order type: MARKET.
Idea: an event-driven monthly rotation executed with explicit orders. On the
first bar of each new month, rank the universe by 6-month momentum, then sell
names that dropped out of the top set and buy the new entrants at market.
Universe: broad asset-class ETFs.
Rebalance: event-driven — the "rebalance" is a month-boundary event handled
inside _compute_orders, not the weights RebalancePolicy engine.

This shows how to express calendar rebalancing in order mode: you detect the
event yourself and emit the trades, getting realistic fills, slippage and
commissions on every rotation.

Usage:
    ./strategy.sh example_orders_17_monthly_rotation_orders
"""

import asyncio
import os

from backtester import Backtester
from backtester.execution.commission import PerShareCommission
from backtester.execution.order_types import Order, OrderSide, OrderType
from backtester.execution.slippage import FixedBpsSlippage


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class MonthlyRotationOrders(Backtester):
    """Event-driven monthly momentum rotation in order mode."""

    LOOKBACK = 126  # ~6 months
    TOP_N = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_month = None

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        # Rebalance event = first bar of a new month.
        month_key = (date.year, date.month)
        if month_key == self._last_month:
            return
        self._last_month = month_key

        close = self.instruments_data.get_feature("adj_close")
        if date not in close.index:
            return
        pos_idx = close.index.get_loc(date)
        if pos_idx < self.LOOKBACK:
            return

        momentum = close.iloc[pos_idx] / close.iloc[pos_idx - self.LOOKBACK] - 1.0
        target = set(momentum.dropna().nlargest(self.TOP_N).index)

        # Sell names that fell out of the target set.
        for inst in self.instruments:
            pos = current_positions.get(inst, 0.0)
            if pos > 0 and inst not in target:
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))

        # Buy new entrants at an equal share of NAV.
        per_name_nav = nav / max(1, len(target))
        for inst in target:
            pos = current_positions.get(inst, 0.0)
            if pos == 0:
                shares = int(per_name_nav / bars[inst].close)
                if shares > 0:
                    self.fill_engine.submit(Order(inst, OrderSide.BUY, shares, OrderType.MARKET))


async def main() -> None:
    strategy = MonthlyRotationOrders(
        **_credentials(),
        strategy_name="ExampleOrders17_MonthlyRotationOrders",
        strategy_type="Momentum Rotation",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC"],
        backtest_period={"start": "2012-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.40,
        indicators_config=[],
        slippage_model=FixedBpsSlippage(bps=3.0),
        commission_scheme=PerShareCommission(cost_per_share=0.005, min_per_order=1.0),
        benchmark_symbol="SPY",
        benchmark_name="S&P 500 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
