# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 18 - Signal-Change Event-Driven Rotation
======================================================

Mode: orders.
Order type: MARKET.
Idea: an event-driven strategy with no calendar at all. Each name has an SMA
trend signal; the strategy trades only when a signal flips — buy on a flat->long
flip, sell on a long->flat flip. Between flips it does nothing.
Universe: three predeclared liquid ETFs: SPY, QQQ and IWM.

This is the order-mode counterpart to the weights "signal-change" policy: the
rebalance trigger is the event (the signal flip), and turnover is driven purely
by how often trends change — the lowest-churn way to run a trend strategy.

Usage:
    ./strategy.sh example_orders_18_signal_change_rotation_orders
"""

import asyncio
import os

import pandas as pd

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


class SignalChangeRotationOrders(Backtester):
    """Trade only on SMA trend-signal flips, in order mode."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_signal = {}

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        fast = self.instruments_data.get_feature("SMA_20_close")
        slow = self.instruments_data.get_feature("SMA_50_close")
        if date not in fast.index:
            return

        for inst in self.instruments:
            f = fast.loc[date, inst]
            s = slow.loc[date, inst]
            if pd.isna(f) or pd.isna(s):
                continue

            signal = 1 if f > s else 0
            prev = self._prev_signal.get(inst, 0)
            pos = current_positions.get(inst, 0.0)

            # Event: flat -> long flip.
            if signal == 1 and prev == 0 and pos == 0:
                shares = int(nav * 0.18 / bars[inst].close)
                if shares > 0:
                    self.fill_engine.submit(Order(inst, OrderSide.BUY, shares, OrderType.MARKET))
            # Event: long -> flat flip.
            elif signal == 0 and prev == 1 and pos > 0:
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))

            self._prev_signal[inst] = signal


async def main() -> None:
    strategy = SignalChangeRotationOrders(
        **_credentials(),
        strategy_name="ExampleOrders18_SignalChangeRotationOrders",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM"],
        backtest_period={"start": "2001-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.20,
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [20, 50]}},
        ],
        slippage_model=FixedBpsSlippage(bps=3.0),
        commission_scheme=PerShareCommission(cost_per_share=0.005, min_per_order=1.0),
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
