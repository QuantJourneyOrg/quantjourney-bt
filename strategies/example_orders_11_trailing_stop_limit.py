# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 11 - Trailing Stop-Limit
=======================================

Mode: orders.
Order types: MARKET entry, STOP_TRAIL_LIMIT exit.
Idea: trend entry with a trailing stop that converts to a limit order.
Universe: five large stocks.

The stop follows price upward. When triggered, the order sells only at or above
the activated limit price, so gaps can leave the order open.

Usage:
    ./strategy.sh example_orders_11_trailing_stop_limit
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.execution.commission import FixedBpsCommission
from backtester.execution.order_types import Order, OrderSide, OrderType
from backtester.execution.slippage import FixedBpsSlippage


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class TrailingStopLimitTrend(Backtester):
    """SMA trend entry protected by a trailing stop-limit exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_signal = {}
        self._has_trail_limit = {}

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        sma_fast = self.instruments_data.get_feature("SMA_20_close")
        sma_slow = self.instruments_data.get_feature("SMA_50_close")
        if date not in sma_fast.index:
            return

        for inst in self.instruments:
            fast = sma_fast.loc[date, inst]
            slow = sma_slow.loc[date, inst]
            if pd.isna(fast) or pd.isna(slow):
                continue

            bar = bars[inst]
            pos = current_positions.get(inst, 0.0)
            signal = 1 if fast > slow else 0
            prev = self._prev_signal.get(inst, 0)

            if pos == 0 and self._has_trail_limit.get(inst, False):
                self._has_trail_limit[inst] = False

            if signal == 1 and prev == 0 and pos == 0:
                shares = int(nav * 0.15 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(inst, OrderSide.BUY, shares, OrderType.MARKET))
            elif pos > 0 and not self._has_trail_limit.get(inst, False):
                self.fill_engine.submit(Order(
                    inst,
                    OrderSide.SELL,
                    pos,
                    OrderType.STOP_TRAIL_LIMIT,
                    trail_amount=round(bar.close * 0.04, 2),
                    limit_offset=round(bar.close * 0.005, 2),
                ))
                self._has_trail_limit[inst] = True
            elif signal == 0 and prev == 1 and pos > 0:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                self._has_trail_limit[inst] = False

            self._prev_signal[inst] = signal


async def main() -> None:
    strategy = TrailingStopLimitTrend(
        **_credentials(),
        strategy_name="ExampleOrders11_TrailingStopLimit",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period={"start": "2020-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.25,
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [20, 50]}},
        ],
        slippage_model=FixedBpsSlippage(bps=5.0),
        commission_scheme=FixedBpsCommission(bps=1.0),
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
