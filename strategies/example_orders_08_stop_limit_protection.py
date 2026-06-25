# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 08 - Stop-Limit Protection
=========================================

Mode: orders.
Order types: MARKET entry, STOP_LIMIT exit.
Idea: enter an SMA trend, then protect the downside with a stop-limit sell.
Universe: five liquid technology stocks.

A stop-limit exit avoids selling below the limit price, but it can remain
unfilled after a gap. That fill risk is the point of this example.

Usage:
    ./strategy.sh example_orders_08_stop_limit_protection
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


class StopLimitProtection(Backtester):
    """SMA trend entry with a protective stop-limit exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_signal = {}
        self._has_stop_limit = {}

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

            if pos == 0 and self._has_stop_limit.get(inst, False):
                self._has_stop_limit[inst] = False

            if signal == 1 and prev == 0 and pos == 0:
                shares = int(nav * 0.15 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(inst, OrderSide.BUY, shares, OrderType.MARKET))
            elif pos > 0 and not self._has_stop_limit.get(inst, False):
                entry = self.get_average_entry_price(inst)
                if entry is not None:
                    self.fill_engine.submit(Order(
                        inst,
                        OrderSide.SELL,
                        pos,
                        OrderType.STOP_LIMIT,
                        stop_price=round(entry * 0.96, 2),
                        limit_price=round(entry * 0.955, 2),
                    ))
                    self._has_stop_limit[inst] = True
            elif signal == 0 and prev == 1 and pos > 0:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                self._has_stop_limit[inst] = False

            self._prev_signal[inst] = signal


async def main() -> None:
    strategy = StopLimitProtection(
        **_credentials(),
        strategy_name="ExampleOrders08_StopLimitProtection",
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
