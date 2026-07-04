# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 05 - Stop Breakout Entry
=======================================

Mode: orders.
Order type: STOP.
Idea: place buy-stop orders above the recent 20-day high.
Universe: five large US stocks.

A buy stop waits for price confirmation. If the breakout triggers, the
position is held for 10 bars and then exited at market.

Usage:
    ./strategy.sh example_orders_05_stop_breakout_entry
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


class StopBreakoutEntry(Backtester):
    """Buy-stop breakout strategy with a fixed 10-bar hold."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bar_count = {}
        self._entry_bar = {}

    def _has_pending(self, instrument: str) -> bool:
        return any(o.instrument == instrument and o.is_active for o in self.fill_engine.pending_orders)

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        close = self.instruments_data.get_feature("adj_close")
        if date not in close.index:
            return

        idx = close.index.get_loc(date)
        if idx < 21:
            return

        for inst in self.instruments:
            self._bar_count[inst] = self._bar_count.get(inst, 0) + 1
            bar = bars[inst]
            pos = current_positions.get(inst, 0.0)

            if pos > 0 and inst not in self._entry_bar:
                self._entry_bar[inst] = self._bar_count[inst]
            if pos == 0:
                self._entry_bar.pop(inst, None)

            if pos > 0 and self._bar_count[inst] - self._entry_bar.get(inst, 0) >= 10:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                self._entry_bar.pop(inst, None)
                continue

            if pos == 0 and not self._has_pending(inst):
                prior_high = close[inst].iloc[idx - 20 : idx].max()
                if pd.isna(prior_high):
                    continue
                stop_price = round(prior_high * 1.002, 2)
                shares = int(nav * 0.10 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(
                        inst,
                        OrderSide.BUY,
                        shares,
                        OrderType.STOP,
                        stop_price=stop_price,
                        expires_after_bars=5,
                    ))


async def main() -> None:
    strategy = StopBreakoutEntry(
        **_credentials(),
        strategy_name="ExampleOrders05_StopBreakoutEntry",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period={"start": "2020-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.20,
        indicators_config=[],
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
