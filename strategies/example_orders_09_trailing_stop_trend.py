# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 09 - Trailing Stop Trend
=======================================

Mode: orders.
Order types: MARKET entry, STOP_TRAIL exit.
Idea: enter on SMA trend and let a 4% trailing stop manage the exit.
Universe: three predeclared liquid ETFs: SPY, QQQ and IWM.

The stop ratchets upward as price makes new highs. It does not need manual
updates from the strategy; the FillEngine manages the trailing anchor.

Usage:
    ./strategy.sh example_orders_09_trailing_stop_trend
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


class TrailingStopTrend(Backtester):
    """SMA trend entry with a trailing stop exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_signal = {}
        self._has_trail = {}

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

            if pos == 0 and self._has_trail.get(inst, False):
                self._has_trail[inst] = False

            if signal == 1 and prev == 0 and pos == 0:
                shares = int(nav * 0.15 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(inst, OrderSide.BUY, shares, OrderType.MARKET))
            elif pos > 0 and not self._has_trail.get(inst, False):
                self.fill_engine.submit(
                    Order(
                        inst,
                        OrderSide.SELL,
                        pos,
                        OrderType.STOP_TRAIL,
                        trail_percent=0.04,
                    )
                )
                self._has_trail[inst] = True
            elif signal == 0 and prev == 1 and pos > 0:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                self._has_trail[inst] = False

            self._prev_signal[inst] = signal


async def main() -> None:
    strategy = TrailingStopTrend(
        **_credentials(),
        strategy_name="ExampleOrders09_TrailingStopTrend",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM"],
        backtest_period={"start": "2001-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
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
