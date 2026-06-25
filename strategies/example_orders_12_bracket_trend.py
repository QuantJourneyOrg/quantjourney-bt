# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 12 - Bracket Trend
=================================

Mode: orders.
Order type: BRACKET.
Idea: enter on SMA trend and attach take-profit plus stop-loss exits.
Universe: five large technology stocks.

The bracket parent enters at market. The engine then activates a limit
take-profit and a stop-loss as an OCO pair.

Usage:
    ./strategy.sh example_orders_12_bracket_trend
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.execution.commission import PerShareCommission
from backtester.execution.order_types import BracketSpec, Order, OrderSide, OrderType
from backtester.execution.slippage import FixedBpsSlippage


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class BracketTrend(Backtester):
    """SMA trend entry with a +6% take-profit and -3% stop-loss bracket."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_signal = {}
        self._has_bracket = {}

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

            if pos == 0 and self._has_bracket.get(inst, False):
                self._has_bracket[inst] = False

            if signal == 1 and prev == 0 and pos == 0:
                shares = int(nav * 0.15 / bar.close)
                if shares > 0:
                    bracket = BracketSpec(
                        take_profit_price=round(bar.close * 1.06, 2),
                        stop_loss_price=round(bar.close * 0.97, 2),
                    )
                    self.fill_engine.submit(Order(
                        inst,
                        OrderSide.BUY,
                        shares,
                        OrderType.BRACKET,
                        bracket=bracket,
                    ))
                    self._has_bracket[inst] = True
            elif signal == 0 and prev == 1 and pos > 0:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                self._has_bracket[inst] = False

            self._prev_signal[inst] = signal


async def main() -> None:
    strategy = BracketTrend(
        **_credentials(),
        strategy_name="ExampleOrders12_BracketTrend",
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
