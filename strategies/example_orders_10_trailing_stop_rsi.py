# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 10 - RSI Entry With Trailing Stop
================================================

Mode: orders.
Order types: MARKET entry, STOP_TRAIL exit.
Idea: buy oversold RSI readings, then let a 5% trailing stop handle risk.
Universe: three large technology stocks.

This combines mean-reversion entries with trend-style exit management: if the
rebound continues, the trailing stop keeps moving upward.

Usage:
    ./strategy.sh example_orders_10_trailing_stop_rsi
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


class TrailingStopRSI(Backtester):
    """RSI dip entry protected by a percent trailing stop."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_trail = {}

    def _has_pending(self, instrument: str) -> bool:
        return any(o.instrument == instrument and o.is_active for o in self.fill_engine.pending_orders)

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        rsi = self.instruments_data.get_feature("RSI_14_close")
        if date not in rsi.index:
            return

        for inst in self.instruments:
            value = rsi.loc[date, inst]
            if pd.isna(value):
                continue

            bar = bars[inst]
            pos = current_positions.get(inst, 0.0)

            if pos == 0 and self._has_trail.get(inst, False):
                self._has_trail[inst] = False

            if pos == 0 and value < 38 and not self._has_pending(inst):
                shares = int(nav * 0.15 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(inst, OrderSide.BUY, shares, OrderType.MARKET))
            elif pos > 0 and not self._has_trail.get(inst, False):
                self.fill_engine.submit(Order(
                    inst,
                    OrderSide.SELL,
                    pos,
                    OrderType.STOP_TRAIL,
                    trail_percent=0.05,
                ))
                self._has_trail[inst] = True
            elif pos > 0 and value > 65:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                self._has_trail[inst] = False


async def main() -> None:
    strategy = TrailingStopRSI(
        **_credentials(),
        strategy_name="ExampleOrders10_TrailingStopRSI",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA"],
        backtest_period={"start": "2020-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.25,
        indicators_config=[
            {"function": "RSI", "price_cols": ["close"], "params": {"periods": [14]}},
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
