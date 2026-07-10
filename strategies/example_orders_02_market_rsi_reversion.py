# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 02 - Market RSI Mean Reversion
=============================================

Mode: orders.
Order type: MARKET.
Idea: buy oversold stocks when RSI(14) is below 35, sell when RSI is above 60.
Universe: five liquid mega-cap stocks.

This example uses market orders for both entry and exit, so it is easy to read
and useful as a minimal order-mode mean-reversion template.

Usage:
    ./strategy.sh example_orders_02_market_rsi_reversion
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


class MarketRSIReversion(Backtester):
    """RSI entry/exit strategy implemented with market orders."""

    def _has_pending(self, instrument: str) -> bool:
        return any(
            o.instrument == instrument and o.is_active for o in self.fill_engine.pending_orders
        )

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
            if pos == 0 and value < 35 and not self._has_pending(inst):
                shares = int(nav * 0.12 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(inst, OrderSide.BUY, shares, OrderType.MARKET))
            elif pos > 0 and value > 60:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))


async def main() -> None:
    strategy = MarketRSIReversion(
        **_credentials(),
        strategy_name="ExampleOrders02_MarketRSIReversion",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "META", "AMZN"],
        backtest_period={"start": "2020-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.20,
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
