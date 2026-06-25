# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 04 - Limit Trend Pullback
========================================

Mode: orders.
Order type: LIMIT.
Idea: in an uptrend, wait for a 1% pullback before entering.
Universe: five large-cap stocks.

If close is above SMA(50), the strategy places a limit buy below the close.
Once long, it places a 4% limit take-profit and exits at market if trend fails.

Usage:
    ./strategy.sh example_orders_04_limit_trend_pullback
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


class LimitTrendPullback(Backtester):
    """Trend-following entry that waits for a passive pullback fill."""

    def _has_pending(self, instrument: str) -> bool:
        return any(o.instrument == instrument and o.is_active for o in self.fill_engine.pending_orders)

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        sma = self.instruments_data.get_feature("SMA_50_close")
        if date not in sma.index:
            return

        for inst in self.instruments:
            trend = sma.loc[date, inst]
            if pd.isna(trend):
                continue

            bar = bars[inst]
            pos = current_positions.get(inst, 0.0)
            pending = self._has_pending(inst)

            if pos == 0 and not pending and bar.close > trend:
                shares = int(nav * 0.12 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(
                        inst,
                        OrderSide.BUY,
                        shares,
                        OrderType.LIMIT,
                        limit_price=round(bar.close * 0.99, 2),
                        expires_after_bars=2,
                    ))
            elif pos > 0 and bar.close < trend:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
            elif pos > 0 and not pending:
                entry = self.get_average_entry_price(inst) or bar.close
                self.fill_engine.submit(Order(
                    inst,
                    OrderSide.SELL,
                    pos,
                    OrderType.LIMIT,
                    limit_price=round(entry * 1.04, 2),
                    expires_after_bars=5,
                ))


async def main() -> None:
    strategy = LimitTrendPullback(
        **_credentials(),
        strategy_name="ExampleOrders04_LimitTrendPullback",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period={"start": "2020-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.20,
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50]}},
        ],
        slippage_model=FixedBpsSlippage(bps=3.0),
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
