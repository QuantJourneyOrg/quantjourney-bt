# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 13 - Bracket RSI Reversion
=========================================

Mode: orders.
Order type: BRACKET.
Idea: buy oversold RSI dips with a predefined reward/risk bracket.
Universe: three liquid technology stocks.

Each RSI entry uses a bracket with +4% take-profit and -2% stop-loss. This is
a compact way to express a complete trade lifecycle in one order.

Usage:
    ./strategy.sh example_orders_13_bracket_rsi_reversion
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.execution.commission import FixedBpsCommission
from backtester.execution.order_types import BracketSpec, Order, OrderSide, OrderType
from backtester.execution.slippage import FixedBpsSlippage


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class BracketRSIReversion(Backtester):
    """RSI dip-buying with bracket exits."""

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
                shares = int(nav * 0.15 / bar.close)
                if shares > 0:
                    bracket = BracketSpec(
                        take_profit_price=round(bar.close * 1.04, 2),
                        stop_loss_price=round(bar.close * 0.98, 2),
                    )
                    self.fill_engine.submit(
                        Order(
                            inst,
                            OrderSide.BUY,
                            shares,
                            OrderType.BRACKET,
                            bracket=bracket,
                        )
                    )
            elif pos > 0 and value > 70:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))


async def main() -> None:
    strategy = BracketRSIReversion(
        **_credentials(),
        strategy_name="ExampleOrders13_BracketRSIReversion",
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
