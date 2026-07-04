# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 03 - Limit RSI Dip Buyer
=======================================

Mode: orders.
Order type: LIMIT.
Idea: when RSI is weak, place a passive buy limit below the close.
Universe: three high-volume technology stocks.

Entries are limit buys 1.5% below the signal close. Exits are limit sells 3%
above average entry. Each order expires after three bars.

Usage:
    ./strategy.sh example_orders_03_limit_rsi_dip
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


class LimitRSIDipBuyer(Backtester):
    """Passive RSI dip buyer using limit orders for entry and profit-taking."""

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
            pending = self._has_pending(inst)

            if pos == 0 and not pending and value < 40:
                shares = int(nav * 0.15 / bar.close)
                if shares > 0:
                    self.fill_engine.submit(Order(
                        instrument=inst,
                        side=OrderSide.BUY,
                        quantity=shares,
                        order_type=OrderType.LIMIT,
                        limit_price=round(bar.close * 0.985, 2),
                        expires_after_bars=3,
                    ))
            elif pos > 0 and not pending:
                entry = self.get_average_entry_price(inst) or bar.close
                self.fill_engine.submit(Order(
                    instrument=inst,
                    side=OrderSide.SELL,
                    quantity=pos,
                    order_type=OrderType.LIMIT,
                    limit_price=round(entry * 1.03, 2),
                    expires_after_bars=3,
                ))


async def main() -> None:
    strategy = LimitRSIDipBuyer(
        **_credentials(),
        strategy_name="ExampleOrders03_LimitRSIDip",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA"],
        backtest_period={"start": "2020-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.25,
        indicators_config=[
            {"function": "RSI", "price_cols": ["close"], "params": {"periods": [14]}},
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
