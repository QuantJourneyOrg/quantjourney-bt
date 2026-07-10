# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 14 - OCO Dip Or Breakout
=======================================

Mode: orders.
Order type: OCO.
Idea: submit two competing entry orders: buy a dip or buy a breakout.
Universe: three liquid technology stocks.

The limit-buy leg sits 2% below the close. The stop-buy leg sits 2% above the
close. Whichever fills first cancels the other leg.

Usage:
    ./strategy.sh example_orders_14_oco_dip_or_breakout
"""

import asyncio
import os

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


class OCODipOrBreakout(Backtester):
    """OCO entry pair: mean-reversion dip versus momentum breakout."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bar_count = {}
        self._entry_bar = {}
        self._oco_live = {}

    def _has_pending(self, instrument: str) -> bool:
        return any(
            o.instrument == instrument and o.is_active for o in self.fill_engine.pending_orders
        )

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        for inst in self.instruments:
            self._bar_count[inst] = self._bar_count.get(inst, 0) + 1
            bar = bars[inst]
            pos = current_positions.get(inst, 0.0)

            if pos > 0 and self._oco_live.get(inst, False):
                self._oco_live[inst] = False
                self._entry_bar[inst] = self._bar_count[inst]

            if pos == 0:
                self._entry_bar.pop(inst, None)
                if self._oco_live.get(inst, False) and not self._has_pending(inst):
                    self._oco_live[inst] = False

            if pos > 0 and self._bar_count[inst] - self._entry_bar.get(inst, 0) >= 10:
                self.fill_engine.cancel_all(instrument=inst)
                self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                self._entry_bar.pop(inst, None)
                continue

            if pos == 0 and not self._oco_live.get(inst, False):
                if self._bar_count[inst] % 5 != 0:
                    continue

                shares = int(nav * 0.10 / bar.close)
                if shares <= 0:
                    continue

                oco_id = f"{inst}_{date:%Y%m%d}_dip_or_breakout"
                self.fill_engine.submit(
                    Order(
                        inst,
                        OrderSide.BUY,
                        shares,
                        OrderType.OCO,
                        limit_price=round(bar.close * 0.98, 2),
                        oco_pair_id=oco_id,
                        expires_after_bars=5,
                    )
                )
                self.fill_engine.submit(
                    Order(
                        inst,
                        OrderSide.BUY,
                        shares,
                        OrderType.OCO,
                        stop_price=round(bar.close * 1.02, 2),
                        oco_pair_id=oco_id,
                        expires_after_bars=5,
                    )
                )
                self._oco_live[inst] = True


async def main() -> None:
    strategy = OCODipOrBreakout(
        **_credentials(),
        strategy_name="ExampleOrders14_OCODipOrBreakout",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA"],
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
