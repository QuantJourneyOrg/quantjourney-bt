# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 16 - Intraday 30m Stop Breakout
==============================================

Mode: orders.
Order type: STOP.
Idea: on 30-minute bars, place buy-stop orders above the recent 12-bar high so
entries only trigger on confirmed intraday breakouts. Positions are held for a
fixed number of bars and then exited at market.
Universe: three predeclared liquid ETFs: SPY, QQQ and IWM.

Timeframe-grid note: at 30m the session has ~13 bars, so a 12-bar high is
roughly a prior-session breakout. The buy-stop waits for price confirmation and
demonstrates realistic gap handling (a gap through the stop fills at the worse
open, not the stop price).

Usage:
    ./strategy.sh example_orders_16_intraday_30m_stop_breakout
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta

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


def _recent_period(days: int = 55) -> dict[str, str]:
    # yfinance serves at most ~60 calendar days of 30-minute bars.
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    return {"start": start.isoformat(), "end": end.isoformat()}


class IntradayStopBreakout30m(Backtester):
    """30-minute breakout via buy-stop, fixed holding period."""

    LOOKBACK = 12
    HOLD_BARS = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bar_count = {}
        self._entry_bar = {}

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        if pd.isna(nav) or nav <= 0:
            return

        high = self.instruments_data.get_feature("high")
        if date not in high.index:
            return
        pos_idx = high.index.get_loc(date)
        if pos_idx < self.LOOKBACK:
            return

        for inst in self.instruments:
            self._bar_count[inst] = self._bar_count.get(inst, 0) + 1
            bar = bars[inst]
            close = float(bar.close) if not pd.isna(bar.close) else float("nan")
            if pd.isna(close) or close <= 0:
                continue

            pos = current_positions.get(inst, 0.0)

            # Time-based exit.
            if pos > 0:
                held = self._bar_count[inst] - self._entry_bar.get(inst, 0)
                if held >= self.HOLD_BARS:
                    self.fill_engine.cancel_all(instrument=inst)
                    self.fill_engine.submit(Order(inst, OrderSide.SELL, pos, OrderType.MARKET))
                    self._entry_bar.pop(inst, None)
                continue

            prior_high = high[inst].iloc[pos_idx - self.LOOKBACK : pos_idx].max()
            if pd.isna(prior_high) or prior_high <= 0:
                continue

            shares = int(nav * 0.20 / close)
            if shares > 0:
                self.fill_engine.submit(
                    Order(
                        inst,
                        OrderSide.BUY,
                        shares,
                        OrderType.STOP,
                        stop_price=round(prior_high * 1.001, 2),
                        expires_after_bars=3,
                    )
                )
                self._entry_bar[inst] = self._bar_count[inst]


async def main() -> None:
    strategy = IntradayStopBreakout30m(
        **_credentials(),
        strategy_name="ExampleOrders16_IntradayStopBreakout30m",
        strategy_type="Intraday Breakout",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM"],
        backtest_period={"start": "2026-05-16", "end": "2026-07-11"},
        granularity="30m",
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.25,
        indicators_config=[],
        slippage_model=FixedBpsSlippage(bps=2.0),
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
