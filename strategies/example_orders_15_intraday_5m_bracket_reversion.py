# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 15 - Intraday 5m Bracket Reversion
=================================================

Mode: orders.
Order type: BRACKET.
Idea: on 5-minute bars, buy oversold RSI(14) dips and wrap each entry in a
tight intraday bracket (+0.6% take-profit / -0.4% stop-loss).
Universe: three predeclared liquid ETFs: SPY, QQQ and IWM.

Timeframe-grid note: the 5m cadence is the sweet spot for intraday
mean-reversion — long enough for RSI to be meaningful, short enough for several
round-trips per session. The bracket makes the reward/risk explicit per trade
and shows realistic intraday exit handling (TP and SL as a linked OCO pair).

Usage:
    ./strategy.sh example_orders_15_intraday_5m_bracket_reversion
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta

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


def _recent_period(days: int = 30) -> dict[str, str]:
    # yfinance serves at most ~60 calendar days of 5-minute bars.
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    return {"start": start.isoformat(), "end": end.isoformat()}


class IntradayBracketReversion5m(Backtester):
    """5-minute oversold-RSI entries with a tight bracket exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_bracket = {}

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

            if pos == 0 and self._has_bracket.get(inst, False):
                self._has_bracket[inst] = False

            if value < 30 and pos == 0 and not self._has_bracket.get(inst, False):
                shares = int(nav * 0.20 / bar.close)
                if shares > 0:
                    bracket = BracketSpec(
                        take_profit_price=round(bar.close * 1.006, 2),
                        stop_loss_price=round(bar.close * 0.996, 2),
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
                    self._has_bracket[inst] = True


async def main() -> None:
    strategy = IntradayBracketReversion5m(
        **_credentials(),
        strategy_name="ExampleOrders15_IntradayBracketReversion5m",
        strategy_type="Intraday Mean Reversion",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM"],
        backtest_period={"start": "2026-06-10", "end": "2026-07-11"},
        granularity="5m",
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="orders",
        max_position_size=0.25,
        indicators_config=[
            {"function": "RSI", "price_cols": ["close"], "params": {"periods": [14]}},
        ],
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
