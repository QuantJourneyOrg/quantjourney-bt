# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 09 - Intraday 1h SMA Trend
==========================================

Mode: weights.
Idea: on hourly bars, hold each name only while its SMA(10) is above its
SMA(30), equal weight across active names. A slower intraday-to-swing trend
template.
Universe: four liquid large-caps.
Granularity: 1h intraday data (yfinance keeps a long hourly history).

Timeframe-grid note: 1h is the slowest cadence in the intraday grid
(1m -> 5m -> 30m -> 1h). With ~7 bars per session it behaves like a fast swing
strategy: lower turnover than the 1m/5m templates, and the SMA(10/30) cross is
a clean hourly trend filter. Use this as the "calm" end of the grid to compare
turnover and cost drag against the faster templates.

Usage:
    ./strategy.sh example_weights_09_intraday_1h_sma_trend
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


def _recent_period(days: int = 60) -> dict[str, str]:
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    return {"start": start.isoformat(), "end": end.isoformat()}


class IntradaySMATrend1h(Backtester):
    """Hourly SMA(10/30) trend/cash strategy."""

    def _compute_signals(self) -> pd.DataFrame:
        fast = self.instruments_data.get_feature("SMA_10_close")
        slow = self.instruments_data.get_feature("SMA_30_close")
        valid = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    strategy = IntradaySMATrend1h(
        **_credentials(),
        strategy_name="ExampleWeights09_IntradaySMATrend1h",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "AMZN"],
        backtest_period=_recent_period(days=60),
        source="yfinance",
        granularity="1h",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency=None, rebalance_on_signal_change=True),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [10, 30]}},
        ],
        benchmark_symbol="QQQ",
        benchmark_name="Nasdaq 100 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
