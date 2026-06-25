# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 01 - Daily SMA Trend
====================================

Mode: weights.
Idea: hold each stock only when SMA(50) is above SMA(200).
Universe: five large US technology stocks.
Rebalance: daily calendar rebalance.

This is the simplest weight-based template: compute binary signals, convert
active names to equal weights, and let the engine translate weights to trades.

Usage:
    ./strategy.sh example_weights_01_sma_daily
"""

import asyncio
import os

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


class DailySMATrend(Backtester):
    """Long/cash SMA(50/200) trend strategy with daily rebalancing."""

    def _compute_signals(self) -> pd.DataFrame:
        sma_fast = self.instruments_data.get_feature("SMA_50_close")
        sma_slow = self.instruments_data.get_feature("SMA_200_close")
        valid = sma_fast.notna() & sma_slow.notna()
        return (sma_fast > sma_slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        signals = self.signals
        active = signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    strategy = DailySMATrend(
        **_credentials(),
        strategy_name="ExampleWeights01_DailySMATrend",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency="D"),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}},
        ],
        benchmark_symbol="^GSPC",
        benchmark_name="S&P 500 Index",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
