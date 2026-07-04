# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 02 - Monthly ETF Trend With Drift Band
======================================================

Mode: weights.
Idea: SMA(50/200) trend filter on a diversified ETF universe.
Universe: equities, bonds, gold and commodities.
Rebalance: business month-end, plus a 5% drift trigger between month-ends.

This example shows how a slow strategy can reduce turnover while still forcing
a rebalance when market drift moves actual weights too far from target.

Usage:
    ./strategy.sh example_weights_02_monthly_drift_etf
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


class MonthlyDriftETFTrend(Backtester):
    """ETF trend strategy with month-end rebalance and drift control."""

    def _compute_signals(self) -> pd.DataFrame:
        sma_fast = self.instruments_data.get_feature("SMA_50_close")
        sma_slow = self.instruments_data.get_feature("SMA_200_close")
        valid = sma_fast.notna() & sma_slow.notna()
        return (sma_fast > sma_slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        signals = self.signals
        active = signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.30)


async def main() -> None:
    strategy = MonthlyDriftETFTrend(
        **_credentials(),
        strategy_name="ExampleWeights02_MonthlyDriftETF",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC"],
        backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.30,
        rebalance_policy=RebalancePolicy(frequency="BME", drift_threshold=0.05),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}},
        ],
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
