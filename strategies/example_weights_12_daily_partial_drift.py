# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 12 - Daily Momentum With Partial Drift Rebalance
===============================================================

Mode: weights.
Idea: a daily-updated momentum tilt across large-caps, but instead of fully
rebalancing every day, only trade the positions that have drifted outside a
10% band — a partial rebalance that keeps turnover low.
Universe: canonical US sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY.

This demonstrates partial rebalancing: the difference between snapping the whole
book to target every day (high turnover) and only correcting the positions that
actually breached the band (low turnover, same intent).

Usage:
    ./strategy.sh example_weights_12_daily_partial_drift
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


class DailyPartialDrift(Backtester):
    """Daily momentum weights with partial drift-band rebalancing."""

    LOOKBACK = 63

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        momentum = close / close.shift(self.LOOKBACK) - 1.0
        return (momentum > 0.0).astype(float).where(momentum.notna(), 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        momentum = (close / close.shift(self.LOOKBACK) - 1.0).clip(lower=0.0)
        momentum = momentum.where(self.signals == 1.0, 0.0)
        totals = momentum.sum(axis=1)
        return momentum.div(totals, axis=0).fillna(0.0).clip(upper=0.35)


async def main() -> None:
    strategy = DailyPartialDrift(
        **_credentials(),
        strategy_name="ExampleWeights12_DailyPartialDrift",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
        backtest_period={"start": "2000-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.35,
        rebalance_policy=RebalancePolicy(
            frequency="D",
            drift_threshold=0.10,  # only act when a weight drifts >10% from target
            drift_type="absolute",
            partial_rebalance=True,  # trade only the drifted names
        ),
        indicators_config=[],
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
