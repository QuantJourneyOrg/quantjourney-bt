# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 10 - Monthly Trend With Circuit Breaker
=======================================================

Mode: weights.
Idea: SMA(50/200) trend on a diversified ETF basket, rebalanced monthly, with a
circuit breaker that flattens the book on a large drawdown and waits out a
cooldown before re-engaging.
Universe: equities, bonds, gold, commodities.
Rebalance: business month-end (BME) + max-drawdown circuit breaker (L4).

This demonstrates the risk-off layer of the rebalance engine: normal months
rebalance on the calendar, but a -15% drawdown flattens exposure and a 10-day
cooldown prevents whipsaw re-entry.

Usage:
    ./strategy.sh example_weights_10_monthly_circuit_breaker
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


class MonthlyCircuitBreaker(Backtester):
    """Monthly ETF trend with a drawdown circuit breaker."""

    def _compute_signals(self) -> pd.DataFrame:
        fast = self.instruments_data.get_feature("SMA_50_close")
        slow = self.instruments_data.get_feature("SMA_200_close")
        valid = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.30)


async def main() -> None:
    strategy = MonthlyCircuitBreaker(
        **_credentials(),
        strategy_name="ExampleWeights10_MonthlyCircuitBreaker",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC"],
        backtest_period={"start": "2010-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.30,
        rebalance_policy=RebalancePolicy(
            frequency="BME",
            max_drawdown_trigger=0.15,        # flatten if drawdown exceeds 15%
            max_drawdown_action="flatten",
            circuit_breaker_cooldown_days=10,  # wait 10 trading days before re-engaging
        ),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}},
        ],
        benchmark_symbol="SPY",
        benchmark_name="S&P 500 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
