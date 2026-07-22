# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 11 - Quarterly Momentum With Tracking-Error & Cost Gate
=======================================================================

Mode: weights.
Idea: quarterly momentum rotation on a broad ETF universe, but layered with a
tracking-error trigger (rebalance early if the book drifts too far from the
benchmark) and an annual-turnover budget (a cost gate that suppresses trading
once the rolling turnover budget is spent).
Universe: canonical multi-asset ETFs: SPY, EFA, EEM, TLT, IEF, GLD, DBC and VNQ.

This demonstrates the two rebalance layers most useful to institutional
mandates: staying within a tracking-error band vs a benchmark, and respecting a
turnover budget so the strategy does not trade itself to death.

Usage:
    ./strategy.sh example_weights_11_quarterly_te_cost_gate
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


class QuarterlyTECostGate(Backtester):
    """Quarterly momentum with TE trigger and turnover budget."""

    LOOKBACK = 126  # ~6 months of trading days
    TOP_N = 3

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for i in range(self.LOOKBACK, len(close)):
            momentum = close.iloc[i] / close.iloc[i - self.LOOKBACK] - 1.0
            picks = momentum.dropna().nlargest(self.TOP_N).index
            signals.iloc[i, signals.columns.isin(picks)] = 1.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.40)


async def main() -> None:
    strategy = QuarterlyTECostGate(
        **_credentials(),
        strategy_name="ExampleWeights11_QuarterlyTECostGate",
        strategy_type="Long Only",
        initial_capital=100_000,
        instruments=["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
        backtest_period={"start": "2007-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.40,
        rebalance_policy=RebalancePolicy(
            frequency="BQE",
            tracking_error_threshold=0.06,  # rebalance early if annualized TE vs benchmark > 6%
            tracking_error_window=63,
            max_annual_turnover=4.0,  # cost gate: cap rolling 252d turnover at 4x
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
