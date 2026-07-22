# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 04 - Quarterly Dual Momentum
============================================

Mode: weights.
Idea: rank ETFs by 12-month return, hold the top two only if return is positive.
Universe: canonical multi-asset ETFs: SPY, EFA, EEM, TLT, IEF, GLD, DBC and VNQ.

Dual momentum combines relative momentum (which asset is strongest) and
absolute momentum (do not own it if its own trend is negative).

Usage:
    ./strategy.sh example_weights_04_quarterly_dual_momentum
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


class QuarterlyDualMomentum(Backtester):
    """Quarterly relative-plus-absolute momentum ETF rotation."""

    lookback = 252
    top_n = 2

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for i in range(self.lookback, len(close)):
            momentum = close.iloc[i] / close.iloc[i - self.lookback] - 1.0
            eligible = momentum.dropna()
            picks = eligible[eligible > 0.0].nlargest(self.top_n).index
            signals.iloc[i, signals.columns.isin(picks)] = 1.0

        return signals

    def _compute_weights(self) -> pd.DataFrame:
        signals = self.signals
        active = signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.50)


async def main() -> None:
    strategy = QuarterlyDualMomentum(
        **_credentials(),
        strategy_name="ExampleWeights04_QuarterlyDualMomentum",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
        backtest_period={"start": "2007-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.50,
        rebalance_policy=RebalancePolicy(frequency="BQE"),
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
