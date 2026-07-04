# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 15 - Cross-Sectional Momentum (Long/Short)
==========================================================

Mode: weights (dollar-neutral long/short).
Idea: each month, rank the universe by 12-month price momentum, go long the top
names and short the bottom names in equal dollar amounts. The classic
cross-sectional momentum factor.
Universe: twelve liquid large-caps across sectors.

Signal: 12-month (252-bar) return per name; long top 3, short bottom 3.
Weights: +0.5 spread across longs, -0.5 across shorts (gross 1.0, net ~0).

Note: short borrow/financing is not modeled (research approximation).

Usage:
    ./strategy.sh example_weights_15_cross_sectional_momentum
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


class CrossSectionalMomentum(Backtester):
    """Dollar-neutral 12-month cross-sectional momentum."""

    LOOKBACK = 252
    N_SIDE = 3

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for i in range(self.LOOKBACK, len(close)):
            momentum = (close.iloc[i] / close.iloc[i - self.LOOKBACK] - 1.0).dropna()
            if len(momentum) < 2 * self.N_SIDE:
                continue
            longs = momentum.nlargest(self.N_SIDE).index
            shorts = momentum.nsmallest(self.N_SIDE).index
            signals.iloc[i, signals.columns.isin(longs)] = 1.0
            signals.iloc[i, signals.columns.isin(shorts)] = -1.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        longs = (self.signals == 1.0).astype(float)
        shorts = (self.signals == -1.0).astype(float)
        long_w = longs.div(longs.sum(axis=1), axis=0).fillna(0.0) * 0.5
        short_w = shorts.div(shorts.sum(axis=1), axis=0).fillna(0.0) * 0.5
        return long_w - short_w


async def main() -> None:
    strategy = CrossSectionalMomentum(
        **_credentials(),
        strategy_name="ExampleWeights15_CrossSectionalMomentum",
        strategy_type="Market Neutral",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
                     "JPM", "XOM", "JNJ", "PG", "WMT", "KO"],
        backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=1.0,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        indicators_config=[],
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
