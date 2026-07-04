# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 16 - Cross-Sectional Short-Term Reversal (Long/Short)
====================================================================

Mode: weights (dollar-neutral long/short).
Idea: the mirror image of momentum. Over short horizons, recent losers tend to
bounce and recent winners tend to give back. Each week, rank the universe by
1-month return, LONG the biggest losers and SHORT the biggest winners.
Universe: twelve liquid large-caps across sectors.

Signal: 21-bar (~1-month) return per name; long bottom 3, short top 3.
Weights: +0.5 spread across longs, -0.5 across shorts (gross 1.0, net ~0).

Note: short borrow/financing is not modeled (research approximation). Short-term
reversal is turnover-heavy; costs matter — enable slippage/commissions before
drawing conclusions.

Usage:
    ./strategy.sh example_weights_16_cross_sectional_reversal
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


class CrossSectionalReversal(Backtester):
    """Dollar-neutral 1-month cross-sectional reversal."""

    LOOKBACK = 21
    N_SIDE = 3

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for i in range(self.LOOKBACK, len(close)):
            ret = (close.iloc[i] / close.iloc[i - self.LOOKBACK] - 1.0).dropna()
            if len(ret) < 2 * self.N_SIDE:
                continue
            longs = ret.nsmallest(self.N_SIDE).index    # buy the losers
            shorts = ret.nlargest(self.N_SIDE).index     # short the winners
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
    strategy = CrossSectionalReversal(
        **_credentials(),
        strategy_name="ExampleWeights16_CrossSectionalReversal",
        strategy_type="Market Neutral",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
                     "JPM", "XOM", "JNJ", "PG", "WMT", "KO"],
        backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=1.0,
        rebalance_policy=RebalancePolicy(frequency="W", weekday=4),
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
