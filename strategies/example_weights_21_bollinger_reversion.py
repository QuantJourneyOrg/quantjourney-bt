# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 21 - Bollinger Band Mean Reversion
==================================================

Mode: weights.
Idea: buy a name when its price closes below the lower Bollinger Band (a
stretched-cheap signal) and hold until it reverts back above the moving-average
midline. A classic band-based mean-reversion template.
Universe: five liquid mega-cap stocks.

Signal: mid = SMA(20), band = mid +/- 2 * rolling std(20).
- close < lower band  -> enter long
- close > mid          -> exit
Bands are computed inline from adjusted close.

Usage:
    ./strategy.sh example_weights_21_bollinger_reversion
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


class BollingerReversion(Backtester):
    """Long when price closes below the lower band, exit at the midline."""

    WINDOW = 20
    N_STD = 2.0

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        mid = close.rolling(self.WINDOW).mean()
        std = close.rolling(self.WINDOW).std()
        lower = mid - self.N_STD * std

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for inst in close.columns:
            holding = False
            for t in range(len(close)):
                c, lo, mi = close[inst].iloc[t], lower[inst].iloc[t], mid[inst].iloc[t]
                if pd.isna(lo) or pd.isna(mi):
                    holding = False
                elif not holding and c < lo:
                    holding = True
                elif holding and c > mi:
                    holding = False
                signals[inst].iloc[t] = 1.0 if holding else 0.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    strategy = BollingerReversion(
        **_credentials(),
        strategy_name="ExampleWeights21_BollingerReversion",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency="D"),
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
