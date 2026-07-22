# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 13 - Pairs Trading (Ratio Z-Score)
==================================================

Mode: weights (market-neutral long/short).
Idea: trade the mean-reverting spread between two closely related names using a
log-ratio z-score. When the spread stretches, short the rich leg and long the
cheap leg; unwind when it reverts.
Universe: predeclared KO / PEP consumer-staples pair (pedagogical, not data-mined).

Signal: spread = log(KO) - log(PEP); z-score over a 60-bar window.
- z > +2  -> short KO, long PEP  (KO rich relative to PEP)
- z < -2  -> long KO, short PEP
- |z| < 0.5 -> flat
Weights are +/-0.5 per leg (gross 1.0, net ~0 = dollar-neutral).

Note: short borrow/financing is not modeled (research approximation), so the
long/short carry is excluded from returns. See the repository README.

Usage:
    ./strategy.sh example_weights_13_pairs_ratio_zscore
"""

import asyncio
import os

import numpy as np
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


class PairsRatioZScore(Backtester):
    """Market-neutral KO/PEP pair on a log-ratio z-score."""

    LOOKBACK = 60
    ENTRY = 2.0
    EXIT = 0.5

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        a, b = self.instruments[0], self.instruments[1]
        spread = np.log(close[a]) - np.log(close[b])
        z = (spread - spread.rolling(self.LOOKBACK).mean()) / spread.rolling(self.LOOKBACK).std()

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        ia, ib = signals.columns.get_loc(a), signals.columns.get_loc(b)
        state = 0  # +1 = long A / short B, -1 = short A / long B
        for i in range(len(z)):
            zi = z.iloc[i]
            if pd.isna(zi):
                state = 0
            elif state == 0:
                if zi > self.ENTRY:
                    state = -1
                elif zi < -self.ENTRY:
                    state = 1
            elif abs(zi) < self.EXIT:
                state = 0
            signals.iloc[i, ia] = float(state)
            signals.iloc[i, ib] = float(-state)
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        return self.signals * 0.5


async def main() -> None:
    strategy = PairsRatioZScore(
        **_credentials(),
        strategy_name="ExampleWeights13_PairsRatioZScore",
        strategy_type="Market Neutral",
        initial_capital=100_000,
        instruments=["KO", "PEP"],
        backtest_period={"start": "2000-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=1.0,
        rebalance_policy=RebalancePolicy(frequency="D"),
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
