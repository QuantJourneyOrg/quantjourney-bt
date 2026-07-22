# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 14 - Pairs Trading (Rolling Hedge Ratio)
=======================================================

Mode: weights (market-neutral long/short).
Idea: same mean-reversion premise as the ratio pair, but the spread is built
from a rolling OLS hedge ratio (beta of A on B) instead of a 1:1 log ratio, so
the pair stays balanced as the relationship drifts.
Universe: predeclared EWA / EWC country-ETF pair (pedagogical, not data-mined).

Signal: beta = rollingCov(logA, logB) / rollingVar(logB);
spread = logA - beta * logB; z-score over a 60-bar window.
- z > +2  -> short EWA, long EWC
- z < -2  -> long EWA, short EWC
- |z| < 0.5 -> flat
Weights are +/-0.5 per leg (dollar-neutral); the hedge ratio drives the signal.

Note: short borrow/financing is not modeled (research approximation).

Usage:
    ./strategy.sh example_weights_14_pairs_hedge_ratio
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


class PairsHedgeRatio(Backtester):
    """Market-neutral EWA/EWC pair on a rolling hedge-ratio spread."""

    LOOKBACK = 60
    ENTRY = 2.0
    EXIT = 0.5

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        a, b = self.instruments[0], self.instruments[1]
        la, lb = np.log(close[a]), np.log(close[b])
        beta = la.rolling(self.LOOKBACK).cov(lb) / lb.rolling(self.LOOKBACK).var()
        spread = la - beta * lb
        z = (spread - spread.rolling(self.LOOKBACK).mean()) / spread.rolling(self.LOOKBACK).std()

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        ia, ib = signals.columns.get_loc(a), signals.columns.get_loc(b)
        state = 0
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
    strategy = PairsHedgeRatio(
        **_credentials(),
        strategy_name="ExampleWeights14_PairsHedgeRatio",
        strategy_type="Market Neutral",
        initial_capital=100_000,
        instruments=["EWA", "EWC"],
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
