# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 05 - Monthly Inverse Volatility Basket
======================================================

Mode: weights.
Idea: allocate more to ETFs with lower recent volatility.
Universe: canonical multi-asset ETFs: SPY, EFA, EEM, TLT, IEF, GLD, DBC and VNQ.

The strategy stays long all assets after the warmup period, but sizes each ETF
by inverse 63-day volatility. This is a simple risk-budgeting example.

Usage:
    ./strategy.sh example_weights_05_monthly_inverse_vol
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


class MonthlyInverseVol(Backtester):
    """Long-only monthly ETF basket weighted by inverse realized volatility."""

    lookback = 63

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signals.iloc[self.lookback :] = 1.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        realized_vol = close.pct_change().rolling(self.lookback).std() * np.sqrt(252)
        inverse_vol = 1.0 / realized_vol.replace(0.0, np.nan)
        weights = inverse_vol.div(inverse_vol.sum(axis=1), axis=0).fillna(0.0)
        return weights.clip(upper=0.35)


async def main() -> None:
    strategy = MonthlyInverseVol(
        **_credentials(),
        strategy_name="ExampleWeights05_MonthlyInverseVol",
        strategy_type="Long Only",
        initial_capital=100_000,
        instruments=["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
        backtest_period={"start": "2007-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.35,
        rebalance_policy=RebalancePolicy(frequency="BME"),
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
