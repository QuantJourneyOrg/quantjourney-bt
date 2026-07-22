# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 18 - Volatility-Targeted Momentum Basket
========================================================

Mode: weights + risk overlay.
Idea: hold a momentum-selected basket (top names by 6-month return) and scale
it to a 15% annualized volatility target, allowing modest leverage. A different
base strategy and a higher target than the trend example, to show how the same
overlay adapts.
Universe: canonical multi-asset ETFs: SPY, EFA, EEM, TLT, IEF, GLD, DBC and VNQ.

Risk overlay: `VolTargetModel(target_vol=0.15, lookback=42, max_leverage=2.0)`.
The base picks the strongest assets; the overlay decides how much to hold.

Usage:
    ./strategy.sh example_weights_18_vol_target_momentum
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.risk import VolTargetModel


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class VolTargetMomentum(Backtester):
    """Momentum basket scaled to a volatility target."""

    LOOKBACK = 126
    TOP_N = 3

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for i in range(self.LOOKBACK, len(close)):
            momentum = (close.iloc[i] / close.iloc[i - self.LOOKBACK] - 1.0).dropna()
            picks = momentum[momentum > 0.0].nlargest(self.TOP_N).index
            signals.iloc[i, signals.columns.isin(picks)] = 1.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.40)


async def main() -> None:
    strategy = VolTargetMomentum(
        **_credentials(),
        strategy_name="ExampleWeights18_VolTargetMomentum",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
        backtest_period={"start": "2007-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.40,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        risk_model=VolTargetModel(target_vol=0.15, lookback=42, max_leverage=2.0),
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
