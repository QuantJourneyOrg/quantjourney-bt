# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 17 - Volatility-Targeted Trend
==============================================

Mode: weights + risk overlay.
Idea: a simple SMA(50/200) trend basket, but instead of holding fixed weights,
scale total exposure so the portfolio targets ~10% annualized volatility. In
calm markets the strategy leans in; in turbulent markets it de-risks.
Universe: canonical multi-asset ETFs: SPY, EFA, EEM, TLT, IEF, GLD, DBC and VNQ.

Risk overlay: `VolTargetModel(target_vol=0.10, lookback=63, max_leverage=1.5)`
is attached via `risk_model=`; the engine applies it between weights and
rebalance. Volatility is estimated on a strictly prior window (no look-ahead).

Usage:
    ./strategy.sh example_weights_17_vol_target_trend
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


class VolTargetTrend(Backtester):
    """SMA(50/200) trend basket scaled to a volatility target."""

    def _compute_signals(self) -> pd.DataFrame:
        fast = self.instruments_data.get_feature("SMA_50_close")
        slow = self.instruments_data.get_feature("SMA_200_close")
        valid = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.34)


async def main() -> None:
    strategy = VolTargetTrend(
        **_credentials(),
        strategy_name="ExampleWeights17_VolTargetTrend",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
        backtest_period={"start": "2007-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.34,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        risk_model=VolTargetModel(target_vol=0.10, lookback=63, max_leverage=1.5),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}},
        ],
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
