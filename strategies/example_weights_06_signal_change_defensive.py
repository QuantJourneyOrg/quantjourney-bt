# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 06 - Signal-Change Defensive Rotation
=====================================================

Mode: weights.
Idea: if SPY is above its SMA(200), hold risk ETFs; otherwise hold defensive ETFs.
Universe: SPY, QQQ, IWM for risk-on; TLT and GLD for risk-off.
Rebalance: only when target weights change materially.

This demonstrates signal-driven rebalancing: the calendar is disabled, so the
engine trades only when the regime changes.

Usage:
    ./strategy.sh example_weights_06_signal_change_defensive
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


class SignalChangeDefensiveRotation(Backtester):
    """Risk-on/risk-off ETF rotation with signal-change rebalance trigger."""

    risk_on_assets = ["SPY", "QQQ", "IWM"]
    defensive_assets = ["TLT", "GLD"]

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        sma_200 = self.instruments_data.get_feature("SMA_200_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        risk_on = close["SPY"] > sma_200["SPY"]
        for date in close.index:
            if pd.isna(sma_200.loc[date, "SPY"]):
                continue
            assets = self.risk_on_assets if bool(risk_on.loc[date]) else self.defensive_assets
            signals.loc[date, assets] = 1.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        signals = self.signals
        active = signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.50)


async def main() -> None:
    strategy = SignalChangeDefensiveRotation(
        **_credentials(),
        strategy_name="ExampleWeights06_SignalChangeDefensive",
        strategy_type="Risk On / Risk Off",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM", "TLT", "GLD"],
        backtest_period={"start": "2010-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.50,
        rebalance_policy=RebalancePolicy(
            frequency=None,
            rebalance_on_signal_change=True,
            signal_change_threshold=0.01,
        ),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [200]}},
        ],
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
