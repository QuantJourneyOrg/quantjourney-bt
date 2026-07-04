# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 19 - Risk Parity (Multi-Asset ERC)
==================================================

Mode: weights + risk overlay.
Idea: hold a diversified multi-asset basket, but size positions so each asset
contributes equal risk (equal risk contribution, ERC) rather than equal dollars.
Low-volatility assets (bonds) get more capital; high-volatility assets (equities,
commodities) get less.
Universe: equities, bonds, gold, commodities.

Base: stay invested in all assets after warm-up (equal target weights).
Risk overlay: `RiskParityModel(lookback=63)` replaces the equal weights with ERC
weights estimated on a strictly prior covariance window (no look-ahead).

Usage:
    ./strategy.sh example_weights_19_risk_parity_multiasset
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.risk import RiskParityModel


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class RiskParityMultiAsset(Backtester):
    """Equal-risk-contribution multi-asset portfolio."""

    WARMUP = 63

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signals.iloc[self.WARMUP:] = 1.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0)


async def main() -> None:
    strategy = RiskParityMultiAsset(
        **_credentials(),
        strategy_name="ExampleWeights19_RiskParityMultiAsset",
        strategy_type="Long Only",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC"],
        backtest_period={"start": "2012-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=1.0,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        risk_model=RiskParityModel(lookback=63),
        indicators_config=[],
        benchmark_symbol="SPY",
        benchmark_name="S&P 500 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
