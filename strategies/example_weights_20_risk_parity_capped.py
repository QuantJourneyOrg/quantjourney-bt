# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 20 - Risk Parity With Position Cap (Chained Overlays)
====================================================================

Mode: weights + chained risk overlays.
Idea: equal-risk-contribution across US sector ETFs, then a hard per-position
cap so no single sector dominates. This shows how to CHAIN risk models: risk
parity first, position limit second.
Universe: US sector ETFs (tech, financials, energy, health care, staples,
industrials, utilities).

Risk overlay: `RiskModelChain([RiskParityModel(lookback=126),
PositionLimitModel(max_weight=0.25)])` — ERC weights are computed, then capped
at 25% each and renormalized. A longer 126-bar covariance window makes the ERC
estimate steadier than the multi-asset example.

Usage:
    ./strategy.sh example_weights_20_risk_parity_capped
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.risk import RiskParityModel, PositionLimitModel, RiskModelChain


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class RiskParityCapped(Backtester):
    """Sector ERC portfolio with a per-position cap."""

    WARMUP = 126

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
    strategy = RiskParityCapped(
        **_credentials(),
        strategy_name="ExampleWeights20_RiskParityCapped",
        strategy_type="Long Only",
        initial_capital=100_000,
        instruments=["XLK", "XLF", "XLE", "XLV", "XLP", "XLI", "XLU"],
        backtest_period={"start": "2012-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=1.0,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        risk_model=RiskModelChain([
            RiskParityModel(lookback=126),
            PositionLimitModel(max_weight=0.25),
        ]),
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
