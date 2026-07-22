# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 23 - FX Time-Series Momentum
============================================

Mode: weights (price-return research proxy).
Idea: trade each USD-quoted spot-FX pair in the direction of its six-month
momentum and scale active signals by inverse 63-day volatility.
Universe: EURUSD, GBPUSD, AUDUSD and NZDUSD provider spot-FX proxies.

This example tests FX data and portfolio logic. Weights mode does not model
standard lots, FX margin, rollover/swap points, bid/ask spread, or conversion
of non-USD PnL. Use the order example for contract-aware lot sizing.

Usage:
    ./strategy.sh example_weights_23_fx_time_series_momentum
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


def build_fx_time_series_momentum(
    close: pd.DataFrame,
    *,
    momentum_lookback: int = 126,
    volatility_lookback: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return direction signals and unit-gross inverse-volatility weights."""
    momentum = close.pct_change(momentum_lookback, fill_method=None)
    signals = np.sign(momentum).where(momentum.notna(), 0.0).astype(float)
    volatility = close.pct_change(fill_method=None).rolling(
        volatility_lookback, min_periods=volatility_lookback
    ).std() * np.sqrt(252.0)
    inverse_volatility = 1.0 / volatility.replace(0.0, np.nan)
    raw = signals * inverse_volatility
    weights = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return signals, weights


class FXTimeSeriesMomentum(Backtester):
    """Six-month FX trend with inverse-volatility sizing."""

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        return build_fx_time_series_momentum(close)[0]

    def _compute_weights(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        return build_fx_time_series_momentum(close)[1]


async def main() -> None:
    strategy = FXTimeSeriesMomentum(
        **_credentials(),
        strategy_name="ExampleWeights23_FXTimeSeriesMomentum",
        strategy_type="Long / Short FX Proxy",
        initial_capital=500_000,
        instruments=["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"],
        backtest_period={"start": "2010-01-04", "end": "2026-01-01"},
        benchmark_symbol="DX-Y.NYB",
        benchmark_name="US Dollar Index",
        source="yfinance",
        execution_mode="weights",
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
