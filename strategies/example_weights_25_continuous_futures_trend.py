# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 25 - Continuous Futures Trend Proxy
===================================================

Mode: weights (price-return research proxy).
Idea: diversified long/short trend following over equity index, rates, energy,
metals, and grains, with inverse-volatility weights.

Yahoo symbols ending in ``=F`` are provider-managed continuous series. This
example does not select dated contracts, control the roll rule, book roll
slippage, apply exchange margin, or calculate multiplier-based PnL. Use it for
signal research, not for an executable futures performance claim.

Usage:
    ./strategy.sh example_weights_25_continuous_futures_trend
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


def build_continuous_futures_trend(
    close: pd.DataFrame,
    *,
    fast_window: int = 63,
    slow_window: int = 252,
    volatility_window: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return moving-average trend signals and unit-gross inverse-vol weights."""
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    valid = fast.notna() & slow.notna()
    signals = np.sign(fast - slow).where(valid, 0.0).astype(float)
    volatility = close.pct_change(fill_method=None).rolling(
        volatility_window, min_periods=volatility_window
    ).std() * np.sqrt(252.0)
    raw = signals / volatility.replace(0.0, np.nan)
    weights = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return signals, weights


class ContinuousFuturesTrend(Backtester):
    """Diversified trend signals on provider-managed continuous futures."""

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        return build_continuous_futures_trend(close)[0]

    def _compute_weights(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        return build_continuous_futures_trend(close)[1]


async def main() -> None:
    strategy = ContinuousFuturesTrend(
        **_credentials(),
        strategy_name="ExampleWeights25_ContinuousFuturesTrend",
        strategy_type="Long / Short Futures Proxy",
        initial_capital=1_000_000,
        instruments=["MES=F", "MNQ=F", "ZN=F", "CL=F", "GC=F", "ZC=F"],
        backtest_period={"start": "2010-01-01", "end": "2026-01-01"},
        source="yfinance",
        execution_mode="weights",
        rebalance_policy=RebalancePolicy(frequency="BME"),
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
