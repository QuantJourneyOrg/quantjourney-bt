# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 24 - FX Cross-Sectional Momentum
================================================

Mode: weights (dollar-neutral price-return proxy).
Idea: rank four USD-quoted spot pairs by three-month return, go long the
strongest base currency and short the weakest. Gross exposure is one and net
exposure is zero before the engine's cash buffer.
Universe: EURUSD, GBPUSD, AUDUSD and NZDUSD provider spot-FX proxies.

This intentionally uses only XXX/USD pairs so every price move has the same
directional interpretation. It does not include forward carry, swaps, FX
margin, or executable bid/ask quotes.

Usage:
    ./strategy.sh example_weights_24_fx_cross_sectional_momentum
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


def build_fx_cross_sectional_momentum(
    close: pd.DataFrame,
    *,
    lookback: int = 63,
    names_per_side: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cross-sectional long/short signals and dollar-neutral weights."""
    scores = close.pct_change(lookback, fill_method=None)
    signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for date, row in scores.iterrows():
        valid = row.dropna()
        if len(valid) < 2 * names_per_side:
            continue
        longs = valid.nlargest(names_per_side).index
        shorts = valid.nsmallest(names_per_side).index
        signals.loc[date, longs] = 1.0
        signals.loc[date, shorts] = -1.0

    long_mask = (signals > 0.0).astype(float)
    short_mask = (signals < 0.0).astype(float)
    long_weights = (
        long_mask.div(long_mask.sum(axis=1).replace(0.0, float("nan")), axis=0).fillna(0.0) * 0.5
    )
    short_weights = (
        short_mask.div(short_mask.sum(axis=1).replace(0.0, float("nan")), axis=0).fillna(0.0) * 0.5
    )
    return signals, long_weights - short_weights


class FXCrossSectionalMomentum(Backtester):
    """Long the strongest and short the weakest USD-quoted FX pair."""

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        return build_fx_cross_sectional_momentum(close)[0]

    def _compute_weights(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        return build_fx_cross_sectional_momentum(close)[1]


async def main() -> None:
    strategy = FXCrossSectionalMomentum(
        **_credentials(),
        strategy_name="ExampleWeights24_FXCrossSectionalMomentum",
        strategy_type="Market Neutral FX Proxy",
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
