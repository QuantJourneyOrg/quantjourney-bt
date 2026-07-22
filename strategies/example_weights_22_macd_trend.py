# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 22 - MACD Trend
===============================

Mode: weights.
Idea: hold each name while its MACD line is above its signal line — a momentum
trend filter that reacts faster than a simple long-window SMA crossover.
Universe: canonical US sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY.

Signal: MACD = EMA(12) - EMA(26); signal line = EMA(9) of MACD.
- MACD > signal line  -> long
- MACD < signal line  -> flat
MACD is computed inline from adjusted close (a multi-output indicator that does
not map to the single-series `indicators_config` path).

Usage:
    ./strategy.sh example_weights_22_macd_trend
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


class MACDTrend(Backtester):
    """Long while MACD is above its signal line."""

    FAST, SLOW, SIGNAL = 12, 26, 9

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        ema_fast = close.ewm(span=self.FAST, adjust=False).mean()
        ema_slow = close.ewm(span=self.SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.SIGNAL, adjust=False).mean()
        valid = close.notna()
        return (macd > signal_line).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    strategy = MACDTrend(
        **_credentials(),
        strategy_name="ExampleWeights22_MACDTrend",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
        backtest_period={"start": "2000-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
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
