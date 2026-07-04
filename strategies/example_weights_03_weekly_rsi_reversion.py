# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 03 - Weekly RSI Mean Reversion
==============================================

Mode: weights.
Idea: enter when RSI(14) is below 35, stay long until RSI rises above 60.
Universe: five liquid mega-cap stocks.
Rebalance: weekly on Friday.

This example uses a small state machine inside _compute_signals instead of a
one-bar condition, so entries and exits can have different thresholds.

Usage:
    ./strategy.sh example_weights_03_weekly_rsi_reversion
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


class WeeklyRSIReversion(Backtester):
    """RSI mean-reversion strategy with weekly execution."""

    def _compute_signals(self) -> pd.DataFrame:
        rsi = self.instruments_data.get_feature("RSI_14_close")
        signals = pd.DataFrame(0.0, index=rsi.index, columns=rsi.columns)
        in_position = pd.Series(False, index=rsi.columns)

        for date, row in rsi.iterrows():
            for inst, value in row.items():
                if pd.isna(value):
                    in_position[inst] = False
                elif not in_position[inst] and value < 35:
                    in_position[inst] = True
                elif in_position[inst] and value > 60:
                    in_position[inst] = False
            signals.loc[date] = in_position.astype(float)

        return signals

    def _compute_weights(self) -> pd.DataFrame:
        signals = self.signals
        active = signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    strategy = WeeklyRSIReversion(
        **_credentials(),
        strategy_name="ExampleWeights03_WeeklyRSIReversion",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "META", "AMZN"],
        backtest_period={"start": "2018-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency="W", weekday=4),
        indicators_config=[
            {"function": "RSI", "price_cols": ["close"], "params": {"periods": [14]}},
        ],
        benchmark_symbol="QQQ",
        benchmark_name="Nasdaq 100 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
