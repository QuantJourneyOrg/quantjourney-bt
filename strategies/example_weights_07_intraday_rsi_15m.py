# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 07 - Intraday RSI 15m
=====================================

Mode: weights.
Idea: use yfinance-backed 15-minute bars from /bt/prepare and hold an equal
weight basket when RSI(14) is oversold.
Universe: liquid mega-cap stocks.
Granularity: 15m historical intraday data.

This example intentionally keeps the trading rule small: enter when RSI is
below 35, stay invested until RSI rises above 55, and rebalance when the signal
changes.

Usage:
    ./strategy.sh example_weights_07_intraday_rsi_15m
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone

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


def _recent_period(days: int = 21) -> dict[str, str]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return {"start": start.isoformat(), "end": end.isoformat()}


class IntradayRSI15m(Backtester):
    """Minimal 15-minute RSI mean-reversion strategy."""

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
                elif in_position[inst] and value > 55:
                    in_position[inst] = False
            signals.loc[date] = in_position.astype(float)

        return signals

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    strategy = IntradayRSI15m(
        **_credentials(),
        strategy_name="ExampleWeights07_IntradayRSI15m",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "META"],
        backtest_period=_recent_period(days=21),
        source="yfinance",
        granularity="15m",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency=None, rebalance_on_signal_change=True),
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
