# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 08 - Intraday 1m EMA Scalp
==========================================

Mode: weights.
Idea: hold each name only while its fast EMA(9) is above its slow EMA(21) on
1-minute bars. A fast, high-turnover trend-follow / scalp template.
Universe: three predeclared liquid ETFs: SPY, QQQ and IWM.

Timeframe-grid note: this is the fastest cadence in the intraday grid
(1m -> 5m -> 30m -> 1h). At 1m, turnover is high and costs dominate, so this
template is mostly a stress-test of the engine's intraday timestamp handling
and signal-change rebalancing.

Usage:
    ./strategy.sh example_weights_08_intraday_1m_ema_scalp
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta

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


def _recent_period(days: int = 5) -> dict[str, str]:
    # yfinance serves at most ~7 calendar days of 1-minute bars.
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    return {"start": start.isoformat(), "end": end.isoformat()}


class IntradayEMAScalp1m(Backtester):
    """1-minute EMA(9/21) trend/cash strategy."""

    def _compute_signals(self) -> pd.DataFrame:
        fast = self.instruments_data.get_feature("EMA_9_close")
        slow = self.instruments_data.get_feature("EMA_21_close")
        valid = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.34)


async def main() -> None:
    strategy = IntradayEMAScalp1m(
        **_credentials(),
        strategy_name="ExampleWeights08_IntradayEMAScalp1m",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM"],
        backtest_period={"start": "2026-07-05", "end": "2026-07-11"},
        granularity="1m",
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.34,
        rebalance_policy=RebalancePolicy(frequency=None, rebalance_on_signal_change=True),
        indicators_config=[
            {"function": "EMA", "price_cols": ["close"], "params": {"periods": [9, 21]}},
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
