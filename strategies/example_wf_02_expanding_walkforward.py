# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example WF 02 - Expanding Walk-Forward Validation
=================================================

Mode: weights + walk-forward.
Idea: same SMA(50/200) trend strategy as WF 01, but validated with an EXPANDING
walk-forward — the training window grows over time (anchored start, moving end)
while the test window slides forward. This mimics a strategy that keeps all
history as it accumulates.
Universe: five large US technology stocks.

What this teaches: rolling vs expanding schemes answer different questions.
Rolling asks "does a fixed-length recent history generalize?"; expanding asks
"does an ever-growing history generalize?". Compare the OOS Sharpe and overfit
ratio against WF 01 to see how the scheme changes the verdict.

Usage:
    ./strategy.sh example_wf_02_expanding_walkforward
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.walkforward import WalkForwardConfig, WalkForwardEngine
from backtester.walkforward.statistics.interpretation import interpret_metrics


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class SMATrendForWF(Backtester):
    """SMA(50/200) long/cash trend, used as the walk-forward subject."""

    def _compute_signals(self) -> pd.DataFrame:
        fast = self.instruments_data.get_feature("SMA_50_close")
        slow = self.instruments_data.get_feature("SMA_200_close")
        valid = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    strategy = SMATrendForWF(
        **_credentials(),
        strategy_name="ExampleWF02_ExpandingWalkForward",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period={"start": "2012-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}},
        ],
        benchmark_symbol="^GSPC",
        benchmark_name="S&P 500 Index",
        show_text_reports=False,
        save_portfolio_plots=False,
    )
    await strategy.run_strategy()

    config = WalkForwardConfig(
        scheme="expanding",
        train_months=24,       # minimum initial training window
        test_months=6,
        step_months=6,
        purge_days=5,
        embargo_pct=0.01,
    )
    engine = WalkForwardEngine(config=config, initial_capital=100_000)
    result = engine.run(strategy.portfolio_data)

    print(result.summary())

    verdicts = interpret_metrics({
        "overfit_ratio": result.overfit_ratio,
        "efficiency": result.efficiency,
        "sharpe_decay": result.sharpe_decay,
    })
    print("\nWalk-forward traffic lights:")
    for v in verdicts:
        print(f"  {v}")


if __name__ == "__main__":
    asyncio.run(main())
