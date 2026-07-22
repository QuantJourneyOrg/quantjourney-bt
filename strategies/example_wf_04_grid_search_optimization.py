# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example WF 04 - Grid Search Parameter Optimization
==================================================

Mode: weights + optimization.
Idea: find the best SMA fast/slow window pair for a trend strategy with an
exhaustive GRID SEARCH. Each candidate is scored by running a real backtest and
reading its annualized Sharpe.
Universe: canonical US sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY.

What this teaches: grid search evaluates every combination in a discrete grid.
It is simple and complete, but the cost grows with the product of the grid
sizes. Here the grid is deliberately small (3 x 3 = 9). The market data is
fetched once and cached server-side, so each candidate only recomputes signals
and the NAV locally.

Note: main() is synchronous because the optimizer's evaluate function is
synchronous; each candidate runs the async backtest via asyncio.run().

Usage:
    ./strategy.sh example_wf_04_grid_search_optimization
"""

import asyncio
import os

import numpy as np
import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.walkforward.optimization.grid import GridSearchOptimizer


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


class SMACrossoverTunable(Backtester):
    """SMA crossover whose windows are read from instance attributes."""

    def _compute_signals(self) -> pd.DataFrame:
        fast = self.instruments_data.get_feature(f"SMA_{self._fast}_close")
        slow = self.instruments_data.get_feature(f"SMA_{self._slow}_close")
        valid = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


def _build(fast: int, slow: int) -> SMACrossoverTunable:
    strategy = SMACrossoverTunable(
        **_credentials(),
        strategy_name=f"SMA_{fast}_{slow}",
        initial_capital=100_000,
        instruments=["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
        backtest_period={"start": "2000-01-03", "end": "2026-01-01"},
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [fast, slow]}},
        ],
        show_text_reports=False,
        save_portfolio_plots=False,
    )
    strategy._fast = fast
    strategy._slow = slow
    return strategy


def _sharpe(fast: int, slow: int) -> float:
    strategy = _build(fast, slow)
    asyncio.run(strategy.run_strategy())
    nav = strategy.portfolio_data.net_asset_value
    returns = nav.pct_change().dropna()
    if returns.std() == 0 or returns.empty:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def main() -> None:
    def evaluate(params) -> float:
        fast, slow = int(params["fast"]), int(params["slow"])
        if fast >= slow:
            return -999.0
        return _sharpe(fast, slow)

    optimizer = GridSearchOptimizer(
        param_grid={
            "fast": [20, 50, 80],
            "slow": [100, 150, 200],
        },
        objective="sharpe",
    )
    result = optimizer.optimize_fn(evaluate)

    print("Grid search results")
    print(f"  best params:  {result.best_params}")
    print(f"  best Sharpe:  {result.best_objective:.4f}")
    print(f"  evaluated:    {result.n_evaluated} combinations")


if __name__ == "__main__":
    main()
