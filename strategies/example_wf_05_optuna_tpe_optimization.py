# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example WF 05 - Optuna TPE Optimization + Walk-Forward
======================================================

Mode: weights + optimization + walk-forward.
Idea: use Optuna's Tree-structured Parzen Estimator (TPE) to search SMA fast/slow
windows over continuous integer ranges, then validate the winning parameters
with a rolling walk-forward.
Universe: canonical US sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY.

What this teaches: for anything past a tiny grid, Bayesian optimization (Optuna
TPE) finds good parameters in far fewer evaluations than exhaustive grid search
by adaptively sampling promising regions. A ``less_than`` constraint enforces
fast < slow. After optimization, the best params are re-run and validated with
walk-forward so you never trust an in-sample optimum on its own.

Note: main() is synchronous; each Optuna trial runs the async backtest via
asyncio.run(). Market data is fetched once and cached, so trials only recompute
signals/NAV. n_trials is kept modest for a runnable example.

Requires the optional Optuna dependency:
    pip install optuna

Usage:
    ./strategy.sh example_wf_05_optuna_tpe_optimization
"""

import asyncio
import os

import numpy as np
import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.walkforward import WalkForwardConfig, WalkForwardEngine
from backtester.walkforward.optimization import optimizer_factory
from backtester.walkforward.statistics.interpretation import interpret_metrics


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


def _run(fast: int, slow: int) -> SMACrossoverTunable:
    strategy = _build(fast, slow)
    asyncio.run(strategy.run_strategy())
    return strategy


def _sharpe_of(strategy) -> float:
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
        return _sharpe_of(_run(fast, slow))

    optimizer = optimizer_factory(
        "optuna",
        param_space={
            "fast": {"type": "int", "low": 10, "high": 80},
            "slow": {"type": "int", "low": 100, "high": 250},
        },
        n_trials=30,
        sampler="tpe",
        pruner="median",
        seed=42,
        constraints=[{"type": "less_than", "param": "fast", "than": "slow"}],
    )
    result = optimizer.optimize_fn(evaluate)

    best_fast = int(result.best_params["fast"])
    best_slow = int(result.best_params["slow"])
    print("Optuna TPE results")
    print(f"  best params:  fast={best_fast}, slow={best_slow}")
    print(f"  best Sharpe:  {result.best_objective:.4f}")
    print(f"  evaluated:    {result.n_evaluated} trials")

    # Validate the winner out-of-sample with rolling walk-forward.
    best = _run(best_fast, best_slow)
    config = WalkForwardConfig(
        scheme="rolling",
        train_months=24,
        test_months=6,
        step_months=6,
        purge_days=5,
        extra_pre_oos_purge_pct=0.01,
    )
    wf = WalkForwardEngine(config=config, initial_capital=100_000).run(best.portfolio_data)
    print("\nWalk-forward validation of the Optuna-best params:")
    print(wf.summary())

    verdicts = interpret_metrics(
        {
            "overfit_ratio": wf.overfit_ratio,
            "efficiency": wf.efficiency,
            "sharpe_decay": wf.sharpe_decay,
        }
    )
    for v in verdicts:
        print(f"  {v}")


if __name__ == "__main__":
    main()
