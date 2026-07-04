"""
Grid Search Optimizer — exhaustive Cartesian-product parameter sweep.

Evaluates all combinations in ``param_grid``, optionally capped at
``max_combinations`` (random subsample when the grid is too large).

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtester.walkforward.optimization.result import OptimizationResult


class GridSearchOptimizer:
    """
    Exhaustive grid search over a discrete parameter space.

    Usage::

        optimizer = GridSearchOptimizer(
            param_grid={"fast": [10, 20, 50], "slow": [100, 150, 200]},
            objective="sharpe",
        )
        result = optimizer.optimize_fn(evaluate_fn)

    ``evaluate_fn(params: dict) -> float`` should return the objective
    value (higher is better by default).
    """

    def __init__(
        self,
        param_grid: Dict[str, list] | None = None,
        objective: str = "sharpe",
        max_combinations: int = 500,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self._param_grid = param_grid or {}
        self._objective = objective
        self._max_combinations = max_combinations
        self._seed = seed

    def optimize_fn(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
    ) -> OptimizationResult:
        """
        Run grid search using a synchronous evaluation function.

        Args:
            evaluate_fn: ``params -> objective_value`` (higher = better).

        Returns:
            OptimizationResult with best params, objective, and all trial data.
        """
        t0 = time.time()

        # Build all combinations
        keys = list(self._param_grid.keys())
        values = list(self._param_grid.values())
        all_combos = list(itertools.product(*values))

        # Subsample if too many
        if len(all_combos) > self._max_combinations:
            rng = random.Random(self._seed)
            all_combos = rng.sample(all_combos, self._max_combinations)

        # Evaluate
        records: List[Dict[str, Any]] = []
        best_score = -np.inf
        best_params: Dict[str, Any] = {}

        for combo in all_combos:
            params = dict(zip(keys, combo))
            try:
                score = evaluate_fn(params)
            except Exception:
                score = -np.inf

            records.append({**params, "objective": score})

            if score > best_score:
                best_score = score
                best_params = params.copy()

        elapsed = time.time() - t0

        results_df = pd.DataFrame(records)

        return OptimizationResult(
            best_params=best_params,
            best_objective=float(best_score),
            n_evaluated=len(records),
            elapsed_seconds=elapsed,
            all_results=results_df,
        )

    async def optimize(
        self,
        backtester_factory: Callable[..., Any],
        train_start: str,
        train_end: str,
        base_config: Dict[str, Any],
        *,
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OptimizationResult:
        """Protocol-compatible async wrapper around optimize_fn."""

        def evaluate_fn(params: Dict[str, Any]) -> float:
            import asyncio

            merged = {
                **base_config,
                **params,
                "backtest_period": {"start": train_start, "end": train_end},
            }
            bt = backtester_factory(**merged)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(bt.run())
            nav = bt.portfolio_data.net_asset_value
            returns = nav.pct_change().dropna()
            if returns.std() == 0 or len(returns) < 2:
                return 0.0
            return float(returns.mean() / returns.std() * np.sqrt(252))

        return self.optimize_fn(evaluate_fn)
