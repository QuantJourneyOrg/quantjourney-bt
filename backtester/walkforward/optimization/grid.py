"""
Grid Search Optimizer — exhaustive Cartesian-product parameter sweep.

Evaluates all combinations in ``param_grid``, optionally capped at
``max_combinations`` (random subsample when the grid is too large).

Copyright (c) 2026 QuantJourney.
Updated: 07.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import random
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from backtester.walkforward.optimization.result import (
    OptimizationResult,
    TrialRecord,
)

logger = logging.getLogger(__name__)


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
        param_grid: dict[str, list] | None = None,
        objective: str = "sharpe",
        max_combinations: int = 500,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self._param_grid = param_grid or {}
        self._objective = objective
        self._max_combinations = max_combinations
        self._seed = seed

    # ── Combination builder ───────────────────────────────────────────

    def _build_combos(self) -> tuple[list[str], list[tuple]]:
        """Cartesian product, subsampled at max_combinations."""
        keys = list(self._param_grid.keys())
        values = list(self._param_grid.values())
        all_combos = list(itertools.product(*values))

        if len(all_combos) > self._max_combinations:
            rng = random.Random(self._seed)
            all_combos = rng.sample(all_combos, self._max_combinations)

        return keys, all_combos

    def _finalize(
        self,
        trials: list[TrialRecord],
        best_params: dict[str, Any],
        best_score: float,
        elapsed: float,
    ) -> OptimizationResult:
        """Build the OptimizationResult, flagging total failure loudly."""
        records = []
        for tr in trials:
            records.append({**tr.params, "objective": tr.value})
        results_df = pd.DataFrame(records)

        n_completed = sum(1 for t in trials if t.state == "COMPLETE")
        n_failed = sum(1 for t in trials if t.state == "FAIL")
        all_failed = bool(trials) and n_completed == 0

        if all_failed:
            logger.error(
                "GridSearchOptimizer: ALL %d trials failed — best_params is "
                "empty and the fold will run unoptimized. Check the "
                "backtester_factory / evaluate_fn errors logged above.",
                len(trials),
            )

        return OptimizationResult(
            best_params=best_params,
            best_objective=float(best_score),
            n_evaluated=len(trials),
            elapsed_seconds=elapsed,
            all_results=results_df,
            trials=trials,
            n_completed=n_completed,
            n_failed=n_failed,
            all_trials_failed=all_failed,
        )

    # ── Sync API ──────────────────────────────────────────────────────

    def optimize_fn(
        self,
        evaluate_fn: Callable[[dict[str, Any]], float],
    ) -> OptimizationResult:
        """
        Run grid search using a synchronous evaluation function.

        Args:
            evaluate_fn: ``params -> objective_value`` (higher = better).

        Returns:
            OptimizationResult with best params, objective, and all trial data.
        """
        t0 = time.time()
        keys, all_combos = self._build_combos()

        trials: list[TrialRecord] = []
        best_score = -np.inf
        best_params: dict[str, Any] = {}

        for i, combo in enumerate(all_combos):
            params = dict(zip(keys, combo, strict=True))
            trial_t0 = time.time()
            try:
                score = float(evaluate_fn(params))
                state = "COMPLETE"
            except Exception as e:
                logger.warning("Grid trial %d failed for %s: %s", i, params, e)
                score = -np.inf
                state = "FAIL"

            trials.append(
                TrialRecord(
                    number=i,
                    params=params,
                    value=score,
                    duration_seconds=time.time() - trial_t0,
                    state=state,
                )
            )

            if state == "COMPLETE" and score > best_score:
                best_score = score
                best_params = params.copy()

        return self._finalize(trials, best_params, best_score, time.time() - t0)

    # ── Async API (walk-forward integration) ──────────────────────────

    async def optimize(
        self,
        backtester_factory: Callable[..., Any],
        train_start: str,
        train_end: str,
        base_config: dict[str, Any],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OptimizationResult:
        """
        Protocol-compatible async grid search.

        Each trial's backtest is awaited directly on the caller's event
        loop — never via ``loop.run_until_complete`` (which raises
        ``RuntimeError`` inside a running loop and used to silently fail
        every trial).
        """
        t0 = time.time()
        keys, all_combos = self._build_combos()

        trials: list[TrialRecord] = []
        best_score = -np.inf
        best_params: dict[str, Any] = {}

        for i, combo in enumerate(all_combos):
            if cancel_check and cancel_check():
                logger.info("GridSearchOptimizer: cancelled after %d trials", i)
                break

            params = dict(zip(keys, combo, strict=True))
            trial_t0 = time.time()
            try:
                score = await self._evaluate_backtest(
                    backtester_factory, params, train_start, train_end, base_config
                )
                state = "COMPLETE"
            except Exception as e:
                logger.warning("Grid trial %d failed for %s: %s", i, params, e)
                score = -np.inf
                state = "FAIL"

            trials.append(
                TrialRecord(
                    number=i,
                    params=params,
                    value=score,
                    duration_seconds=time.time() - trial_t0,
                    state=state,
                )
            )

            if state == "COMPLETE" and score > best_score:
                best_score = score
                best_params = params.copy()

            if progress_callback:
                progress_callback(
                    {
                        "trial": i,
                        "total": len(all_combos),
                        "value": score,
                        "best": best_score,
                        "params": params,
                        "elapsed": round(time.time() - t0, 1),
                    }
                )

        return self._finalize(trials, best_params, best_score, time.time() - t0)

    @staticmethod
    async def _evaluate_backtest(
        backtester_factory: Callable[..., Any],
        params: dict[str, Any],
        train_start: str,
        train_end: str,
        base_config: dict[str, Any],
    ) -> float:
        """Run one IS backtest for a param combo and return its Sharpe."""
        merged = {
            **base_config,
            **params,
            "backtest_period": {"start": train_start, "end": train_end},
        }
        bt = backtester_factory(**merged)
        # Same resolution as FoldRunner: the real Backtester exposes
        # run_strategy(), lightweight test doubles expose run().
        runner = getattr(bt, "run_strategy", None) or getattr(bt, "run", None)
        if runner is None:
            raise ValueError("backtester_factory result must provide run_strategy() or run()")
        run_result = runner()
        if inspect.isawaitable(run_result):
            await run_result
        nav = bt.portfolio_data.net_asset_value
        returns = nav.pct_change().dropna()
        if returns.std() == 0 or len(returns) < 2:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))
