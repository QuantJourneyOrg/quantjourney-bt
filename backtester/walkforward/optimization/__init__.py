"""
Optimization subpackage — pluggable parameter optimizers.

Provides Grid Search and institutional-grade Optuna (TPE/CMA-ES/QMC)
optimization with warm-start, pruning, multi-objective, convergence
early-stopping, and parameter importance.

Usage::

    from backtester.walkforward.optimization import optimizer_factory

    # Quick — just grid search
    opt = optimizer_factory("grid", param_grid={"fast": [10, 20, 50]})
    result = opt.optimize_fn(eval_fn)

    # Institutional — Optuna with all the bells
    opt = optimizer_factory(
        "optuna",
        param_space={"fast": {"type": "int", "low": 5, "high": 50}},
        n_trials=200, sampler="tpe", pruner="hyperband",
    )
    result = opt.optimize_fn(
        eval_fn,
        progress_callback=lambda info: print(info),
        cancel_check=lambda: False,
    )

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from backtester.walkforward.optimization.base import Optimizer
from backtester.walkforward.optimization.result import OptimizationResult, TrialRecord
from backtester.walkforward.optimization.summary import (
    optimization_summary,
    optimization_summary_dict,
)

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "TrialRecord",
    "optimization_summary",
    "optimization_summary_dict",
    "optimizer_factory",
]


def optimizer_factory(method: str, **kwargs) -> Optimizer:
    """Create an optimizer instance by method name."""
    if method == "grid":
        from backtester.walkforward.optimization.grid import GridSearchOptimizer

        return GridSearchOptimizer(**kwargs)
    elif method == "optuna":
        from backtester.walkforward.optimization.optuna_ import OptunaOptimizer

        return OptunaOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimization method: {method!r}")
