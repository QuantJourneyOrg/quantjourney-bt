"""
Optuna Optimizer — Institutional-grade Bayesian parameter optimization.

Multi-sampler (TPE / CMA-ES / QMC / Random), adaptive pruning,
warm-start, multi-objective Pareto, convergence early-stopping,
parameter importance, and real-time progress callbacks.

Guarded import: ``pip install optuna``.

Usage::

    optimizer = OptunaOptimizer(
        param_space={
            "fast_window":  {"type": "int",   "low": 5,  "high": 50},
            "slow_window":  {"type": "int",   "low": 50, "high": 200},
            "risk_pct":     {"type": "float", "low": 0.01, "high": 0.10, "log": True},
            "regime":       {"type": "categorical", "choices": ["trend", "mean_rev"]},
        },
        n_trials=200,
        sampler="tpe",
        pruner="hyperband",
    )

    result = optimizer.optimize_fn(
        evaluate_fn,
        progress_callback=lambda info: print(info["trial"], info["best"]),
        cancel_check=lambda: False,
    )

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester.walkforward.optimization.result import (
    OptimizationResult,
    TrialRecord,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
_VALID_SAMPLERS = ("tpe", "cmaes", "qmc", "random")
_VALID_PRUNERS = ("median", "percentile", "hyperband", "none")


def _check_optuna():
    """Guard import — raises helpful error if optuna missing."""
    try:
        import optuna  # noqa: F401
        return optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for OptunaOptimizer. "
            "Install with: pip install optuna"
        )


# ═══════════════════════════════════════════════════════════════════
# Param-space helpers
# ═══════════════════════════════════════════════════════════════════
def _suggest_param(trial, name: str, spec: dict) -> Any:
    """
    Suggest a parameter from a rich spec dict.

    Spec keys:
        type:    "int" | "float" | "categorical"
        low:     lower bound (int/float)
        high:    upper bound (int/float)
        step:    step size (optional, int/float)
        log:     log-scale (optional bool, default False)
        choices: list of values (categorical)
    """
    ptype = spec.get("type", "float")

    if ptype == "int":
        kwargs: Dict[str, Any] = {"name": name, "low": int(spec["low"]), "high": int(spec["high"])}
        if spec.get("step"):
            kwargs["step"] = int(spec["step"])
        if spec.get("log"):
            kwargs["log"] = True
        return trial.suggest_int(**kwargs)

    elif ptype == "float":
        kwargs = {"name": name, "low": float(spec["low"]), "high": float(spec["high"])}
        if spec.get("step"):
            kwargs["step"] = float(spec["step"])
        if spec.get("log"):
            kwargs["log"] = True
        return trial.suggest_float(**kwargs)

    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])

    else:
        raise ValueError(f"Unknown param type {ptype!r} for {name}")


def _make_distribution(spec: dict):
    """Convert a param spec to an Optuna Distribution (for warm-start)."""
    import optuna

    ptype = spec.get("type", "float")
    if ptype == "int":
        return optuna.distributions.IntDistribution(
            low=int(spec["low"]),
            high=int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    elif ptype == "float":
        kw: Dict[str, Any] = {"low": float(spec["low"]), "high": float(spec["high"])}
        if spec.get("step"):
            kw["step"] = float(spec["step"])
        if spec.get("log"):
            kw["log"] = True
        return optuna.distributions.FloatDistribution(**kw)
    elif ptype == "categorical":
        return optuna.distributions.CategoricalDistribution(spec["choices"])
    else:
        raise ValueError(f"Unknown param type {ptype!r}")


# ═══════════════════════════════════════════════════════════════════
# Sampler factory
# ═══════════════════════════════════════════════════════════════════
def _create_sampler(name: str, seed: int, n_startup: int = 10):
    """Instantiate Optuna sampler by name."""
    import optuna

    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            constant_liar=True,
            n_startup_trials=n_startup,
        )
    elif name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=n_startup)
    elif name == "qmc":
        return optuna.samplers.QMCSampler(seed=seed)
    elif name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(
            f"Unknown sampler {name!r}. Valid: {_VALID_SAMPLERS}"
        )


# ═══════════════════════════════════════════════════════════════════
# Pruner factory
# ═══════════════════════════════════════════════════════════════════
def _create_pruner(name: str):
    """Instantiate Optuna pruner by name."""
    import optuna

    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    elif name == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=25.0, n_startup_trials=5, n_warmup_steps=3,
        )
    elif name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=10, reduction_factor=3,
        )
    elif name == "none":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(
            f"Unknown pruner {name!r}. Valid: {_VALID_PRUNERS}"
        )


# ═══════════════════════════════════════════════════════════════════
# OptunaOptimizer
# ═══════════════════════════════════════════════════════════════════
class OptunaOptimizer:
    """
    Institutional-grade Bayesian optimizer powered by Optuna.

    Features:
        - Multi-sampler: TPE (default), CMA-ES, QMC, Random
        - Adaptive pruning: Median, Percentile, Hyperband, or None
        - Warm-start from previous study results
        - Multi-objective Pareto optimization
        - Convergence early-stopping with configurable patience
        - Real-time progress callbacks
        - Cancel support via callback
        - Parameter importance (fANOVA)
        - Cross-parameter constraint support
        - Full TrialRecord history

    Param space format (supports both legacy tuple and rich dict)::

        # Legacy (backwards-compatible):
        {"fast": ("int", 5, 50), "slow": ("float", 0.01, 0.1)}

        # Rich (recommended):
        {
            "fast": {"type": "int", "low": 5, "high": 50, "step": 5},
            "slow": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "mode": {"type": "categorical", "choices": ["trend", "mean_rev"]},
        }
    """

    def __init__(
        self,
        param_space: Dict[str, Any] | None = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: float | None = None,
        objective: str = "sharpe",
        direction: str = "maximize",
        directions: List[str] | None = None,  # multi-objective
        sampler: str = "tpe",
        pruner: str = "median",
        seed: int = 42,
        n_startup_trials: int = 10,
        patience: int | None = None,  # convergence early-stop
        warm_start_trials: List[Dict[str, Any]] | None = None,
        constraints: List[Dict[str, Any]] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        _check_optuna()

        self._param_space = self._normalise_space(param_space or {})
        self._n_trials = n_trials
        self._n_jobs = n_jobs
        self._timeout = timeout
        self._objective = objective
        self._direction = direction
        self._directions = directions
        self._sampler_name = sampler
        self._pruner_name = pruner
        self._seed = seed
        self._n_startup = n_startup_trials
        self._patience = patience
        self._warm_start_trials = warm_start_trials or []
        self._constraints = constraints or []
        self._verbose = verbose

    # ── Normalise legacy tuple format to dict ──
    @staticmethod
    def _normalise_space(space: Dict[str, Any]) -> Dict[str, dict]:
        """Convert mixed tuple/dict param specs to uniform dict format."""
        out: Dict[str, dict] = {}
        for name, spec in space.items():
            if isinstance(spec, dict):
                out[name] = spec
            elif isinstance(spec, (tuple, list)):
                # Legacy: ("int", 5, 50) or ("categorical", [...])
                ptype = spec[0]
                if ptype == "categorical":
                    out[name] = {"type": "categorical", "choices": spec[1]}
                else:
                    out[name] = {"type": ptype, "low": spec[1], "high": spec[2]}
            else:
                raise ValueError(
                    f"Invalid param spec for {name!r}: {spec!r}. "
                    "Expected dict or tuple."
                )
        return out

    # ── Core API ──
    def optimize_fn(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float | Tuple[float, ...]],
        *,
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        metrics_fn: Callable[[Dict[str, Any]], Dict[str, float]] | None = None,
    ) -> OptimizationResult:
        """
        Run optimization using a synchronous evaluation function.

        Args:
            evaluate_fn: ``params -> objective_value`` (higher = better),
                or ``params -> (val1, val2, ...)`` for multi-objective.
            progress_callback: Called after each trial with a dict:
                ``{"trial": int, "total": int, "value": float, "best": float,
                  "params": dict, "elapsed": float, "eta": float}``.
            cancel_check: ``() -> bool``; if returns True, study stops.
            metrics_fn: Optional ``params -> {metric: value}`` for rich
                trial records. If None, only the objective is stored.

        Returns:
            OptimizationResult with full institutional-grade detail.
        """
        import optuna

        if not self._verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        t0 = time.time()
        is_multi = self._directions is not None and len(self._directions) > 1
        trial_records: List[TrialRecord] = []
        convergence: List[float] = []
        best_so_far = -np.inf
        no_improve_count = 0
        early_stopped = False
        early_stop_reason = ""
        _cancel_flag = False

        # Max no-improvement patience
        patience = self._patience or max(15, self._n_trials // 3)

        # ── Create study ──
        sampler = _create_sampler(self._sampler_name, self._seed, self._n_startup)
        pruner = _create_pruner(self._pruner_name)

        if is_multi:
            study = optuna.create_study(
                directions=self._directions,
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.create_study(
                direction=self._direction,
                sampler=sampler,
                pruner=pruner,
            )

        # ── Warm-start: enqueue prior best trials ──
        if self._warm_start_trials:
            distributions = {
                name: _make_distribution(spec)
                for name, spec in self._param_space.items()
            }
            for prior in self._warm_start_trials:
                try:
                    trial_obj = optuna.trial.create_trial(
                        params=prior.get("params", {}),
                        distributions=distributions,
                        values=[prior.get("value", 0.0)],
                    )
                    study.add_trial(trial_obj)
                except Exception as e:
                    logger.warning("Warm-start trial skipped: %s", e)

        # ── Objective ──
        def objective(trial: optuna.Trial) -> float | Tuple[float, ...]:
            nonlocal best_so_far, no_improve_count, _cancel_flag

            # Cancel check
            if cancel_check and cancel_check():
                _cancel_flag = True
                trial.study.stop()
                raise optuna.TrialPruned()

            # Convergence early-stop
            if no_improve_count >= patience:
                early_stop_reason_local = (
                    f"No improvement for {no_improve_count} consecutive trials"
                )
                nonlocal early_stopped, early_stop_reason
                early_stopped = True
                early_stop_reason = early_stop_reason_local
                trial.study.stop()
                raise optuna.TrialPruned()

            # Suggest params
            params: Dict[str, Any] = {}
            for name, spec in self._param_space.items():
                params[name] = _suggest_param(trial, name, spec)

            # Cross-parameter constraints
            for c in self._constraints:
                if c.get("type") == "less_than":
                    if params.get(c["param"]) >= params.get(c["than"]):
                        raise optuna.TrialPruned()

            trial_t0 = time.time()
            try:
                value = evaluate_fn(params)
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning("Trial %d failed: %s", trial.number, e)
                trial_records.append(TrialRecord(
                    number=trial.number,
                    params=params,
                    value=float("-inf"),
                    duration_seconds=time.time() - trial_t0,
                    state="FAIL",
                ))
                raise optuna.TrialPruned()

            trial_duration = time.time() - trial_t0

            # Handle multi-objective tuple
            if isinstance(value, (tuple, list)):
                primary = float(value[0])
                values_list = [float(v) for v in value]
            else:
                primary = float(value)
                values_list = None

            # Fetch rich metrics if available
            trial_metrics = None
            if metrics_fn:
                try:
                    trial_metrics = metrics_fn(params)
                except Exception:
                    pass

            # Record
            record = TrialRecord(
                number=trial.number,
                params=params,
                value=primary,
                values=values_list,
                metrics=trial_metrics,
                duration_seconds=trial_duration,
                pruned=False,
                state="COMPLETE",
            )
            trial_records.append(record)

            # Convergence tracking
            if primary > best_so_far:
                best_so_far = primary
                no_improve_count = 0
            else:
                no_improve_count += 1
            convergence.append(best_so_far)

            # Progress callback
            if progress_callback:
                elapsed = time.time() - t0
                completed = len([t for t in trial_records if t.state == "COMPLETE"])
                avg_time = elapsed / max(completed, 1)
                remaining = self._n_trials - trial.number - 1
                eta = avg_time * remaining

                progress_callback({
                    "trial": trial.number,
                    "total": self._n_trials,
                    "value": primary,
                    "best": best_so_far,
                    "params": params,
                    "metrics": trial_metrics,
                    "elapsed": round(elapsed, 1),
                    "eta": round(max(0, eta), 1),
                    "no_improve": no_improve_count,
                    "patience": patience,
                })

            if is_multi and values_list:
                return tuple(values_list)
            return primary

        # ── Run study ──
        try:
            study.optimize(
                objective,
                n_trials=self._n_trials,
                n_jobs=self._n_jobs,
                timeout=self._timeout,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            early_stopped = True
            early_stop_reason = "KeyboardInterrupt"

        elapsed = time.time() - t0

        # ── Collect results ──
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        # Best params
        if is_multi:
            best_trials_pareto = study.best_trials
            best_params = best_trials_pareto[0].params if best_trials_pareto else {}
            best_value = best_trials_pareto[0].values[0] if best_trials_pareto else 0.0
            pareto_front = [
                {"params": t.params, "values": list(t.values)}
                for t in best_trials_pareto
            ]
        else:
            best_params = study.best_params if completed else {}
            best_value = study.best_value if completed else 0.0
            pareto_front = []

        # ── Parameter importance ──
        param_importance: Dict[str, float] = {}
        if len(completed) >= 5:
            try:
                param_importance = optuna.importance.get_param_importances(study)
            except Exception as e:
                logger.warning("Param importance computation failed: %s", e)

        # ── Build results DataFrame (backwards-compat) ──
        if trial_records:
            rows = []
            for tr in trial_records:
                row = {**tr.params, "objective": tr.value}
                if tr.metrics:
                    row.update(tr.metrics)
                rows.append(row)
            results_df = pd.DataFrame(rows)
        else:
            results_df = pd.DataFrame()

        # ── Study metadata ──
        metadata = {
            "sampler": self._sampler_name,
            "pruner": self._pruner_name,
            "seed": self._seed,
            "n_trials_requested": self._n_trials,
            "n_jobs": self._n_jobs,
            "timeout": self._timeout,
            "patience": patience,
            "n_startup_trials": self._n_startup,
            "direction": self._direction,
            "directions": self._directions,
            "warm_start_count": len(self._warm_start_trials),
            "param_space": self._param_space,
            "constraints": self._constraints if hasattr(self, "_constraints") else [],
        }

        return OptimizationResult(
            best_params=best_params,
            best_objective=float(best_value),
            n_evaluated=len(trial_records),
            elapsed_seconds=elapsed,
            all_results=results_df,
            trials=trial_records,
            param_importance=param_importance,
            convergence_curve=convergence,
            pareto_front=pareto_front,
            study_metadata=metadata,
            n_completed=len(completed),
            n_pruned=len(pruned),
            n_failed=len(failed),
            early_stopped=early_stopped,
            early_stop_reason=early_stop_reason,
        )

    # ── Protocol-compatible async wrapper ──
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
        """
        Protocol-compatible async interface for walk-forward integration.

        Args:
            backtester_factory: Callable that produces a fresh backtester.
            train_start: IS window start date.
            train_end: IS window end date.
            base_config: Strategy config to override with param combos.
            progress_callback: Optional real-time progress callback.
            cancel_check: Optional cancellation check callback.

        Returns:
            OptimizationResult with full detail.
        """
        import asyncio

        def evaluate_fn(params: Dict[str, Any]) -> float:
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

        return self.optimize_fn(
            evaluate_fn,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
