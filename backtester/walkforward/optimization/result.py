"""
OptimizationResult — rich data contract for optimizer output.

Institutional-grade result container that captures everything a quant
desk needs: best params, full trial history, parameter importance,
convergence metadata, and Pareto front for multi-objective studies.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class TrialRecord:
    """Single trial in the optimization study."""

    number: int
    params: Dict[str, Any]
    value: float
    values: Optional[List[float]] = None  # multi-objective
    metrics: Optional[Dict[str, float]] = None  # full metrics snapshot
    duration_seconds: float = 0.0
    pruned: bool = False
    state: str = "COMPLETE"  # COMPLETE | PRUNED | FAIL


@dataclass
class OptimizationResult:
    """
    Result of a parameter optimization study.

    Backwards-compatible: ``best_params``, ``best_objective``,
    ``n_evaluated``, ``elapsed_seconds``, ``all_results`` still work.

    New fields provide institutional-grade detail:
    - ``trials`` — structured list of every trial
    - ``param_importance`` — fANOVA-based importance scores
    - ``convergence_curve`` — running best objective per trial
    - ``pareto_front`` — non-dominated solutions (multi-objective)
    - ``study_metadata`` — sampler/pruner config, seed, etc.
    """

    # ── Core (backwards-compatible) ──
    best_params: Dict[str, Any]
    best_objective: float
    n_evaluated: int
    elapsed_seconds: float
    all_results: Optional[pd.DataFrame] = None  # param combos × metrics

    # ── Extended ──
    trials: List[TrialRecord] = field(default_factory=list)
    param_importance: Dict[str, float] = field(default_factory=dict)
    convergence_curve: List[float] = field(default_factory=list)
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    study_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Breakdown ──
    n_completed: int = 0
    n_pruned: int = 0
    n_failed: int = 0
    early_stopped: bool = False
    early_stop_reason: str = ""

    def best_trial(self) -> Optional[TrialRecord]:
        """Return the trial with the best objective value."""
        completed = [t for t in self.trials if t.state == "COMPLETE"]
        if not completed:
            return None
        return max(completed, key=lambda t: t.value)

    def top_k(self, k: int = 10) -> List[TrialRecord]:
        """Return top-k trials by objective value (descending)."""
        completed = [t for t in self.trials if t.state == "COMPLETE"]
        return sorted(completed, key=lambda t: t.value, reverse=True)[:k]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-safe dictionary."""
        return {
            "best_params": self.best_params,
            "best_objective": self.best_objective,
            "n_evaluated": self.n_evaluated,
            "n_completed": self.n_completed,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "early_stopped": self.early_stopped,
            "early_stop_reason": self.early_stop_reason,
            "param_importance": self.param_importance,
            "convergence_curve": self.convergence_curve,
            "pareto_front": self.pareto_front,
            "study_metadata": self.study_metadata,
            "trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "values": t.values,
                    "metrics": t.metrics,
                    "duration_seconds": round(t.duration_seconds, 3),
                    "pruned": t.pruned,
                    "state": t.state,
                }
                for t in self.trials
            ],
        }

    def summary(self, verbose: bool = True) -> str:
        """
        Generate a detailed text report of this optimization study.

        Args:
            verbose: Full report (True) or compact executive summary (False).

        Returns:
            Multi-line formatted text string.
        """
        from backtester.walkforward.optimization.summary import optimization_summary
        return optimization_summary(self, verbose=verbose)

    def summary_dict(self) -> Dict[str, Any]:
        """
        Build a comprehensive summary dictionary for programmatic access.

        Returns:
            Nested dict with sections: overview, configuration, best_trial,
            trial_statistics, convergence, param_importance, param_ranges,
            top_trials, stability, diagnostics.
        """
        from backtester.walkforward.optimization.summary import optimization_summary_dict
        return optimization_summary_dict(self)
