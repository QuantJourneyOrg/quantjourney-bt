"""
Optimization Summary — Institutional-grade text report generator.
=================================================================

Generates a detailed, structured text report from an OptimizationResult,
covering every aspect of the study: configuration, trial statistics,
convergence analysis, param importance, top-k trials, stability
diagnostics, and actionable recommendations.

Inspired by the kind of report a quant desk at Citadel / Two Sigma
would produce after a parameter optimization study.

Usage::

    from backtester.walkforward.optimization.summary import optimization_summary
    text = optimization_summary(result)
    print(text)

Or as a dict for programmatic access::

    from backtester.walkforward.optimization.summary import optimization_summary_dict
    data = optimization_summary_dict(result)

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backtester.walkforward.optimization.result import OptimizationResult, TrialRecord


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _fmt(v: float, decimals: int = 4) -> str:
    """Smart formatting: large numbers → 2dp, small → 4dp."""
    if abs(v) >= 100:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:.{decimals}f}"
    return f"{v:.{max(decimals, 6)}f}"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def _bar(value: float, max_val: float, width: int = 30) -> str:
    """ASCII bar chart segment."""
    if max_val == 0:
        return ""
    filled = int(value / max_val * width)
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> str:
    return f"\n{'═' * 72}\n  {title}\n{'═' * 72}\n"


def _subsection(title: str) -> str:
    return f"\n  ── {title} {'─' * max(0, 60 - len(title))}\n"


# ═══════════════════════════════════════════════════════════════════════
# Summary Dict (programmatic)
# ═══════════════════════════════════════════════════════════════════════

def optimization_summary_dict(result: OptimizationResult) -> Dict[str, Any]:
    """
    Build a comprehensive summary dictionary from an OptimizationResult.

    Returns a nested dict with sections:
        - overview: high-level stats
        - configuration: sampler, pruner, etc.
        - best_trial: winning params and objective
        - trial_statistics: mean/std/median/percentiles of objectives
        - convergence: improvement rate, stagnation analysis
        - param_importance: importance ranking
        - param_ranges: observed ranges per parameter
        - top_trials: top-10 trials
        - stability: how stable the best region is
        - diagnostics: warnings and recommendations
    """
    completed = [t for t in result.trials if t.state == "COMPLETE"]
    pruned = [t for t in result.trials if t.state == "PRUNED"]
    failed = [t for t in result.trials if t.state == "FAIL"]
    objectives = [t.value for t in completed]

    meta = result.study_metadata or {}
    out: Dict[str, Any] = {}

    # ── Overview ──
    out["overview"] = {
        "total_trials": result.n_evaluated,
        "completed": len(completed),
        "pruned": len(pruned),
        "failed": len(failed),
        "elapsed_seconds": result.elapsed_seconds,
        "elapsed_human": _duration(result.elapsed_seconds),
        "best_objective": result.best_objective,
        "early_stopped": result.early_stopped,
        "early_stop_reason": result.early_stop_reason,
        "avg_trial_time": result.elapsed_seconds / max(len(completed), 1),
        "trials_per_second": len(completed) / max(result.elapsed_seconds, 0.001),
    }

    # ── Configuration ──
    out["configuration"] = {
        "sampler": meta.get("sampler", "unknown"),
        "pruner": meta.get("pruner", "unknown"),
        "direction": meta.get("direction", "maximize"),
        "directions": meta.get("directions"),
        "n_trials_requested": meta.get("n_trials_requested", "?"),
        "n_jobs": meta.get("n_jobs", 1),
        "timeout": meta.get("timeout"),
        "patience": meta.get("patience"),
        "seed": meta.get("seed"),
        "n_startup_trials": meta.get("n_startup_trials"),
        "warm_start_count": meta.get("warm_start_count", 0),
    }

    # ── Best trial ──
    best = result.best_trial()
    if best:
        out["best_trial"] = {
            "number": best.number,
            "objective": best.value,
            "params": best.params,
            "metrics": best.metrics,
            "duration": best.duration_seconds,
        }
    else:
        out["best_trial"] = None

    # ── Trial statistics ──
    if objectives:
        obj_arr = np.array(objectives)
        out["trial_statistics"] = {
            "count": len(objectives),
            "mean": float(np.mean(obj_arr)),
            "std": float(np.std(obj_arr)),
            "median": float(np.median(obj_arr)),
            "min": float(np.min(obj_arr)),
            "max": float(np.max(obj_arr)),
            "q25": float(np.percentile(obj_arr, 25)),
            "q75": float(np.percentile(obj_arr, 75)),
            "q90": float(np.percentile(obj_arr, 90)),
            "q95": float(np.percentile(obj_arr, 95)),
            "iqr": float(np.percentile(obj_arr, 75) - np.percentile(obj_arr, 25)),
            "coefficient_of_variation": float(np.std(obj_arr) / abs(np.mean(obj_arr)))
            if np.mean(obj_arr) != 0 else float("inf"),
            "skewness": float(_skewness(obj_arr)),
            "kurtosis": float(_kurtosis(obj_arr)),
        }
    else:
        out["trial_statistics"] = {}

    # ── Convergence analysis ──
    if result.convergence_curve:
        curve = result.convergence_curve
        improvements = sum(1 for i in range(1, len(curve))
                           if curve[i] > curve[i - 1])
        last_improve = 0
        for i in range(1, len(curve)):
            if curve[i] > curve[i - 1]:
                last_improve = i

        # Phase analysis: early vs late improvements
        mid_point = len(curve) // 2
        early_improvements = sum(1 for i in range(1, min(mid_point, len(curve)))
                                  if curve[i] > curve[i - 1])
        late_improvements = sum(1 for i in range(mid_point, len(curve))
                                 if curve[i] > curve[i - 1])

        total_gain = curve[-1] - curve[0] if len(curve) > 1 else 0
        first_half_gain = curve[mid_point] - curve[0] if mid_point > 0 else 0
        second_half_gain = curve[-1] - curve[mid_point] if mid_point < len(curve) else 0

        out["convergence"] = {
            "total_improvements": improvements,
            "improvement_rate": improvements / max(len(curve) - 1, 1),
            "last_improvement_at_trial": last_improve,
            "stagnation_tail": len(curve) - 1 - last_improve,
            "total_gain": total_gain,
            "first_half_gain": first_half_gain,
            "second_half_gain": second_half_gain,
            "early_improvements": early_improvements,
            "late_improvements": late_improvements,
            "gain_ratio_first_half": first_half_gain / total_gain if total_gain else 0,
            "fully_converged": (len(curve) - 1 - last_improve) > len(curve) * 0.3,
        }
    else:
        out["convergence"] = {}

    # ── Parameter importance ──
    out["param_importance"] = dict(
        sorted(result.param_importance.items(), key=lambda x: x[1], reverse=True)
    ) if result.param_importance else {}

    # ── Parameter ranges observed ──
    if completed:
        param_names = list(completed[0].params.keys())
        param_ranges: Dict[str, Dict[str, Any]] = {}
        for pname in param_names:
            vals = [t.params.get(pname) for t in completed if pname in t.params]
            numeric_vals = [v for v in vals if isinstance(v, (int, float))]
            if numeric_vals:
                arr = np.array(numeric_vals)
                # Best region: params from top-10% trials
                top_pct = max(3, int(len(completed) * 0.1))
                top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_pct]
                top_vals = [t.params[pname] for t in top_trials
                            if pname in t.params and isinstance(t.params[pname], (int, float))]
                top_arr = np.array(top_vals) if top_vals else arr

                param_ranges[pname] = {
                    "type": "numeric",
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "best_region_min": float(top_arr.min()),
                    "best_region_max": float(top_arr.max()),
                    "best_region_mean": float(top_arr.mean()),
                    "best_value": result.best_params.get(pname),
                }
            else:
                # Categorical
                counts = Counter(vals)
                top_pct = max(3, int(len(completed) * 0.1))
                top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_pct]
                top_vals = [t.params[pname] for t in top_trials if pname in t.params]
                top_counts = Counter(top_vals)
                param_ranges[pname] = {
                    "type": "categorical",
                    "distribution": dict(counts),
                    "top_region_distribution": dict(top_counts),
                    "best_value": result.best_params.get(pname),
                }
        out["param_ranges"] = param_ranges
    else:
        out["param_ranges"] = {}

    # ── Top trials ──
    out["top_trials"] = [
        {
            "rank": i + 1,
            "number": t.number,
            "objective": t.value,
            "params": t.params,
            "metrics": t.metrics,
            "duration": t.duration_seconds,
        }
        for i, t in enumerate(result.top_k(10))
    ]

    # ── Stability analysis ──
    if len(completed) >= 5:
        top5 = result.top_k(5)
        top5_vals = [t.value for t in top5]
        top5_std = np.std(top5_vals) if len(top5_vals) > 1 else 0
        top5_range = max(top5_vals) - min(top5_vals) if top5_vals else 0

        # Parameter consistency in top-5
        param_consistency: Dict[str, float] = {}
        param_names = list(top5[0].params.keys()) if top5 else []
        for pname in param_names:
            vals = [t.params[pname] for t in top5 if isinstance(t.params.get(pname), (int, float))]
            if len(vals) >= 2:
                cv = np.std(vals) / abs(np.mean(vals)) if np.mean(vals) != 0 else float("inf")
                param_consistency[pname] = float(cv)

        out["stability"] = {
            "top5_objective_std": float(top5_std),
            "top5_objective_range": float(top5_range),
            "top5_objectives": top5_vals,
            "param_consistency_cv": param_consistency,
            "is_stable": top5_std < abs(result.best_objective) * 0.05 if result.best_objective else True,
        }
    else:
        out["stability"] = {}

    # ── Diagnostics & recommendations ──
    diagnostics = _build_diagnostics(result, out, meta, completed, pruned, objectives)
    out["diagnostics"] = diagnostics

    return out


# ═══════════════════════════════════════════════════════════════════════
# Diagnostics Engine — institutional-grade issue detection
# ═══════════════════════════════════════════════════════════════════════

_SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "CAUTION": 2, "INFO": 3, "SUGGESTION": 4}


def _build_diagnostics(
    result: OptimizationResult,
    summary: Dict[str, Any],
    meta: Dict[str, Any],
    completed: list,
    pruned: list,
    objectives: list,
) -> list:
    """
    Run 15+ diagnostic checks and return sorted list of findings.

    Categories:
        1. CONVERGENCE quality (too fast, 1st-half dominance, stagnation)
        2. PARAMETER dominance & boundaries
        3. OVERFIT signals (param instability, trivial runtime)
        4. STUDY quality (failure rate, pruning, sample size)
        5. MISSING VALIDATION (walk-forward, OOS, costs)
    """
    diags = []

    def _add(severity: str, code: str, message: str, detail: str = ""):
        diags.append({
            "severity": severity,
            "code": code,
            "message": message,
            "detail": detail,
        })

    conv = summary.get("convergence", {})
    stab = summary.get("stability", {})
    ts = summary.get("trial_statistics", {})
    param_imp = summary.get("param_importance", {})
    param_ranges = summary.get("param_ranges", {})
    param_space = meta.get("param_space", {})

    # ─────────────────────────────────────────────────────────────
    # 1. CONVERGENCE QUALITY
    # ─────────────────────────────────────────────────────────────

    # 1a. Too-fast convergence — 100% gain in first half
    gain_ratio = conv.get("gain_ratio_first_half", 0)
    if gain_ratio >= 0.95 and conv.get("total_improvements", 0) > 2:
        _add("CAUTION", "CONV_TOO_FAST",
             f"Convergence suspiciously fast — {_pct(gain_ratio)} of total gain "
             f"achieved in the first half of trials.",
             "This means the 2nd half of the study was pure exploration with no "
             "improvement. Possible causes: (1) parameter space is too small/easy, "
             "(2) objective function is very smooth with few local optima, "
             "(3) signal is trivially strong, or (4) overfitting is easy. "
             "Consider expanding the search space or adding regularization.")

    # 1b. Early stop with short study
    if result.early_stopped and result.n_evaluated < 50:
        _add("WARNING", "CONV_EARLY_FEW_TRIALS",
             f"Study early-stopped after only {result.n_evaluated} trials.",
             "Early convergence with few trials may indicate the search space is "
             "too constrained. The optimizer may not have explored enough of the "
             "space to find the true optimum.")

    # 1c. Early stop — informational
    elif result.early_stopped and conv.get("stagnation_tail", 0) > 20:
        _add("INFO", "CONV_EARLY_STOPPED",
             f"Study converged early after {result.n_evaluated} trials. "
             f"Stagnation tail: {conv.get('stagnation_tail')} trials.",
             "The parameter space appears well-explored.")

    # 1d. Zero 2nd-half gain
    if conv.get("second_half_gain") is not None and conv.get("total_gain", 0) > 0:
        second_ratio = conv.get("second_half_gain", 0) / conv.get("total_gain", 1)
        if second_ratio < 0.01 and len(completed) > 30:
            _add("CAUTION", "CONV_FLAT_SECOND_HALF",
                 f"Second half contributed only {_pct(second_ratio)} of total gain.",
                 "Bayesian sampler found the optimum quickly. If this is a realistic "
                 "backtest, good — the signal is clear. If in-sample only, suspect "
                 "overfitting to a smooth objective surface.")

    # 1e. Still improving at end — needs more trials
    if not result.early_stopped and result.n_evaluated >= meta.get("n_trials_requested", 0):
        if conv.get("late_improvements", 0) > 0:
            _add("SUGGESTION", "CONV_NEEDS_MORE",
                 "Still seeing improvements in later trials. "
                 "Consider increasing n_trials for better convergence.")

    # ─────────────────────────────────────────────────────────────
    # 2. PARAMETER DOMINANCE & BOUNDARIES
    # ─────────────────────────────────────────────────────────────

    # 2a. Single-parameter dominance (>50%)
    if param_imp:
        sorted_imp = sorted(param_imp.items(), key=lambda x: x[1], reverse=True)
        top_param, top_imp_val = sorted_imp[0]
        if top_imp_val > 0.50:
            _add("WARNING", "PARAM_DOMINANT",
                 f"Parameter '{top_param}' dominates at {_pct(top_imp_val)} importance.",
                 f"When a single parameter explains >50% of objective variance, the "
                 f"strategy is primarily driven by that one lever. Other parameters "
                 f"are doing marginal work. This may indicate the strategy is really "
                 f"a '{top_param}-tuned filter' rather than a multi-factor model. "
                 f"Not an error — but you must understand what you're actually trading.")

        # 2b. Low-importance params
        low_imp = [k for k, v in param_imp.items() if v < 0.05]
        if low_imp:
            _add("SUGGESTION", "PARAM_LOW_IMPORTANCE",
                 f"Parameters with negligible importance (<5%): "
                 f"{', '.join(low_imp)}.",
                 "Consider fixing these to reduce search space dimensionality "
                 "and improve optimization efficiency.")

    # 2c. Boundary-hitting — best param value is at edge of search space
    if param_space and param_ranges:
        for pname, spec in param_space.items():
            if spec.get("type") in ("int", "float"):
                best_val = result.best_params.get(pname)
                lo = spec.get("low")
                hi = spec.get("high")
                if best_val is not None and lo is not None and hi is not None:
                    rng = hi - lo
                    if rng > 0:
                        # Within 5% of boundary
                        margin = rng * 0.05
                        at_lower = best_val <= lo + margin
                        at_upper = best_val >= hi - margin
                        if at_upper:
                            _add("WARNING", "PARAM_BOUNDARY_UPPER",
                                 f"Best '{pname}' = {_fmt(best_val)} is at the UPPER "
                                 f"boundary of [{lo}, {hi}].",
                                 f"The optimum may lie beyond your search range. "
                                 f"Extend the upper bound (e.g., to {hi * 1.5:.1f}) "
                                 f"and re-run to check if objective improves further.")
                        elif at_lower:
                            _add("WARNING", "PARAM_BOUNDARY_LOWER",
                                 f"Best '{pname}' = {_fmt(best_val)} is at the LOWER "
                                 f"boundary of [{lo}, {hi}].",
                                 f"The optimum may lie below your search range. "
                                 f"Extend the lower bound and re-run.")

        # 2d. Top-10% region at boundary
        for pname, info in param_ranges.items():
            if info.get("type") != "numeric":
                continue
            spec = param_space.get(pname, {})
            lo = spec.get("low")
            hi = spec.get("high")
            if lo is None or hi is None:
                continue
            rng = hi - lo
            if rng <= 0:
                continue
            margin = rng * 0.10
            br_max = info.get("best_region_max", 0)
            br_min = info.get("best_region_min", 0)
            if br_max >= hi - margin and br_min >= hi - margin * 3:
                _add("CAUTION", "PARAM_REGION_BOUNDARY",
                     f"Top-10% region for '{pname}' clusters near upper boundary "
                     f"[{_fmt(br_min)} → {_fmt(br_max)}] vs space [{lo}, {hi}].",
                     "The optimal region is squeezed against the boundary. "
                     "Consider extending the range to allow proper exploration.")

    # ─────────────────────────────────────────────────────────────
    # 3. OVERFIT SIGNALS
    # ─────────────────────────────────────────────────────────────

    # 3a. Parameter instability in top-K (high CV)
    cv_map = stab.get("param_consistency_cv", {})
    unstable_params = [p for p, cv in cv_map.items() if cv > 0.30]
    if unstable_params:
        _add("CAUTION", "OVERFIT_PARAM_UNSTABLE",
             f"High parameter instability in top-5: "
             f"{', '.join(f'{p} (CV={cv_map[p]:.2f})' for p in unstable_params)}.",
             "When the 'best' region shows very different parameter values, the "
             "objective surface is likely flat or noisy around the optimum. "
             "Multiple very different configurations score similarly — a classic "
             "overfit symptom. The specific winning params may not generalise.")

    # 3b. Trivial trial duration — brute-force risk
    avg_trial = summary.get("overview", {}).get("avg_trial_time", 0)
    if avg_trial < 0.010 and len(completed) > 30:
        _add("CAUTION", "OVERFIT_TRIVIAL_RUNTIME",
             f"Average trial time is {_duration(avg_trial)} — extremely fast.",
             "When each trial takes <10ms, the evaluation function is trivially "
             "cheap. This raises brute-force overfit risk: the optimizer can try "
             "so many combinations that it nearly exhausts the space. "
             "Ensure this is a realistic backtest with proper costs/slippage, "
             "not a simplified proxy that's easy to game.")

    # 3c. Top-5 objective variance too low relative to overall spread
    if stab and ts:
        top5_range = stab.get("top5_objective_range", 0)
        overall_range = ts.get("max", 0) - ts.get("min", 0)
        if overall_range > 0 and top5_range / overall_range < 0.02 and len(completed) > 50:
            _add("INFO", "OVERFIT_NARROW_PEAK",
                 f"Top-5 spread ({_fmt(top5_range)}) is tiny relative to overall "
                 f"range ({_fmt(overall_range)}).",
                 "The optimal region is an extremely narrow peak. This can mean "
                 "a genuine sharp optimum, or fragile overfitting where a slightly "
                 "different dataset would shift the peak significantly.")

    # 3d. High top-5 stability but unstable params = red flag combo
    if stab.get("is_stable") and len(unstable_params) >= 2:
        _add("WARNING", "OVERFIT_STABLE_OBJ_UNSTABLE_PARAMS",
             "Objectives are stable but parameters are spread — "
             "flat objective surface detected.",
             "When very different parameter combinations produce nearly "
             "identical objectives, the strategy may not be sensitive to "
             "these parameters at all. This is an overfit plateau: "
             "any configuration 'works' in-sample, none may generalise "
             "to out-of-sample. Run walk-forward to confirm edge exists.")

    # ─────────────────────────────────────────────────────────────
    # 4. STUDY QUALITY
    # ─────────────────────────────────────────────────────────────

    # 4a. High failure rate
    if result.n_failed > result.n_evaluated * 0.2:
        _add("WARNING", "QUALITY_HIGH_FAILURES",
             f"High failure rate: {result.n_failed}/{result.n_evaluated} "
             f"trials failed ({_pct(result.n_failed / max(result.n_evaluated, 1))}).",
             "Check parameter ranges and evaluation function. Many failures "
             "waste computational budget and may bias the sampler.")

    # 4b. Aggressive pruning
    pruned_pct = len(pruned) / max(result.n_evaluated, 1)
    if pruned_pct > 0.5:
        _add("WARNING", "QUALITY_OVER_PRUNED",
             f"Over {_pct(pruned_pct)} of trials were pruned.",
             "Pruner may be too aggressive — consider relaxing threshold "
             "or switching from percentile to median pruner.")

    # 4c. Small sample for TPE
    if meta.get("sampler") == "tpe" and result.n_evaluated < 30:
        _add("INFO", "QUALITY_FEW_TRIALS_TPE",
             f"TPE sampler works best with 50+ trials. "
             f"Only {result.n_evaluated} completed.",
             "With few trials, TPE has insufficient data for Bayesian "
             "advantage over random search. Consider increasing n_trials.")

    # 4d. Small search space dimensionality warning
    n_params = len(result.best_params)
    if n_params >= 6 and result.n_evaluated < n_params * 20:
        _add("WARNING", "QUALITY_UNDERPOWERED",
             f"{n_params} parameters with only {result.n_evaluated} trials "
             f"(< {n_params * 20} recommended).",
             "High-dimensional spaces need more trials for reliable optimization. "
             f"Consider at least {n_params * 30} trials or reducing dimensions.")

    # ─────────────────────────────────────────────────────────────
    # 5. MISSING VALIDATION — the questions that MUST be asked
    # ─────────────────────────────────────────────────────────────

    # We can detect whether walk-forward was used by checking metadata
    has_wf = meta.get("walk_forward", False) or meta.get("is_walk_forward", False)
    has_oos = meta.get("out_of_sample", False) or meta.get("oos_validated", False)
    has_costs = meta.get("includes_costs", False)

    if not has_wf:
        _add("CRITICAL", "VALIDATION_NO_WALK_FORWARD",
             "No walk-forward validation detected.",
             "This optimization was run on a single in-sample period. "
             "Without walk-forward, the 'best' parameters are very likely "
             "overfit to historical noise. Results are NOT actionable for "
             "live trading. Run walk_forward_optimize() to get reliable estimates "
             "of out-of-sample performance.")

    if not has_oos and not has_wf:
        _add("CRITICAL", "VALIDATION_NO_OOS",
             "No out-of-sample validation detected.",
             "In-sample optimization without OOS holdout is the #1 cause of "
             "strategy failure in production. Reserve 20-30% of data for "
             "out-of-sample testing, or use walk-forward which inherently "
             "provides OOS estimates at each fold.")

    if not has_costs:
        _add("WARNING", "VALIDATION_NO_COSTS",
             "Cannot confirm that transaction costs are included.",
             "If the objective function does not include realistic transaction "
             "costs (commissions, spread, slippage), optimized parameters will "
             "overfit to gross returns. Strategies that look profitable before "
             "costs often disappear after. Ensure your backtester applies "
             "realistic cost models.")

    # ─────────────────────────────────────────────────────────────
    # 6. DISTRIBUTION QUALITY
    # ─────────────────────────────────────────────────────────────

    # 6a. Extremely low CV — all trials perform similarly
    cv_val = ts.get("coefficient_of_variation", 0)
    if cv_val < 0.05 and len(completed) > 20:
        _add("WARNING", "DIST_LOW_VARIANCE",
             f"Objective CV = {cv_val:.3f} — extremely low variance across trials.",
             "Nearly all parameter combinations produce similar results. "
             "This suggests the objective is insensitive to parameters "
             "(flat surface). The 'best' params may just be noise.")

    # 6b. Heavy positive skew — many bad trials, few good
    skew = ts.get("skewness", 0)
    if skew < -1.0:
        _add("INFO", "DIST_LEFT_SKEWED",
             f"Objective distribution is left-skewed (skew={skew:.2f}).",
             "Most trials cluster near high values with a tail of bad outcomes. "
             "The parameter space is generally favourable.")
    elif skew > 1.0:
        _add("CAUTION", "DIST_RIGHT_SKEWED",
             f"Objective distribution is right-skewed (skew={skew:.2f}).",
             "Most trials produce poor results with only a few good outcomes. "
             "The optimal region may be very narrow and fragile.")

    # Sort by severity
    diags.sort(key=lambda d: _SEVERITY_ORDER.get(d["severity"], 99))
    return diags


# ═══════════════════════════════════════════════════════════════════════
# Summary Text
# ═══════════════════════════════════════════════════════════════════════

def optimization_summary(
    result: OptimizationResult,
    *,
    verbose: bool = True,
    width: int = 72,
) -> str:
    """
    Generate a rich, detailed text report from an OptimizationResult.

    Args:
        result: OptimizationResult from any optimizer.
        verbose: Include all sections (False = compact executive summary).
        width: Line width for formatting.

    Returns:
        Multi-line formatted text string.
    """
    d = optimization_summary_dict(result)
    ov = d["overview"]
    cfg = d["configuration"]
    lines: List[str] = []

    # ── Header ──
    lines.append("")
    lines.append("╔" + "═" * (width - 2) + "╗")
    lines.append("║" + " OPTUNA OPTIMIZATION REPORT ".center(width - 2) + "║")
    lines.append("║" + " QuantJourney Institutional Analytics ".center(width - 2) + "║")
    lines.append("╚" + "═" * (width - 2) + "╝")

    # ── Executive Summary ──
    lines.append(_section("EXECUTIVE SUMMARY"))
    lines.append(f"  Best Objective:    {_fmt(ov['best_objective'])}")
    lines.append(f"  Total Trials:      {ov['total_trials']}"
                 f"  (✓ {ov['completed']}  ✂ {ov['pruned']}  ✗ {ov['failed']})")
    lines.append(f"  Wall Time:         {ov['elapsed_human']}"
                 f"  ({ov['trials_per_second']:.1f} trials/sec)")
    lines.append(f"  Avg Trial:         {_duration(ov['avg_trial_time'])}")
    if ov["early_stopped"]:
        lines.append(f"  Early Stopped:     Yes — {ov['early_stop_reason']}")
    lines.append("")

    # ── Best Params ──
    bt = d.get("best_trial")
    if bt:
        lines.append(f"  🏆 WINNING TRIAL #{bt['number']}:")
        for k, v in bt["params"].items():
            imp = d.get("param_importance", {}).get(k)
            imp_str = f"  (importance: {_pct(imp)})" if imp else ""
            if isinstance(v, float):
                lines.append(f"     {k:>24s} = {_fmt(v)}{imp_str}")
            else:
                lines.append(f"     {k:>24s} = {v}{imp_str}")
        if bt.get("metrics"):
            lines.append(f"\n  Metrics snapshot:")
            for mk, mv in bt["metrics"].items():
                lines.append(f"     {mk:>24s} = {_fmt(mv)}")

    if not verbose:
        return "\n".join(lines)

    # ── Configuration ──
    lines.append(_section("STUDY CONFIGURATION"))
    lines.append(f"  Sampler:           {cfg['sampler'].upper()}")
    if cfg["sampler"] == "tpe":
        lines.append(f"                     (multivariate=True, constant_liar=True)")
    lines.append(f"  Pruner:            {cfg['pruner'].upper()}")
    lines.append(f"  Direction:         {cfg['direction'].upper()}")
    if cfg["directions"]:
        lines.append(f"  Multi-objective:   {cfg['directions']}")
    lines.append(f"  N Trials:          {cfg['n_trials_requested']}"
                 f"  (completed: {ov['completed']})")
    lines.append(f"  N Jobs:            {cfg['n_jobs']}")
    if cfg["timeout"]:
        lines.append(f"  Timeout:           {_duration(cfg['timeout'])}")
    if cfg["patience"]:
        lines.append(f"  Patience:          {cfg['patience']} trials")
    lines.append(f"  Startup Trials:    {cfg['n_startup_trials']}")
    lines.append(f"  Seed:              {cfg['seed']}")
    if cfg["warm_start_count"]:
        lines.append(f"  Warm-start:        {cfg['warm_start_count']} prior trials")

    # ── Trial Statistics ──
    ts = d.get("trial_statistics", {})
    if ts:
        lines.append(_section("OBJECTIVE STATISTICS"))
        lines.append(f"  Count:             {ts['count']}")
        lines.append(f"  Mean ± Std:        {_fmt(ts['mean'])} ± {_fmt(ts['std'])}")
        lines.append(f"  Median:            {_fmt(ts['median'])}")
        lines.append(f"  Range:             [{_fmt(ts['min'])}, {_fmt(ts['max'])}]")
        lines.append(f"  IQR (Q25–Q75):     [{_fmt(ts['q25'])}, {_fmt(ts['q75'])}]")
        lines.append(f"  Q90:               {_fmt(ts['q90'])}")
        lines.append(f"  Q95:               {_fmt(ts['q95'])}")
        lines.append(f"  CV:                {_fmt(ts['coefficient_of_variation'], 3)}")
        lines.append(f"  Skewness:          {_fmt(ts['skewness'], 3)}")
        lines.append(f"  Kurtosis:          {_fmt(ts['kurtosis'], 3)}")

        # ASCII Distribution
        lines.append(_subsection("Distribution"))
        _add_ascii_histogram(lines, result)

    # ── Convergence ──
    conv = d.get("convergence", {})
    if conv:
        lines.append(_section("CONVERGENCE ANALYSIS"))
        lines.append(f"  Total Improvements:       {conv['total_improvements']}")
        lines.append(f"  Improvement Rate:         {_pct(conv['improvement_rate'])}")
        lines.append(f"  Last Improvement At:      Trial #{conv['last_improvement_at_trial']}")
        lines.append(f"  Stagnation Tail:          {conv['stagnation_tail']} trials")
        lines.append(f"  Total Gain:               {_fmt(conv['total_gain'])}")
        lines.append(f"  1st Half Gain:            {_fmt(conv['first_half_gain'])}"
                     f"  ({_pct(conv['gain_ratio_first_half'])} of total)")
        lines.append(f"  2nd Half Gain:            {_fmt(conv['second_half_gain'])}")
        lines.append(f"  Early Improvements:       {conv['early_improvements']}")
        lines.append(f"  Late Improvements:        {conv['late_improvements']}")
        if conv.get("fully_converged"):
            lines.append(f"  ✓ Study appears fully converged")
        else:
            lines.append(f"  → Study may benefit from additional trials")

        # ASCII convergence curve
        lines.append(_subsection("Convergence Curve"))
        _add_ascii_convergence(lines, result)

    # ── Parameter Importance ──
    imp = d.get("param_importance", {})
    if imp:
        lines.append(_section("PARAMETER IMPORTANCE"))
        max_imp = max(imp.values()) if imp else 1
        for name, val in imp.items():
            bar = _bar(val, max_imp, 25)
            lines.append(f"  {name:>24s}  {bar}  {_pct(val)}")

    # ── Parameter Ranges ──
    ranges = d.get("param_ranges", {})
    if ranges:
        lines.append(_section("PARAMETER ANALYSIS"))
        for pname, info in ranges.items():
            if info["type"] == "numeric":
                lines.append(_subsection(pname))
                lines.append(f"    Searched:      [{_fmt(info['min'])} → {_fmt(info['max'])}]")
                lines.append(f"    Mean ± Std:    {_fmt(info['mean'])} ± {_fmt(info['std'])}")
                lines.append(f"    Best Region:   [{_fmt(info['best_region_min'])} → "
                             f"{_fmt(info['best_region_max'])}]  "
                             f"(mean: {_fmt(info['best_region_mean'])})")
                lines.append(f"    Winning Value: {_fmt(info['best_value'])}")
            else:
                lines.append(_subsection(f"{pname} (categorical)"))
                dist = info.get("distribution", {})
                top_dist = info.get("top_region_distribution", {})
                total = sum(dist.values()) or 1
                for cat, cnt in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    top_cnt = top_dist.get(cat, 0)
                    bar = _bar(cnt, total, 20)
                    lines.append(f"    {str(cat):>20s}  {bar}  {cnt}x"
                                 f"  (top-10%: {top_cnt}x)")
                lines.append(f"    Winning Value: {info['best_value']}")

    # ── Top 10 Trials ──
    top = d.get("top_trials", [])
    if top:
        lines.append(_section("TOP 10 TRIALS"))
        # Header
        lines.append(f"  {'Rank':>4}  {'Trial':>5}  {'Objective':>12}  "
                     f"{'Duration':>10}  Parameters")
        lines.append(f"  {'─' * 4}  {'─' * 5}  {'─' * 12}  {'─' * 10}  {'─' * 30}")
        for t in top:
            params_str = ", ".join(
                f"{k}={_fmt(v) if isinstance(v, float) else v}"
                for k, v in t["params"].items()
            )
            lines.append(f"  {t['rank']:>4}  #{t['number']:>4}  "
                         f"{_fmt(t['objective']):>12}  "
                         f"{_duration(t['duration']):>10}  {params_str}")

    # ── Stability ──
    stab = d.get("stability", {})
    if stab:
        lines.append(_section("STABILITY ANALYSIS"))
        lines.append(f"  Top-5 Obj Std:     {_fmt(stab['top5_objective_std'])}")
        lines.append(f"  Top-5 Obj Range:   {_fmt(stab['top5_objective_range'])}")
        lines.append(f"  Top-5 Objectives:  {[round(v, 4) for v in stab['top5_objectives']]}")
        if stab.get("is_stable"):
            lines.append(f"  ✓ Optimal region is STABLE — results are reliable")
        else:
            lines.append(f"  ⚠ Optimal region shows HIGH VARIANCE — results may be noisy")

        cv = stab.get("param_consistency_cv", {})
        if cv:
            lines.append(_subsection("Parameter Consistency (CV in Top-5)"))
            for pname, cv_val in sorted(cv.items(), key=lambda x: x[1]):
                status = "✓ tight" if cv_val < 0.1 else "~ moderate" if cv_val < 0.3 else "⚠ spread"
                lines.append(f"    {pname:>24s}  CV = {cv_val:.3f}  {status}")

    # ── Diagnostics ──
    diags = d.get("diagnostics", [])
    if diags:
        lines.append(_section("DIAGNOSTICS & RECOMMENDATIONS"))

        # Group by severity
        severity_icons = {
            "CRITICAL": "🔴", "WARNING": "🟡", "CAUTION": "🟠",
            "INFO": "ℹ️ ", "SUGGESTION": "💡",
        }
        # Count by severity for header
        sev_counts = Counter(d["severity"] for d in diags)
        sev_summary = "  ".join(
            f"{severity_icons.get(s, '•')} {s}: {c}"
            for s, c in sorted(sev_counts.items(),
                                key=lambda x: _SEVERITY_ORDER.get(x[0], 99))
        )
        lines.append(f"  {sev_summary}")
        lines.append("")

        for diag in diags:
            icon = severity_icons.get(diag["severity"], "•")
            code = diag.get("code", "")
            code_str = f" [{code}]" if code else ""
            lines.append(f"  {icon} [{diag['severity']}]{code_str}")
            lines.append(f"     {diag['message']}")
            if diag.get("detail"):
                # Wrap detail text at ~64 chars
                detail = diag["detail"]
                words = detail.split()
                line = "     "
                for w in words:
                    if len(line) + len(w) + 1 > 68:
                        lines.append(line)
                        line = "     " + w
                    else:
                        line += (" " if len(line) > 5 else "") + w
                if line.strip():
                    lines.append(line)
            lines.append("")

    # ── Footer ──
    lines.append("")
    lines.append("─" * width)
    lines.append(f"  Generated by QuantJourney Optimization Engine")
    lines.append(f"  Sampler: {cfg['sampler']} | Pruner: {cfg['pruner']} | "
                 f"Trials: {ov['completed']}/{cfg['n_trials_requested']} | "
                 f"Time: {ov['elapsed_human']}")
    lines.append("─" * width)
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# ASCII Visualizations
# ═══════════════════════════════════════════════════════════════════════

def _add_ascii_histogram(lines: List[str], result: OptimizationResult,
                         width: int = 50, height: int = 12):
    """Add ASCII histogram of objective values."""
    completed = [t for t in result.trials if t.state == "COMPLETE"]
    if len(completed) < 3:
        return

    values = [t.value for t in completed]
    n_bins = min(width, max(8, len(values) // 3))

    hist, edges = np.histogram(values, bins=n_bins)
    max_count = max(hist)
    if max_count == 0:
        return

    for row in range(height, 0, -1):
        threshold = max_count * row / height
        line = "  │"
        for count in hist:
            if count >= threshold:
                line += "█"
            elif count >= threshold - max_count / height / 2:
                line += "▄"
            else:
                line += " "
        lines.append(line)

    lines.append("  └" + "─" * n_bins)
    lines.append(f"   {_fmt(edges[0]):>8s}{' ' * max(0, n_bins - 20)}{_fmt(edges[-1]):>8s}")


def _add_ascii_convergence(lines: List[str], result: OptimizationResult,
                           width: int = 55, height: int = 10):
    """Add ASCII convergence curve."""
    curve = result.convergence_curve
    if not curve or len(curve) < 2:
        return

    # Downsample if too many points
    if len(curve) > width:
        step = len(curve) / width
        sampled = [curve[int(i * step)] for i in range(width)]
    else:
        sampled = curve

    lo = min(sampled)
    hi = max(sampled)
    rng = hi - lo or 1

    for row in range(height, 0, -1):
        threshold = lo + rng * row / height
        line = "  │"
        for val in sampled:
            if val >= threshold:
                line += "█"
            elif val >= threshold - rng / height / 2:
                line += "▄"
            else:
                line += " "
        lines.append(line)

    lines.append("  └" + "─" * len(sampled))
    lines.append(f"   Trial 0{' ' * max(0, len(sampled) - 16)}Trial {len(curve) - 1}")
    lines.append(f"   {_fmt(lo)} → {_fmt(hi)}  (Δ = {_fmt(hi - lo)})")


# ═══════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════

def _skewness(arr: np.ndarray) -> float:
    """Compute skewness."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(arr)
    if n < 4:
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3)
