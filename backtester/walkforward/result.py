"""
Walk-Forward Result Data Contracts.

``FoldResult`` — per-fold IS/OOS metrics + diagnostics.
``WalkForwardResult`` — aggregate across all folds.

These are immutable data objects. All computation lives in
``runner.py``, ``engine.py``, and the ``statistics/`` subpackage.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backtester.walkforward.folds.base import Fold

# ── Per-Fold Result ───────────────────────────────────────────────────


@dataclass(frozen=True)
class FoldResult:
    """Immutable result for a single walk-forward fold."""

    fold: Fold  # fold geometry (train/oos boundaries, purge)

    # IS metrics
    is_sharpe: float
    is_cagr: float
    is_max_dd: float
    is_volatility: float
    is_n_trades: int
    is_win_rate: float
    is_profit_factor: float
    is_avg_holding_days: float
    is_turnover_ann: float

    # OOS metrics
    oos_sharpe: float
    oos_cagr: float
    oos_max_dd: float
    oos_volatility: float
    oos_n_trades: int
    oos_win_rate: float
    oos_profit_factor: float
    oos_avg_holding_days: float
    oos_turnover_ann: float

    # OOS data
    oos_returns: pd.Series  # daily OOS returns
    oos_nav: pd.Series  # OOS NAV (rebased to 1.0)

    # Diagnostics
    overfit_ratio: float  # IS Sharpe / OOS Sharpe
    efficiency: float  # OOS CAGR / IS CAGR
    sanity_warnings: list[str] = field(default_factory=list)
    fingerprint: str = ""

    # "ok" or "failed". A failed fold (empty NAV window, refit crash)
    # carries NaN metrics — never silent zeros — and is excluded from
    # engine aggregates.
    fold_status: str = "ok"

    # Optimization (None when no optimizer is used)
    best_params: dict[str, Any] | None = None
    optimizer_n_evals: int | None = None
    optimizer_best_objective: float | None = None
    # Completed-trial objective values (IS, annualized Sharpe) — the trial
    # population used for DSR's E[max SR] deflation.
    optimizer_trial_values: list[float] | None = None

    # Rolling top-K rank stability (not canonical CSCV PBO): trials'
    # OOS objective values and the IS-selected trial's OOS rank logit.
    rank_stability_candidate_oos: list[float] | None = None
    rank_stability_selected_logit: float | None = None

    # Cost sensitivity (optional)
    cost_sensitivity: dict[int, dict[str, float]] | None = None

    @property
    def pbo_candidate_oos(self) -> list[float] | None:
        """Deprecated compatibility alias for rank-stability candidates."""
        return self.rank_stability_candidate_oos

    @property
    def pbo_selected_logit(self) -> float | None:
        """Deprecated compatibility alias for the rank-stability logit."""
        return self.rank_stability_selected_logit


# ── Aggregate Result ──────────────────────────────────────────────────


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward result across all folds."""

    # ── Per-fold ──
    folds: list[FoldResult]
    config_dict: dict[str, Any]  # frozen copy of WalkForwardConfig.to_dict()

    # ── Aggregate OOS ──
    oos_sharpe: float = 0.0
    oos_cagr: float = 0.0
    oos_max_dd: float = 0.0
    oos_returns: pd.Series | None = None
    oos_nav: pd.Series | None = None
    # Stationary-block-bootstrap CI for the composite Sharpe
    # (seeded from WalkForwardConfig.seed; see statistics.aggregation).
    sharpe_ci_5pct: float | None = None
    sharpe_ci_95pct: float | None = None

    # ── Overfitting diagnostics ──
    overfit_ratio: float = 0.0
    efficiency: float = 0.0
    sharpe_decay: float = 0.0
    deflated_sharpe: float | None = None  # probability in [0, 1]
    deflated_sharpe_reason: str | None = None  # why DSR is unavailable (when None)
    # ``pooled_walk_forward_dsr_style`` is an explicitly labelled extension:
    # its trial objectives come from multiple chronological training folds,
    # not one canonical common trial population. With N=1 the statistic is PSR.
    deflated_sharpe_method: str | None = None
    dsr_raw_completed_trials: int | None = None
    dsr_effective_trials: float | None = None
    walk_forward_top_k_rank_failure_rate: float | None = None
    rank_stability_available: bool = False
    rank_stability_reason: str | None = None

    # ── Parameter stability ──
    param_stability: dict[str, float] | None = None
    param_trajectory: pd.DataFrame | None = None
    param_jaccard: float | None = None

    # ── Cost sensitivity ──
    cost_sensitivity: pd.DataFrame | None = None

    # ── Meta ──
    fingerprint: str = ""
    warnings: list[str] = field(default_factory=list)
    mode: str = "slice_diagnostics"

    # ── Derived properties ──

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def pbo(self) -> float | None:
        """Deprecated 0.12.x alias; this value is not canonical CSCV PBO."""
        return self.walk_forward_top_k_rank_failure_rate

    @property
    def pbo_available(self) -> bool:
        """Deprecated compatibility alias for ``rank_stability_available``."""
        return self.rank_stability_available

    @property
    def pbo_reason(self) -> str | None:
        """Deprecated compatibility alias for ``rank_stability_reason``."""
        return self.rank_stability_reason

    @property
    def fold_boundaries(self) -> pd.DataFrame:
        """DataFrame with fold_id, train_start, train_end, oos_start, oos_end."""
        records = []
        for fr in self.folds:
            records.append(
                {
                    "fold_id": fr.fold.fold_id,
                    "train_start": fr.fold.train_start.strftime("%Y-%m-%d"),
                    "train_end": fr.fold.train_end.strftime("%Y-%m-%d"),
                    "oos_start": fr.fold.oos_start.strftime("%Y-%m-%d"),
                    "oos_end": fr.fold.oos_end.strftime("%Y-%m-%d"),
                    "is_sharpe": fr.is_sharpe,
                    "oos_sharpe": fr.oos_sharpe,
                }
            )
        return pd.DataFrame(records)

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (for archival / fingerprinting)."""
        d: dict[str, Any] = {
            "n_folds": self.n_folds,
            "oos_sharpe": self.oos_sharpe,
            "oos_cagr": self.oos_cagr,
            "oos_max_dd": self.oos_max_dd,
            "sharpe_ci_5pct": self.sharpe_ci_5pct,
            "sharpe_ci_95pct": self.sharpe_ci_95pct,
            "overfit_ratio": self.overfit_ratio,
            "efficiency": self.efficiency,
            "sharpe_decay": self.sharpe_decay,
            "deflated_sharpe": self.deflated_sharpe,
            "deflated_sharpe_reason": self.deflated_sharpe_reason,
            "deflated_sharpe_method": self.deflated_sharpe_method,
            "dsr_raw_completed_trials": self.dsr_raw_completed_trials,
            "dsr_effective_trials": self.dsr_effective_trials,
            "walk_forward_top_k_rank_failure_rate": (self.walk_forward_top_k_rank_failure_rate),
            "rank_stability_available": self.rank_stability_available,
            "rank_stability_reason": self.rank_stability_reason,
            # Deprecated compatibility keys for 0.12.x.
            "pbo": self.pbo,
            "pbo_available": self.pbo_available,
            "pbo_reason": self.pbo_reason,
            "fingerprint": self.fingerprint,
            "mode": self.mode,
            "warnings": self.warnings,
            "config": self.config_dict,
        }
        # Per-fold summary (no heavy Series)
        fold_summaries = []
        for fr in self.folds:
            fold_summaries.append(
                {
                    "fold_id": fr.fold.fold_id,
                    "scheme": fr.fold.scheme,
                    "train_start": str(fr.fold.train_start.date()),
                    "train_end": str(fr.fold.train_end.date()),
                    "oos_start": str(fr.fold.oos_start.date()),
                    "oos_end": str(fr.fold.oos_end.date()),
                    "is_sharpe": fr.is_sharpe,
                    "oos_sharpe": fr.oos_sharpe,
                    "is_cagr": fr.is_cagr,
                    "oos_cagr": fr.oos_cagr,
                    "overfit_ratio": fr.overfit_ratio,
                    "efficiency": fr.efficiency,
                    "best_params": fr.best_params,
                    "fold_status": fr.fold_status,
                }
            )
        d["folds"] = fold_summaries

        if self.cost_sensitivity is not None:
            d["cost_sensitivity"] = self.cost_sensitivity.to_dict(orient="records")

        return d

    # ── Display ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Rich-ready summary string for console output."""
        lines = []
        scheme = self.config_dict.get("scheme", "?")
        train_m = self.config_dict.get("train_months", "?")
        test_m = self.config_dict.get("test_months", "?")
        slice_mode = self.mode == "slice_diagnostics"
        failed = [fr for fr in self.folds if fr.fold_status != "ok"]

        if slice_mode:
            lines.append("=" * 80)
            lines.append("IN-SAMPLE SLICE DIAGNOSTICS (not out-of-sample)")
            lines.append(
                "All metrics below are slices of ONE full-period run — NOT out-of-sample evidence."
            )
            lines.append(
                "Pass backtester_factory to WalkForwardEngine for honest per-fold refit OOS."
            )
            lines.append("=" * 80)
        lines.append(
            f"Walk-Forward Analysis — {self.n_folds} folds ({scheme}, {train_m}m/{test_m}m)"
        )
        lines.append(f"Fingerprint: {self.fingerprint[:12]}")
        lines.append(f"Mode: {self.mode}")
        if failed:
            failed_ids = ", ".join(str(fr.fold.fold_id) for fr in failed)
            lines.append(
                f"⚠ {len(failed)}/{self.n_folds} FOLDS FAILED "
                f"(ids: {failed_ids}) — excluded from aggregates"
            )
        lines.append("")

        # Per-fold table
        lines.append(
            f"{'Fold':>4} │ {'IS Period':<23} │ {'IS Sharpe':>9} │ "
            f"{'OOS Period':<23} │ {'OOS Sharpe':>10}"
        )
        lines.append("─" * 80)

        for fr in self.folds:
            is_period = (
                f"{fr.fold.train_start.strftime('%Y-%m')} → {fr.fold.train_end.strftime('%Y-%m')}"
            )
            oos_period = (
                f"{fr.fold.oos_start.strftime('%Y-%m')} → {fr.fold.oos_end.strftime('%Y-%m')}"
            )
            failed_tag = "  ← FAILED" if fr.fold_status != "ok" else ""
            lines.append(
                f"{fr.fold.fold_id:>4} │ {is_period:<23} │ "
                f"{fr.is_sharpe:>9.2f} │ {oos_period:<23} │ "
                f"{fr.oos_sharpe:>10.2f}{failed_tag}"
            )

        lines.append("─" * 80)
        if slice_mode:
            lines.append("AGGREGATE (IN-SAMPLE SLICES — not OOS)")
            sr_label = "Composite Sharpe (IS slice):"
            cagr_label = "CAGR (IS slice):    "
            dd_label = "Max DD (IS slice):  "
        else:
            lines.append("AGGREGATE OOS")
            sr_label = "Composite OOS Sharpe:       "
            cagr_label = "OOS CAGR:           "
            dd_label = "OOS Max DD:         "
        ci_str = ""
        if self.sharpe_ci_5pct is not None and self.sharpe_ci_95pct is not None:
            ci_str = f" [5%: {self.sharpe_ci_5pct:.2f}, 95%: {self.sharpe_ci_95pct:.2f}]"
        lines.append(f"  {sr_label} {self.oos_sharpe:.2f}{ci_str}")
        lines.append(
            f"  {cagr_label}  {self.oos_cagr:>7.1%}    Overfit Ratio: {self.overfit_ratio:.2f}"
        )
        lines.append(
            f"  {dd_label}  {self.oos_max_dd:>7.1%}    Efficiency:    {self.efficiency:.2f}"
        )
        lines.append(f"  Sharpe Decay:         {self.sharpe_decay:+.3f}/fold")

        if self.deflated_sharpe is not None:
            if self.deflated_sharpe_method == "pooled_walk_forward_dsr_style":
                dsr_label = "Pooled WF DSR-style"
            elif self.deflated_sharpe_method == "probabilistic_sharpe_n1":
                dsr_label = "Probabilistic Sharpe N=1"
            else:
                dsr_label = "Deflated Sharpe"
            lines.append(f"  {dsr_label} (prob):{self.deflated_sharpe:>7.2f}")
            if self.dsr_raw_completed_trials is not None:
                effective = (
                    self.dsr_effective_trials
                    if self.dsr_effective_trials is not None
                    else float(self.dsr_raw_completed_trials)
                )
                lines.append(
                    "    Trials: "
                    f"raw completed={self.dsr_raw_completed_trials}, "
                    f"effective N={effective:g}"
                )
        elif self.deflated_sharpe_reason:
            lines.append(f"  Deflated Sharpe:      n/a ({self.deflated_sharpe_reason})")
        rank_failure = self.walk_forward_top_k_rank_failure_rate
        rank_render = f"{rank_failure:.2f}" if rank_failure is not None else "n/a"
        lines.append(f"  WF top-K rank failure:{rank_render:>8}")
        if rank_failure is None and self.rank_stability_reason:
            lines.append(f"    (unavailable: {self.rank_stability_reason})")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)
