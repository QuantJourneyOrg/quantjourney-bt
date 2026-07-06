"""
Walk-Forward Result Data Contracts.

``FoldResult`` — per-fold IS/OOS metrics + diagnostics.
``WalkForwardResult`` — aggregate across all folds.

These are immutable data objects. All computation lives in
``runner.py``, ``engine.py``, and the ``statistics/`` subpackage.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
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
    oos_returns: pd.Series   # daily OOS returns
    oos_nav: pd.Series       # OOS NAV (rebased to 1.0)

    # Diagnostics
    overfit_ratio: float     # IS Sharpe / OOS Sharpe
    efficiency: float        # OOS CAGR / IS CAGR
    sanity_warnings: List[str] = field(default_factory=list)
    fingerprint: str = ""

    # Optimization (None when no optimizer is used)
    best_params: Optional[Dict[str, Any]] = None
    optimizer_n_evals: Optional[int] = None
    optimizer_best_objective: Optional[float] = None

    # Cost sensitivity (optional)
    cost_sensitivity: Optional[Dict[int, Dict[str, float]]] = None


# ── Aggregate Result ──────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Aggregate walk-forward result across all folds."""

    # ── Per-fold ──
    folds: List[FoldResult]
    config_dict: Dict[str, Any]  # frozen copy of WalkForwardConfig.to_dict()

    # ── Aggregate OOS ──
    oos_sharpe: float = 0.0
    oos_cagr: float = 0.0
    oos_max_dd: float = 0.0
    oos_returns: Optional[pd.Series] = None
    oos_nav: Optional[pd.Series] = None

    # ── Overfitting diagnostics ──
    overfit_ratio: float = 0.0
    efficiency: float = 0.0
    sharpe_decay: float = 0.0
    deflated_sharpe: Optional[float] = None
    pbo: Optional[float] = None

    # ── Parameter stability ──
    param_stability: Optional[Dict[str, float]] = None
    param_trajectory: Optional[pd.DataFrame] = None
    param_jaccard: Optional[float] = None

    # ── Cost sensitivity ──
    cost_sensitivity: Optional[pd.DataFrame] = None

    # ── Meta ──
    fingerprint: str = ""
    warnings: List[str] = field(default_factory=list)
    mode: str = "slice_diagnostics"

    # ── Derived properties ──

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def fold_boundaries(self) -> pd.DataFrame:
        """DataFrame with fold_id, train_start, train_end, oos_start, oos_end."""
        records = []
        for fr in self.folds:
            records.append({
                "fold_id": fr.fold.fold_id,
                "train_start": fr.fold.train_start.strftime("%Y-%m-%d"),
                "train_end": fr.fold.train_end.strftime("%Y-%m-%d"),
                "oos_start": fr.fold.oos_start.strftime("%Y-%m-%d"),
                "oos_end": fr.fold.oos_end.strftime("%Y-%m-%d"),
                "is_sharpe": fr.is_sharpe,
                "oos_sharpe": fr.oos_sharpe,
            })
        return pd.DataFrame(records)

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict (for archival / fingerprinting)."""
        d: Dict[str, Any] = {
            "n_folds": self.n_folds,
            "oos_sharpe": self.oos_sharpe,
            "oos_cagr": self.oos_cagr,
            "oos_max_dd": self.oos_max_dd,
            "overfit_ratio": self.overfit_ratio,
            "efficiency": self.efficiency,
            "sharpe_decay": self.sharpe_decay,
            "deflated_sharpe": self.deflated_sharpe,
            "pbo": self.pbo,
            "fingerprint": self.fingerprint,
            "mode": self.mode,
            "warnings": self.warnings,
            "config": self.config_dict,
        }
        # Per-fold summary (no heavy Series)
        fold_summaries = []
        for fr in self.folds:
            fold_summaries.append({
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
            })
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

        lines.append(
            f"Walk-Forward Analysis — {self.n_folds} folds "
            f"({scheme}, {train_m}m/{test_m}m)"
        )
        lines.append(f"Fingerprint: {self.fingerprint[:12]}")
        lines.append(f"Mode: {self.mode}")
        lines.append("")

        # Per-fold table
        lines.append(
            f"{'Fold':>4} │ {'IS Period':<23} │ {'IS Sharpe':>9} │ "
            f"{'OOS Period':<23} │ {'OOS Sharpe':>10}"
        )
        lines.append("─" * 80)

        for fr in self.folds:
            is_period = (
                f"{fr.fold.train_start.strftime('%Y-%m')} → "
                f"{fr.fold.train_end.strftime('%Y-%m')}"
            )
            oos_period = (
                f"{fr.fold.oos_start.strftime('%Y-%m')} → "
                f"{fr.fold.oos_end.strftime('%Y-%m')}"
            )
            lines.append(
                f"{fr.fold.fold_id:>4} │ {is_period:<23} │ "
                f"{fr.is_sharpe:>9.2f} │ {oos_period:<23} │ "
                f"{fr.oos_sharpe:>10.2f}"
            )

        lines.append("─" * 80)
        lines.append("AGGREGATE OOS")
        lines.append(f"  Composite OOS Sharpe: {self.oos_sharpe:>8.2f}    "
                      f"Overfit Ratio: {self.overfit_ratio:.2f}")
        lines.append(f"  OOS CAGR:             {self.oos_cagr:>7.1%}    "
                      f"Efficiency:    {self.efficiency:.2f}")
        lines.append(f"  OOS Max DD:           {self.oos_max_dd:>7.1%}    "
                      f"Sharpe Decay:  {self.sharpe_decay:+.3f}/fold")

        if self.deflated_sharpe is not None:
            lines.append(f"  Deflated Sharpe:      {self.deflated_sharpe:>8.2f}    "
                          f"PBO:           {self.pbo:.2f}" if self.pbo is not None else "")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)
