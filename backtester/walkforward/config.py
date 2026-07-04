"""
Walk-Forward Configuration — frozen, validated config for WF engine.

Uses a plain frozen dataclass with __post_init__ validation (no Pydantic
dependency required). Serializable via to_dict() for fingerprinting.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Literal, Optional, Union


@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Complete configuration for a walk-forward validation run.

    All fields have sensible defaults — only ``scheme``, ``train_months``,
    and ``test_months`` are typically customised by the user.
    """

    # ── Fold Geometry ─────────────────────────────────────────────────
    scheme: Literal["rolling", "expanding", "anchored", "cpcv"] = "rolling"
    train_months: int = 24
    test_months: int = 6
    min_train_months: int = 12
    step_months: Optional[int] = None  # default = test_months
    n_splits: Optional[int] = None     # CPCV only

    # ── Purging & Embargo ─────────────────────────────────────────────
    purge_days: int = 5
    embargo_pct: float = 0.01
    max_holding_period_days: Optional[int] = None

    # ── Optimization ──────────────────────────────────────────────────
    optimization: Optional[Dict[str, Any]] = None

    # ── Statistical Controls ──────────────────────────────────────────
    compute_deflated_sharpe: bool = True
    compute_pbo: bool = True
    pbo_n_partitions: int = 16
    min_oos_sharpe: float = 0.0

    # ── Cost Sensitivity ──────────────────────────────────────────────
    cost_sensitivity_bps: List[int] = field(
        default_factory=lambda: [0, 5, 10, 20]
    )
    base_slippage_model: Any = None  # SlippageModel instance
    base_commission_scheme: Any = None  # CommissionScheme instance

    # ── Execution ─────────────────────────────────────────────────────
    max_workers: int = 1
    n_jobs_optimizer: int = -1
    verbose: bool = True
    seed: int = 42

    # ── Derived ───────────────────────────────────────────────────────

    @property
    def effective_step_months(self) -> int:
        """Step between fold starts; defaults to test_months (non-overlapping OOS)."""
        return self.step_months if self.step_months is not None else self.test_months

    def __post_init__(self) -> None:
        if self.train_months < 1:
            raise ValueError("train_months must be >= 1")
        if self.test_months < 1:
            raise ValueError("test_months must be >= 1")
        if self.min_train_months < 1:
            raise ValueError("min_train_months must be >= 1")
        if self.purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        if not (0.0 <= self.embargo_pct <= 1.0):
            raise ValueError("embargo_pct must be in [0, 1]")
        if self.scheme == "cpcv" and (self.n_splits is None or self.n_splits < 2):
            raise ValueError("CPCV scheme requires n_splits >= 2")

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict (strip non-serialisable objects)."""
        d = asdict(self)
        # SlippageModel / CommissionScheme are not JSON-safe
        d["base_slippage_model"] = (
            type(self.base_slippage_model).__name__
            if self.base_slippage_model is not None
            else None
        )
        d["base_commission_scheme"] = (
            type(self.base_commission_scheme).__name__
            if self.base_commission_scheme is not None
            else None
        )
        return d
