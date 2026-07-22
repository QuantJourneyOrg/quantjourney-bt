"""
Walk-Forward Configuration — frozen, validated config for WF engine.

Uses a plain frozen dataclass with __post_init__ validation (no Pydantic
dependency required). Serializable via to_dict() for fingerprinting.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


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
    step_months: int | None = None  # default = test_months
    n_splits: int | None = None  # CPCV only

    # ── Pre-OOS Purging ───────────────────────────────────────────────
    purge_days: int = 5
    # Percentage-based extension of the exclusion immediately before OOS.
    # This is not a classical post-test embargo across later train folds.
    extra_pre_oos_purge_pct: float | None = None
    # Deprecated compatibility alias for extra_pre_oos_purge_pct.
    embargo_pct: float = 0.01
    max_holding_period_days: int | None = None

    # ── Optimization ──────────────────────────────────────────────────
    optimization: dict[str, Any] | None = None

    # ── Statistical Controls ──────────────────────────────────────────
    compute_deflated_sharpe: bool = True
    # Optional effective number of independent trials for DSR. When
    # unset, the aggregate uses the raw number of finite completed
    # optimizer trials as a conservative approximation.
    dsr_effective_n_trials: float | None = None

    # Rolling top-K OOS rank-stability diagnostic. This is not canonical
    # CSCV PBO. The pbo_* fields below are retained as compatibility
    # aliases for 0.12.x.
    compute_rank_stability: bool | None = None
    rank_stability_trials: int | None = None
    compute_pbo: bool = True
    pbo_n_partitions: int = 16
    # Deprecated alias for rank_stability_trials. 0 disables the extra
    # backtests when the new field is unset.
    pbo_trials: int = 0
    min_oos_sharpe: float = 0.0

    # ── Cost Sensitivity ──────────────────────────────────────────────
    cost_sensitivity_bps: list[int] = field(default_factory=lambda: [0, 5, 10, 20])
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

    @property
    def rank_stability_enabled(self) -> bool:
        """Resolved enable flag, honoring the legacy ``compute_pbo`` alias."""
        if self.compute_rank_stability is not None:
            return self.compute_rank_stability
        return self.compute_pbo

    @property
    def resolved_rank_stability_trials(self) -> int:
        """Resolved top-K size, honoring the legacy ``pbo_trials`` alias."""
        if self.rank_stability_trials is not None:
            return self.rank_stability_trials
        return self.pbo_trials

    @property
    def resolved_extra_pre_oos_purge_pct(self) -> float:
        """Resolved pre-OOS percentage, honoring legacy ``embargo_pct``."""
        if self.extra_pre_oos_purge_pct is not None:
            return self.extra_pre_oos_purge_pct
        return self.embargo_pct

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
        if self.extra_pre_oos_purge_pct is not None and not (
            0.0 <= self.extra_pre_oos_purge_pct <= 1.0
        ):
            raise ValueError("extra_pre_oos_purge_pct must be in [0, 1]")
        if self.dsr_effective_n_trials is not None and (
            not math.isfinite(self.dsr_effective_n_trials) or self.dsr_effective_n_trials < 1.0
        ):
            raise ValueError("dsr_effective_n_trials must be finite and >= 1")
        if self.rank_stability_trials is not None and (
            self.rank_stability_trials < 0 or self.rank_stability_trials == 1
        ):
            raise ValueError("rank_stability_trials must be 0 (disabled) or >= 2")
        if self.pbo_trials < 0 or self.pbo_trials == 1:
            raise ValueError("pbo_trials must be 0 (disabled) or >= 2")
        if (
            self.rank_stability_trials is not None
            and self.pbo_trials != 0
            and self.rank_stability_trials != self.pbo_trials
        ):
            raise ValueError(
                "rank_stability_trials and deprecated pbo_trials disagree; set only one"
            )
        if self.scheme == "cpcv":
            # Fail at config time instead of deep inside fold generation.
            raise NotImplementedError(
                "scheme='cpcv' is not implemented yet; use rolling, expanding or anchored"
            )

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
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
        d["rank_stability_enabled"] = self.rank_stability_enabled
        d["resolved_rank_stability_trials"] = self.resolved_rank_stability_trials
        d["resolved_extra_pre_oos_purge_pct"] = self.resolved_extra_pre_oos_purge_pct
        return d
