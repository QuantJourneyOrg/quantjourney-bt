"""
    Conditional Rebalancing Engine  v2
    ───────────────────────────────────
    Composable rebalancing rules for weight-based strategies.

    Architecture
    ────────────
        RebalanceAt      — execution timing enum (OPEN / CLOSE / VWAP_WINDOW)
        RebalancePolicy  — declarative config (frozen dataclass)
        RebalanceEngine  — evaluates policy against market state
        RebalancePresets  — common institutional configurations

    Trigger composition (OR, then gated):
        rebalance = (Calendar OR Drift OR Tracking-Error
                     OR Signal-change OR Risk-breaker)
                    AND Cost-gate

    Default policy (frequency="D", rebalance_at=CLOSE) reproduces the
    legacy daily-rebalance behaviour exactly.

    Changes from v1
    ────────────────
    • FIX  circuit-breaker cooldown exit → force_rebal_next flag
    • FIX  turnover calc cached (was computed twice)
    • ADD  rebalance_at enum (OPEN / CLOSE / VWAP_WINDOW)
    • ADD  rolling 252-day turnover window (replaces naïve annual reset)
    • ADD  tracking_error_threshold trigger (L2b, alongside drift)
    • ADD  partial_rebalance implementation (only drifted positions trade)
    • ADD  avoid_short_term_gains flag (basic tax-lot awareness)
    • DOC  _signal_change_flags operates on target_weights, not raw signals

    See: _docs/ADR_REBALANCING.md for full design rationale.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import enum
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# RebalanceAt — execution timing enum
# ─────────────────────────────────────────────────────────────────────

class RebalanceAt(enum.Enum):
    """
    When within the bar to execute the rebalance.

    Downstream consumers (FillEngine / CostEngine) use this to select
    the fill price and corresponding slippage model:

        OPEN         — fill at next-bar open  (most common for daily)
        CLOSE        — fill at current-bar close
        VWAP_WINDOW  — fill at bar VWAP (or TWAP proxy)

    The RebalanceEngine itself is price-agnostic; it only records the
    *timing intent* on ``engine.rebalance_at`` so that the caller
    (``core._compute_performance_weight_based``) can route fills
    accordingly.
    """
    OPEN = "open"
    CLOSE = "close"
    VWAP_WINDOW = "vwap_window"


# ─────────────────────────────────────────────────────────────────────
# RebalancePolicy — declarative, frozen config
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RebalancePolicy:
    """
    Declarative rebalancing policy — passed to Backtester constructor.

    Layers
    ──────
        L1   Calendar schedule         — deterministic dates
        L2a  Drift band                — market-driven trigger
        L2b  Tracking-error trigger    — benchmark-relative drift
        L3   Signal-change trigger     — alpha-driven
        L4   Risk circuit breaker      — protective override
        L5   Turnover / cost gate      — rolling 252-day budget

    Execution timing
    ────────────────
        rebalance_at  — OPEN / CLOSE / VWAP_WINDOW
        Propagated to FillEngine so slippage matches execution intent.

    Tax awareness
    ─────────────
        avoid_short_term_gains — if True, the engine will skip selling
        positions held < 252 trading days when reducing weight,
        preferring to sell longer-held lots first (basic LIFO avoidance).
        Full tax-lot matching requires an external TaxLotTracker; this
        flag provides a simple first-order heuristic.
    """

    # ── Execution timing ──
    rebalance_at: RebalanceAt = RebalanceAt.CLOSE

    # ── L1: Calendar schedule ──
    frequency: Optional[str] = "D"
    #   "D"   = daily (current default)
    #   "W"   = weekly (every Friday, or custom weekday)
    #   "BME" = business month-end  (also accepts legacy "BM")
    #   "BQE" = business quarter-end (also accepts legacy "BQ")
    #   "BYE" = business year-end   (also accepts legacy "BA")
    #   "21D" = every 21 trading days
    #   None  = never calendar-rebalance (signal/drift only)
    weekday: int = 4  # 0=Mon..4=Fri.  Used when frequency="W"

    # ── L2a: Drift band ──
    drift_threshold: Optional[float] = None
    #   Max absolute weight drift before forced rebalance.
    drift_type: str = "absolute"  # "absolute" | "relative"

    # ── L2b: Tracking-error trigger ──
    tracking_error_threshold: Optional[float] = None
    #   Annualised tracking error vs benchmark.  Requires benchmark_returns
    #   to be passed to engine.run().  If benchmark_returns is None this
    #   trigger is silently skipped.
    tracking_error_window: int = 21  # rolling days for TE estimate

    # ── L3: Signal-change trigger ──
    rebalance_on_signal_change: bool = False
    signal_change_threshold: float = 0.0
    #   NOTE: signal change is detected on *target_weights* (post-
    #   normalisation), not on raw signals.  This means a change in
    #   the investable universe (instrument added/removed) will fire
    #   even if the underlying signal did not change.  This is
    #   intentional — universe changes require rebalancing.

    # ── L4: Risk circuit breaker ──
    max_drawdown_trigger: Optional[float] = None
    #   E.g. -0.15 means flatten at -15% drawdown.
    max_drawdown_action: str = "flatten"  # "flatten" | "halve"
    circuit_breaker_cooldown_days: int = 5

    # ── L5: Cost gate / turnover budget ──
    max_annual_turnover: Optional[float] = None
    #   Rolling 252-trading-day turnover budget (one-way).
    #   Replaces the naïve calendar-year reset.
    min_trade_size: float = 0.0

    # ── Partial rebalance ──
    partial_rebalance: bool = False
    #   If True, only positions whose individual drift exceeds
    #   ``drift_threshold`` (or tracking-error) are snapped to target.
    #   Untouched positions continue to drift.  Reduces turnover
    #   significantly for large portfolios (30+ positions).

    # ── Tax-lot awareness ──
    avoid_short_term_gains: bool = False
    #   Basic heuristic: when reducing a position, prefer to keep lots
    #   held < 252 days and sell longer-held lots first.  In a backtest
    #   this translates to "don't sell positions that were entered
    #   recently" by penalising weight reductions for young positions.
    short_term_days: int = 252  #   Holding-period threshold (trading days).

    def __repr__(self) -> str:
        parts = [f"freq={self.frequency}"]
        if self.rebalance_at != RebalanceAt.CLOSE:
            parts.append(f"at={self.rebalance_at.value}")
        if self.drift_threshold is not None:
            parts.append(f"drift={self.drift_threshold:.1%}")
        if self.tracking_error_threshold is not None:
            parts.append(f"te={self.tracking_error_threshold:.1%}")
        if self.rebalance_on_signal_change:
            parts.append("signal_change")
        if self.max_drawdown_trigger is not None:
            parts.append(f"dd_breaker={self.max_drawdown_trigger:.0%}")
        if self.max_annual_turnover is not None:
            parts.append(f"max_to={self.max_annual_turnover:.0f}x")
        if self.partial_rebalance:
            parts.append("partial")
        if self.avoid_short_term_gains:
            parts.append("tax_aware")
        return f"RebalancePolicy({', '.join(parts)})"


# ─────────────────────────────────────────────────────────────────────
# RebalancePresets — common configurations
# ─────────────────────────────────────────────────────────────────────

class RebalancePresets:
    """Pre-configured rebalancing policies."""

    DAILY = RebalancePolicy()

    WEEKLY = RebalancePolicy(frequency="W", weekday=4)

    MONTHLY = RebalancePolicy(frequency="BME")

    QUARTERLY = RebalancePolicy(frequency="BQE")

    YEARLY = RebalancePolicy(frequency="BYE")

    MONTHLY_WITH_DRIFT = RebalancePolicy(
        frequency="BME",
        drift_threshold=0.05,
    )

    MONTHLY_PARTIAL = RebalancePolicy(
        frequency="BME",
        drift_threshold=0.05,
        partial_rebalance=True,
    )

    RISK_MANAGED = RebalancePolicy(
        frequency="BME",
        drift_threshold=0.05,
        max_drawdown_trigger=-0.15,
        max_annual_turnover=4.0,
    )

    SIGNAL_DRIVEN = RebalancePolicy(
        frequency=None,
        rebalance_on_signal_change=True,
        signal_change_threshold=0.01,
    )

    TAX_AWARE_MONTHLY = RebalancePolicy(
        frequency="BME",
        drift_threshold=0.05,
        avoid_short_term_gains=True,
    )

    INSTITUTIONAL = RebalancePolicy(
        frequency="BME",
        rebalance_at=RebalanceAt.VWAP_WINDOW,
        drift_threshold=0.03,
        partial_rebalance=True,
        max_annual_turnover=6.0,
        avoid_short_term_gains=True,
    )


# ─────────────────────────────────────────────────────────────────────
# RebalanceEngine — evaluates policy, produces flags + drifted weights
# ─────────────────────────────────────────────────────────────────────

class RebalanceEngine:
    """
    Evaluates a RebalancePolicy against market state and produces:
      1. rebalance_flags  — pd.Series[bool], True on rebalance days
      2. actual_weights   — pd.DataFrame of realised (drifted) weights
      3. stats            — dict of rebalance statistics

    The engine runs a lightweight loop over bars:
      - On rebalance days: snap weights to target
        (or only drifted positions if partial_rebalance=True)
      - Between rebalance days: let weights drift with prices

    Performance: ~20 ms for 2 500 bars × 50 instruments (numpy inner loop).
    """

    def __init__(self, policy: RebalancePolicy):
        self.policy = policy
        self.stats: dict = {}
        self.rebalance_at: RebalanceAt = policy.rebalance_at

    # ── Public API ──────────────────────────────────────────────────

    def _is_simple_policy(self) -> bool:
        """Check if policy is simple enough for vectorized fast-path."""
        p = self.policy
        return (
            p.drift_threshold is None
            and p.tracking_error_threshold is None
            and not p.rebalance_on_signal_change
            and p.max_drawdown_trigger is None
            and p.max_annual_turnover is None
            and not p.partial_rebalance
            and not p.avoid_short_term_gains
        )

    def _run_vectorized(
        self,
        target_weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Vectorized fast-path for simple calendar-only policies.

        Instead of a Python for-loop over 2000+ bars, this segments the
        timeline into rebalance intervals and applies drift via cumprod
        within each segment — pure pandas/numpy, no per-bar iteration.

        ~10-20x faster than the loop path for simple policies.
        """
        dates = target_weights.index
        instruments = target_weights.columns
        n = len(dates)
        m = len(instruments)

        cal_flags = self._calendar_flags(dates)

        # First bar is always a rebalance
        cal_flags.iloc[0] = True
        rebal_mask = cal_flags.values  # bool array

        # Asset returns as numpy. Missing returns mark an instrument as
        # unavailable for that bar; weights are reassigned only across
        # available instruments before NaNs are converted to 0 for math.
        target_w, ret, available = self._prepare_target_and_returns(
            target_weights,
            asset_returns,
        )
        actual_w = np.empty((n, m), dtype=np.float64)
        return_w = np.empty((n, m), dtype=np.float64)

        # Identify rebalance indices
        rebal_indices = np.where(rebal_mask)[0]

        # Process each segment between consecutive rebalances.
        # carry_w = weights actually held INTO the segment-start bar: trades
        # execute at the close, so the rebalance-day PnL accrues on the
        # previous segment's final drifted weights (first segment: in cash
        # until the first rebalance close).
        carry_w = np.zeros(m, dtype=np.float64)
        for seg_idx in range(len(rebal_indices)):
            start = rebal_indices[seg_idx]
            end = rebal_indices[seg_idx + 1] if seg_idx + 1 < len(rebal_indices) else n

            # Snap to target on rebalance day (held from the close onward)
            w0 = target_w[start]
            return_w[start] = self._mask_unavailable_vector(carry_w, available[start])
            actual_w[start] = w0

            if end - start <= 1:
                carry_w = w0
                continue

            if not available[start + 1: end].all():
                current = w0.copy()
                for pos in range(start + 1, end):
                    period_w = self._mask_unavailable_vector(current, available[pos])
                    return_w[pos] = period_w
                    current = self._drift_with_cash(period_w, ret[pos])
                    actual_w[pos] = current
                carry_w = current
                continue

            # Drift within segment while preserving explicit cash as a zero-return sleeve.
            # Vectorized cumprod for the segment
            seg_ret = ret[start + 1: end]  # (seg_len, m)
            cum_growth = np.cumprod(1.0 + seg_ret, axis=0)  # (seg_len, m)

            # Drifted weights = initial_weight * cumulative_growth / (cash + asset values)
            drifted = w0[np.newaxis, :] * cum_growth  # (seg_len, m)
            cash = self._cash_sleeve(w0)
            denominators = cash + drifted.sum(axis=1, keepdims=True)
            denominators = np.where(np.abs(denominators) > 1e-12, denominators, 1.0)
            drifted /= denominators

            actual_w[start + 1: end] = drifted
            if len(drifted) > 0:
                return_w[start + 1: end] = np.vstack([w0, drifted[:-1]])
                carry_w = drifted[-1]

        actual_weights = pd.DataFrame(actual_w, index=dates, columns=instruments)
        self.return_weights = pd.DataFrame(return_w, index=dates, columns=instruments)
        rebal_flags = pd.Series(rebal_mask, index=dates, name="rebalance")

        rebal_count = int(rebal_mask.sum())
        self.stats = {
            "rebalance_count": rebal_count,
            "calendar_count": rebal_count,
            "drift_count": 0,
            "tracking_error_count": 0,
            "signal_count": 0,
            "circuit_breaker_count": 0,
            "partial_positions_saved": 0,
            "rolling_turnover_used": 0.0,
            "rebalance_pct": rebal_count / max(n, 1),
            "avg_days_between": n / max(rebal_count, 1),
            "fast_path": True,
        }

        return actual_weights, rebal_flags

    @staticmethod
    def _cash_sleeve(weights: np.ndarray) -> float:
        """Implicit cash/financing sleeve implied by net weights.

        For long/cash books this is unallocated cash. For levered or
        long/short books this may be negative cash/financing or short-sale
        proceeds. The identity is always ``cash + sum(weights) == 1``.
        """
        clean = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(clean.sum())
        return 1.0 - total

    @classmethod
    def _drift_with_cash(cls, weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Drift asset weights while preserving any explicit cash sleeve."""
        clean_weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        clean_returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        grown = clean_weights * (1.0 + clean_returns)
        denominator = cls._cash_sleeve(clean_weights) + float(grown.sum())
        if abs(denominator) <= 1e-12:
            return np.zeros_like(clean_weights)
        return grown / denominator

    @classmethod
    def _prepare_target_and_returns(
        cls,
        target_weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dates = target_weights.index
        instruments = target_weights.columns
        returns = asset_returns.reindex(index=dates, columns=instruments).astype(float)
        available = returns.notna().to_numpy()
        target = target_weights.to_numpy(dtype=float, copy=True)
        for i in range(len(target)):
            target[i] = cls._mask_unavailable_vector(target[i], available[i])
        ret = returns.fillna(0.0).to_numpy(dtype=float)
        return target, ret, available

    @staticmethod
    def _mask_unavailable_vector(weights: np.ndarray, available: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=True)
        if available.all():
            return clean
        original_gross = float(np.abs(clean).sum())
        clean[~available] = 0.0
        remaining_gross = float(np.abs(clean).sum())
        if original_gross <= 1e-12:
            return clean
        if remaining_gross <= 1e-12:
            return np.zeros_like(clean)
        return clean * (original_gross / remaining_gross)

    @staticmethod
    def _renormalize_traded_sleeve(
        new_w: np.ndarray,
        target_w: np.ndarray,
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        traded_mask = ~keep_mask
        if not traded_mask.any():
            return new_w

        target_traded = target_w[traded_mask]
        target_total = target_w.sum()
        kept_total = new_w[keep_mask].sum()
        if abs(target_total) > 1e-12 and abs(target_traded.sum()) > 1e-12:
            # Clamp the scale at 0: when kept (drifted) weights already sum
            # above the target total, the traded sleeve gets 0 — a negative
            # scale would flip the sign of the traded targets (long → short).
            scale = max((target_total - kept_total) / target_traded.sum(), 0.0)
            new_w[traded_mask] = target_traded * scale
            return new_w

        target_gross = np.abs(target_w).sum()
        kept_gross = np.abs(new_w[keep_mask]).sum()
        traded_gross = np.abs(target_traded).sum()
        if traded_gross > 1e-12:
            new_w[traded_mask] = target_traded * (
                max(target_gross - kept_gross, 0.0) / traded_gross
            )
        return new_w

    def run(
        self,
        target_weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Simulate weight drift and rebalancing.

        Parameters
        ----------
        target_weights : pd.DataFrame
            Target weights per day (already shifted for look-ahead).
        asset_returns : pd.DataFrame
            Daily returns of each asset.
        benchmark_returns : pd.Series, optional
            Daily returns of the benchmark.  Required only when
            ``tracking_error_threshold`` is set in the policy.

        Returns
        -------
        actual_weights : pd.DataFrame
            Realised weights (drifted between rebalances).
        rebal_flags : pd.Series[bool]
            True on days when a rebalance occurred.
        """
        # Fast-path for simple calendar-only policies
        if self._is_simple_policy():
            return self._run_vectorized(target_weights, asset_returns)

        dates = target_weights.index
        instruments = target_weights.columns
        n = len(dates)
        m = len(instruments)

        # Pre-compute calendar + signal flags (vectorised)
        cal_flags = self._calendar_flags(dates)
        sig_flags = self._signal_change_flags(target_weights)

        # Arrays for the simulation loop
        target_w, ret, available = self._prepare_target_and_returns(
            target_weights,
            asset_returns,
        )
        actual_w = np.zeros((n, m), dtype=np.float64)
        return_w = np.zeros((n, m), dtype=np.float64)
        rebal = np.zeros(n, dtype=bool)
        nav = np.ones(n, dtype=np.float64)

        # Benchmark returns for tracking-error trigger
        bench_ret: Optional[np.ndarray] = None
        if benchmark_returns is not None:
            bench_ret = benchmark_returns.reindex(dates).fillna(0.0).values

        current_w = np.zeros(m, dtype=np.float64)
        peak_nav = 1.0
        cb_active = False
        cb_cooldown_left = 0
        force_rebal_next = False    # ← FIX: set True when exiting cooldown

        # Rolling 252-day turnover window (replaces annual reset)
        _ROLLING_WINDOW = 252
        turnover_ring: deque[float] = deque(maxlen=_ROLLING_WINDOW)

        # Position-entry-bar tracking for tax-lot heuristic
        # entry_bar[j] = bar index when position j was last entered / increased
        entry_bar = np.full(m, -9999, dtype=np.int64)

        # Counters for stats
        cal_count = 0
        drift_count = 0
        te_count = 0
        sig_count = 0
        cb_count = 0
        partial_saved = 0   # positions skipped by partial rebalance

        for i in range(n):
            should_rebal = False
            trigger = None

            # ── L4: Circuit breaker ──
            # Skipped on the forced re-entry bar: NAV froze below the old
            # peak while flat, so the stale drawdown would re-trigger the
            # breaker forever and the portfolio would never re-enter.
            if (
                self.policy.max_drawdown_trigger is not None
                and not cb_active
                and not force_rebal_next
            ):
                if i > 0:
                    dd = (nav[i - 1] / peak_nav) - 1.0
                    if dd < self.policy.max_drawdown_trigger:
                        cb_active = True
                        cb_cooldown_left = self.policy.circuit_breaker_cooldown_days
                        cb_count += 1
                        prev_w = current_w.copy()
                        if self.policy.max_drawdown_action == "flatten":
                            current_w = np.zeros(m)
                        elif self.policy.max_drawdown_action == "halve":
                            current_w = current_w * 0.5
                        # Flatten/derisk executes at the close — the day's
                        # PnL is booked on the weights held into the bar.
                        return_w[i] = prev_w
                        actual_w[i] = current_w
                        rebal[i] = True
                        turnover_ring.append(float(np.abs(current_w - prev_w).sum()))
                        day_ret = (prev_w * ret[i]).sum()
                        nav[i] = nav[i - 1] * (1.0 + day_ret)
                        peak_nav = max(peak_nav, nav[i])
                        continue

            # Handle cooldown
            if cb_active:
                cb_cooldown_left -= 1
                if cb_cooldown_left <= 0:
                    cb_active = False
                    force_rebal_next = True   # ← FIX: force rebalance next bar
                return_w[i] = current_w
                actual_w[i] = current_w
                turnover_ring.append(0.0)
                day_ret = (current_w * ret[i]).sum()
                nav[i] = nav[i - 1] * (1.0 + day_ret) if i > 0 else 1.0
                peak_nav = max(peak_nav, nav[i])
                continue

            # ── Forced rebalance after cooldown exit ──
            if force_rebal_next:
                should_rebal = True
                trigger = "post_cooldown"
                cal_count += 1          # count as calendar-equivalent
                force_rebal_next = False
                # Reset the drawdown reference at re-entry so the breaker
                # only re-triggers on a NEW drawdown from the new peak.
                peak_nav = nav[i - 1] if i > 0 else 1.0

            # ── L1: Calendar ──
            if not should_rebal and cal_flags.iloc[i]:
                should_rebal = True
                trigger = "calendar"
                cal_count += 1

            # ── L3: Signal change ──
            if not should_rebal and sig_flags.iloc[i]:
                should_rebal = True
                trigger = "signal"
                sig_count += 1

            # ── L2a: Drift (in-loop, depends on current_w) ──
            if not should_rebal and self.policy.drift_threshold is not None and i > 0:
                tw = target_w[i]
                if self.policy.drift_type == "relative":
                    with np.errstate(divide='ignore', invalid='ignore'):
                        drift = np.where(
                            tw != 0,
                            np.abs(current_w - tw) / np.abs(tw),
                            np.abs(current_w),
                        )
                else:
                    drift = np.abs(current_w - tw)
                max_drift = drift.max()
                if max_drift > self.policy.drift_threshold:
                    should_rebal = True
                    trigger = "drift"
                    drift_count += 1

            # ── L2b: Tracking-error trigger ──
            if (
                not should_rebal
                and self.policy.tracking_error_threshold is not None
                and bench_ret is not None
                and i >= self.policy.tracking_error_window
            ):
                # Annualised TE over rolling window
                # return_w holds beginning-of-day weights — the same basis the
                # NAV accounting uses; actual_w already embeds day j's move.
                window = self.policy.tracking_error_window
                port_ret_window = np.array([
                    (return_w[j] * ret[j]).sum()
                    for j in range(i - window, i)
                ])
                bench_window = bench_ret[i - window: i]
                active_ret = port_ret_window - bench_window
                te_ann = float(np.std(active_ret, ddof=1)) * np.sqrt(252)
                if te_ann > self.policy.tracking_error_threshold:
                    should_rebal = True
                    trigger = "tracking_error"
                    te_count += 1

            # ── Apply rebalance or drift ──
            # Weights actually held INTO this bar: trades execute at the
            # close, so a rebalance-day's PnL accrues on the pre-trade
            # (drifted) weights, not on the new targets.
            pre_rebal_w = current_w.copy()

            if should_rebal or i == 0:
                # L5: Rolling turnover budget check
                trade_turnover = float(np.abs(target_w[i] - current_w).sum())

                if (
                    should_rebal
                    and self.policy.max_annual_turnover is not None
                    and i > 0
                ):
                    rolling_used = sum(turnover_ring)
                    if rolling_used + trade_turnover > self.policy.max_annual_turnover:
                        should_rebal = False

                if should_rebal or i == 0:
                    new_w = target_w[i].copy()

                    # ── Partial rebalance: only snap drifted positions ──
                    if (
                        self.policy.partial_rebalance
                        and self.policy.drift_threshold is not None
                        and i > 0
                    ):
                        per_asset_drift = np.abs(current_w - new_w)
                        keep_mask = per_asset_drift <= self.policy.drift_threshold
                        # Positions within band keep their current (drifted) weight
                        new_w[keep_mask] = current_w[keep_mask]
                        partial_saved += int(keep_mask.sum())
                        # Re-normalise only the traded sleeve so kept weights
                        # remain genuinely kept. For market-neutral books,
                        # use gross exposure rather than net sum.
                        new_w = self._renormalize_traded_sleeve(new_w, target_w[i], keep_mask)

                    # ── Tax-lot avoidance heuristic ──
                    if self.policy.avoid_short_term_gains and i > 0:
                        keep_mask = np.zeros(m, dtype=bool)
                        for j in range(m):
                            if new_w[j] < current_w[j]:
                                # Reducing position — check holding period
                                bars_held = i - entry_bar[j]
                                if bars_held < self.policy.short_term_days:
                                    # Keep current weight (don't sell young lots)
                                    new_w[j] = current_w[j]
                                    keep_mask[j] = True
                        # Re-normalise only the sleeve that can trade, so young
                        # lots remain frozen and market-neutral books use gross.
                        if keep_mask.any():
                            new_w = self._renormalize_traded_sleeve(new_w, target_w[i], keep_mask)

                    # Track turnover (cached — computed once above)
                    actual_turnover = float(np.abs(new_w - current_w).sum())
                    turnover_ring.append(actual_turnover)

                    # Update entry bars for new / increased positions
                    for j in range(m):
                        if new_w[j] > current_w[j] + 1e-9:
                            entry_bar[j] = i

                    current_w = new_w
                    rebal[i] = True
                else:
                    turnover_ring.append(0.0)
            else:
                turnover_ring.append(0.0)

            current_w = self._mask_unavailable_vector(current_w, available[i])
            if rebal[i]:
                period_w = self._mask_unavailable_vector(pre_rebal_w, available[i])
            else:
                period_w = current_w.copy()
            return_w[i] = period_w
            day_ret = (period_w * ret[i]).sum()
            nav[i] = nav[i - 1] * (1.0 + day_ret) if i > 0 else 1.0
            peak_nav = max(peak_nav, nav[i])

            if not rebal[i] and i > 0:
                # DRIFT: w'_j = w_j × (1 + r_j) / (cash + Σ(w_k × (1 + r_k)))
                current_w = self._drift_with_cash(period_w, ret[i])

            actual_w[i] = current_w

        # Build output DataFrames
        actual_weights = pd.DataFrame(actual_w, index=dates, columns=instruments)
        self.return_weights = pd.DataFrame(return_w, index=dates, columns=instruments)
        rebal_flags = pd.Series(rebal, index=dates, name="rebalance")

        # Stats
        rebal_count = int(rebal.sum())
        self.stats = {
            "rebalance_count": rebal_count,
            "calendar_count": cal_count,
            "drift_count": drift_count,
            "tracking_error_count": te_count,
            "signal_count": sig_count,
            "circuit_breaker_count": cb_count,
            "partial_positions_saved": partial_saved,
            "rolling_turnover_used": float(sum(turnover_ring)),
            "rebalance_pct": rebal_count / max(n, 1),
            "avg_days_between": n / max(rebal_count, 1),
        }

        return actual_weights, rebal_flags

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize_freq(freq: str) -> str:
        """Map deprecated pandas freq aliases to modern equivalents."""
        _LEGACY = {"BM": "BME", "BQ": "BQE", "BA": "BYE",
                    "BMS": "BMS", "BQS": "BQS", "BAS": "BYS"}
        return _LEGACY.get(freq, freq)

    def _calendar_flags(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Generate calendar-based rebalance flags."""
        freq = self.policy.frequency

        if freq is None:
            return pd.Series(False, index=dates)

        if freq == "D":
            return pd.Series(True, index=dates)

        if freq == "W":
            return pd.Series(dates.weekday == self.policy.weekday, index=dates)

        # Every N trading days: "21D", "5D", etc.
        if freq.endswith("D") and freq[:-1].isdigit():
            n = int(freq[:-1])
            flags = pd.Series(False, index=dates)
            flags.iloc[::n] = True
            return flags

        # pandas offset aliases: BME, BQE, BYE, etc.
        freq = self._normalize_freq(freq)
        try:
            rebal_dates = pd.date_range(dates[0], dates[-1], freq=freq)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Unsupported rebalance frequency: {self.policy.frequency!r}"
            ) from exc
        # Snap each scheduled date to the last actual trading date on or
        # before it — a scheduled month/quarter-end falling on an exchange
        # holiday would otherwise drop that rebalance entirely.
        pos = np.searchsorted(dates.values, rebal_dates.values, side="right") - 1
        pos = np.unique(pos[pos >= 0])
        flags = np.zeros(len(dates), dtype=bool)
        flags[pos] = True
        return pd.Series(flags, index=dates)

    def _signal_change_flags(self, weights: pd.DataFrame) -> pd.Series:
        """
        Detect significant weight / signal changes.

        NOTE: operates on ``target_weights`` (post-normalisation), not
        on the raw signal DataFrame.  This means weight changes caused
        by universe rotation (instrument added/removed) will trigger a
        rebalance even if the underlying alpha signal is unchanged.
        This is intentional — universe changes require position
        adjustment regardless of signal state.
        """
        if not self.policy.rebalance_on_signal_change:
            return pd.Series(False, index=weights.index)
        weight_diff = weights.diff().abs().max(axis=1)
        return weight_diff > self.policy.signal_change_threshold
