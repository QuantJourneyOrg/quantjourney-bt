"""
Probability of Backtest Overfitting (PBO).

Bailey, Borwein, López de Prado & Zhu (2017),
"The Probability of Backtest Overfitting".

Uses Combinatorial Symmetric Cross-Validation (CSCV) adapted for
walk-forward folds.  Given N fold results each with an IS Sharpe
and an OOS Sharpe, we:

  1. Enumerate all C(N, ⌊N/2⌋) ways to split folds into two halves
     (J = "training proxy", J̄ = "test proxy").
  2. For each split compute the *logit*:
         λ_c = mean(OOS Sharpe in J̄) / mean(IS Sharpe in J)
     which measures how much of the in-sample promise survives out-of-sample.
  3. PBO = fraction of splits where λ_c ≤ 0  (i.e. OOS collapses or
     goes negative, implying the IS optimisation was spurious).

When C(N, N/2) is large we sample up to ``max_combinations`` splits.

Interpretation:
    PBO < 0.15 → low overfit risk
    0.15 – 0.40 → moderate
    PBO > 0.40 → likely overfit

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional, Sequence

import numpy as np


def _n_choose_k(n: int, k: int) -> int:
    """Exact C(n,k) via math.comb (Python ≥ 3.8)."""
    return math.comb(n, k)


def probability_of_backtest_overfitting(
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
    *,
    n_partitions: int = 16,  # kept for API compat; not used internally
    max_combinations: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Compute PBO via CSCV on walk-forward fold results.

    Args:
        is_sharpes:  Per-fold IS (in-sample) Sharpe ratios.
        oos_sharpes: Per-fold OOS (out-of-sample) Sharpe ratios.
                     Must have the same length as *is_sharpes*.
        n_partitions: Unused — retained for backward API compatibility.
        max_combinations: When C(N, N/2) exceeds this number, sample
                          randomly instead of exhaustive enumeration.
        seed: RNG seed for reproducible sampling.

    Returns:
        PBO ∈ [0, 1].  Fraction of combinatorial splits where the
        OOS performance collapses (logit ≤ 0).
    """
    n = len(is_sharpes)
    if n != len(oos_sharpes):
        raise ValueError("is_sharpes and oos_sharpes must have equal length")
    if n < 4:
        # Need at least 4 folds for a meaningful 2/2 split
        return 0.0

    is_arr = np.asarray(is_sharpes, dtype=np.float64)
    oos_arr = np.asarray(oos_sharpes, dtype=np.float64)

    half = n // 2
    total_combos = _n_choose_k(n, half)

    rng = np.random.default_rng(seed)

    # ── Generate or sample splits ─────────────────────────────────────
    if total_combos <= max_combinations:
        # Exhaustive
        splits = list(combinations(range(n), half))
    else:
        # Random sampling without replacement of index-tuples
        seen: set[tuple[int, ...]] = set()
        indices = np.arange(n)
        while len(seen) < max_combinations:
            perm = tuple(sorted(rng.choice(indices, size=half, replace=False)))
            seen.add(perm)
        splits = list(seen)

    # ── Evaluate each split ───────────────────────────────────────────
    n_overfit = 0
    logits: list[float] = []

    for j_indices in splits:
        j_set = set(j_indices)
        j_bar = [i for i in range(n) if i not in j_set]

        # J  = "IS proxy" half  →  mean IS Sharpe (what optimisation found)
        is_mean_j = float(is_arr[list(j_set)].mean())
        # J̄ = "OOS proxy" half →  mean OOS Sharpe (reality check)
        oos_mean_jbar = float(oos_arr[j_bar].mean())

        if is_mean_j > 0:
            lam = oos_mean_jbar / is_mean_j
        elif is_mean_j == 0:
            # IS zero → ambiguous; treat as overfit if OOS also non-positive
            lam = 0.0 if oos_mean_jbar <= 0 else 1.0
        else:
            # IS negative → ratio flips sign; positive OOS is good
            lam = 1.0 if oos_mean_jbar >= 0 else 0.0

        logits.append(lam)
        if lam <= 0.0:
            n_overfit += 1

    pbo = n_overfit / len(splits) if splits else 0.0
    return float(pbo)


def pbo_logit_distribution(
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
    *,
    max_combinations: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Return the full logit distribution λ_c for diagnostic plotting.

    Each element is mean(OOS in J̄) / mean(IS in J) for one combinatorial
    split.  The PBO is simply ``(logits <= 0).mean()``.
    """
    n = len(is_sharpes)
    if n < 4:
        return np.array([], dtype=np.float64)

    is_arr = np.asarray(is_sharpes, dtype=np.float64)
    oos_arr = np.asarray(oos_sharpes, dtype=np.float64)
    half = n // 2

    total_combos = _n_choose_k(n, half)
    rng = np.random.default_rng(seed)

    if total_combos <= max_combinations:
        splits = list(combinations(range(n), half))
    else:
        seen: set[tuple[int, ...]] = set()
        indices = np.arange(n)
        while len(seen) < max_combinations:
            perm = tuple(sorted(rng.choice(indices, size=half, replace=False)))
            seen.add(perm)
        splits = list(seen)

    logits = np.empty(len(splits), dtype=np.float64)
    for idx, j_indices in enumerate(splits):
        j_set = set(j_indices)
        j_bar = [i for i in range(n) if i not in j_set]

        is_mean = float(is_arr[list(j_set)].mean())
        oos_mean = float(oos_arr[j_bar].mean())

        if is_mean > 0:
            logits[idx] = oos_mean / is_mean
        elif is_mean == 0:
            logits[idx] = 0.0 if oos_mean <= 0 else 1.0
        else:
            logits[idx] = 1.0 if oos_mean >= 0 else 0.0

    return logits
