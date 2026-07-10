"""
Probability of Backtest Overfitting (PBO).

Bailey, Borwein, López de Prado & Zhu (2017),
"The Probability of Backtest Overfitting".

True CSCV-style PBO requires TRIAL-LEVEL out-of-sample data: every
candidate configuration must be evaluated on both halves of each
combinatorial split, so that the IS-selected trial's OOS *rank* among
all candidates can be measured.  A walk-forward pipeline that only
records one IS Sharpe and one OOS Sharpe per fold does NOT contain the
information needed to compute PBO — any number derived from fold-level
Sharpes alone is a transfer diagnostic, not PBO.

This module therefore provides:

- ``selected_trial_logit`` / ``pbo_from_selected_ranks`` — the honest,
  computable statistic: per fold, the optimizer's top-K trials are
  re-backtested on the OOS window (opt-in via
  ``WalkForwardConfig.pbo_trials``); the IS-selected trial's relative
  OOS rank ω̄ ∈ (0, 1) yields a logit λ = ln(ω̄ / (1 − ω̄)), and
  PBO = fraction of folds with λ ≤ 0 (selection lands in the bottom
  half OOS).

- ``probability_of_backtest_overfitting`` — DEPRECATED.  The previous
  implementation computed a fold-level IS→OOS ratio with no trials, no
  selection, and no rank, and could label a textbook overfit
  (IS 3.0 → OOS 0.05) as "no overfit".  It now returns ``nan`` and
  warns; use the ``pbo_trials`` pipeline instead.

- ``pbo_logit_distribution`` — retained as a *diagnostic* IS→OOS
  transfer-ratio distribution across combinatorial fold splits (used
  for plotting).  It is NOT the CSCV PBO.

Interpretation (for the real, rank-based PBO):
    PBO < 0.15 → low overfit risk
    0.15 – 0.40 → moderate
    PBO > 0.40 → likely overfit

Copyright (c) 2026 QuantJourney.
Updated: 07.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import logging
import math
import warnings as _warnings
from collections.abc import Sequence
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


def _n_choose_k(n: int, k: int) -> int:
    """Exact C(n,k) via math.comb (Python ≥ 3.8)."""
    return math.comb(n, k)


# ── Rank-based PBO (the computable, honest statistic) ─────────────────


def selected_trial_logit(
    selected_value: float,
    candidate_values: Sequence[float],
) -> float | None:
    """
    Logit of the IS-selected trial's relative OOS rank among candidates.

    Args:
        selected_value: OOS objective value of the IS-selected trial.
        candidate_values: OOS objective values of ALL K evaluated
            candidates (including the selected one).  Higher = better.

    Returns:
        λ = ln(ω̄ / (1 − ω̄)) with ω̄ = rank / (K + 1), where rank uses
        average-tie ranking and K = best.  λ ≤ 0 means the selection
        landed in the bottom half out-of-sample.  ``None`` when the
        statistic is not computable (K < 2 or non-finite selection).
    """
    vals = np.asarray(list(candidate_values), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2 or not np.isfinite(selected_value):
        return None

    # Average-tie rank: 1 = worst … K = best
    rank = (
        float((vals < selected_value).sum()) + (float((vals == selected_value).sum()) + 1.0) / 2.0
    )
    omega = rank / (vals.size + 1.0)
    if not (0.0 < omega < 1.0):
        return None
    return float(math.log(omega / (1.0 - omega)))


def pbo_from_selected_ranks(logits: Sequence[float]) -> float:
    """
    PBO from per-fold selection-rank logits.

    Args:
        logits: One λ per fold, from ``selected_trial_logit``.

    Returns:
        PBO ∈ [0, 1] — fraction of folds where λ ≤ 0 (the IS-selected
        trial ranked in the bottom half OOS).  ``nan`` when no logits
        are available.
    """
    finite = [float(logit) for logit in logits if logit is not None and np.isfinite(logit)]
    if not finite:
        return float("nan")
    return sum(1.0 for logit in finite if logit <= 0.0) / len(finite)


# ── Deprecated fold-level pseudo-PBO ──────────────────────────────────


def probability_of_backtest_overfitting(
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
    *,
    n_partitions: int = 16,  # kept for API compat; never used
    max_combinations: int = 10_000,
    seed: int = 42,
) -> float:
    """
    DEPRECATED — this is NOT the CSCV PBO and always returns ``nan``.

    The former implementation split fold-level (IS, OOS) Sharpe pairs
    combinatorially and reported the fraction of splits with
    mean(OOS)/mean(IS) ≤ 0.  That statistic contains no trials, no
    selection event, and no rank: a strategy collapsing from IS 3.0 to
    OOS 0.05 still scored 0.0 ("no overfit").  Rather than report a
    falsely reassuring number, this function now returns ``nan``.

    Use ``WalkForwardConfig.pbo_trials = K`` (K ≥ 2) with an optimizer
    so the walk-forward runner evaluates the top-K trials OOS per fold;
    the engine then computes the rank-based PBO via
    ``pbo_from_selected_ranks``.
    """
    if len(is_sharpes) != len(oos_sharpes):
        raise ValueError("is_sharpes and oos_sharpes must have equal length")

    _warnings.warn(
        "probability_of_backtest_overfitting(is_sharpes, oos_sharpes) is "
        "deprecated: fold-level Sharpes cannot yield the CSCV PBO. It now "
        "returns nan. Enable WalkForwardConfig.pbo_trials for the "
        "rank-based PBO.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "PBO requested from fold-level Sharpes only — not computable "
        "(requires per-trial OOS evaluation); returning nan."
    )
    return float("nan")


# ── Diagnostic transfer-ratio distribution (plotting only) ────────────


def pbo_logit_distribution(
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
    *,
    max_combinations: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Diagnostic IS→OOS transfer-ratio distribution across fold splits.

    NOTE: This is NOT the CSCV PBO logit distribution — it operates on
    fold-level Sharpes, not trial-level ranks.  Each element is
    mean(OOS in J̄) / mean(IS in J) for one combinatorial split of the
    folds; values ≤ 0 indicate splits where OOS collapsed.  Useful as a
    plot of IS→OOS transfer stability only.
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
