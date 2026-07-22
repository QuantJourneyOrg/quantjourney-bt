"""
Walk-forward rank-stability diagnostics and legacy PBO helpers.

Bailey, Borwein, López de Prado & Zhu (2017),
"The Probability of Backtest Overfitting".

Canonical CSCV PBO requires TRIAL-LEVEL out-of-sample data: every
candidate configuration must be evaluated on both halves of each
combinatorial split, so that the IS-selected trial's OOS *rank* among
all candidates can be measured across symmetric combinations of IS/OOS
blocks.  The rolling walk-forward top-K pipeline in this package does
not perform CSCV and must not be reported as canonical PBO.

This module therefore provides:

- ``selected_trial_rank_logit`` and
  ``walk_forward_top_k_rank_failure_rate`` — a useful rolling
  rank-stability diagnostic.  Per fold, the optimizer's top-K IS trials
  are re-backtested on that fold's OOS window; the IS-selected trial's
  relative OOS rank yields a logit, and the aggregate reports the
  fraction of folds with logit <= 0.

- ``probability_of_backtest_overfitting`` — DEPRECATED.  The previous
  implementation computed a fold-level IS→OOS ratio with no trials, no
  selection, and no rank, and could label a textbook overfit
  (IS 3.0 → OOS 0.05) as "no overfit".  It now returns ``nan`` and
  warns; use the ``rank_stability_trials`` pipeline instead.

- ``is_oos_transfer_distribution`` — an unrelated diagnostic IS→OOS
  transfer-ratio distribution across fold splits (used for plotting).

The former ``selected_trial_logit``, ``pbo_from_selected_ranks`` and
``pbo_logit_distribution`` names remain as deprecated compatibility
wrappers for one release.  They do not turn these rolling diagnostics
into CSCV PBO.

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


# ── Rolling top-K OOS rank-stability diagnostic ──────────────────────


def selected_trial_rank_logit(
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


def walk_forward_top_k_rank_failure_rate(logits: Sequence[float]) -> float:
    """
    Failure rate from rolling-fold selection-rank logits.

    Args:
        logits: One lambda per fold, from ``selected_trial_rank_logit``.

    Returns:
        Value in [0, 1]: fraction of rolling folds where the IS-selected
        trial ranked in the bottom half of the evaluated top-K set OOS.
        ``nan`` when no logits are available.  This is not CSCV PBO.
    """
    finite = [float(logit) for logit in logits if logit is not None and np.isfinite(logit)]
    if not finite:
        return float("nan")
    return sum(1.0 for logit in finite if logit <= 0.0) / len(finite)


def selected_trial_logit(
    selected_value: float,
    candidate_values: Sequence[float],
) -> float | None:
    """Deprecated alias for :func:`selected_trial_rank_logit`."""
    _warnings.warn(
        "selected_trial_logit is deprecated; use selected_trial_rank_logit. "
        "The rolling top-K diagnostic is not canonical CSCV PBO.",
        DeprecationWarning,
        stacklevel=2,
    )
    return selected_trial_rank_logit(selected_value, candidate_values)


def pbo_from_selected_ranks(logits: Sequence[float]) -> float:
    """Deprecated alias for ``walk_forward_top_k_rank_failure_rate``."""
    _warnings.warn(
        "pbo_from_selected_ranks is deprecated; use "
        "walk_forward_top_k_rank_failure_rate. The result is not canonical "
        "CSCV PBO.",
        DeprecationWarning,
        stacklevel=2,
    )
    return walk_forward_top_k_rank_failure_rate(logits)


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

    Use ``WalkForwardConfig.rank_stability_trials = K`` (K >= 2) with an optimizer
    so the walk-forward runner evaluates the top-K trials OOS per fold;
    the engine then computes a rolling top-K rank-failure diagnostic.
    """
    if len(is_sharpes) != len(oos_sharpes):
        raise ValueError("is_sharpes and oos_sharpes must have equal length")

    _warnings.warn(
        "probability_of_backtest_overfitting(is_sharpes, oos_sharpes) is "
        "deprecated: fold-level Sharpes cannot yield the CSCV PBO. It now "
        "returns nan. Enable WalkForwardConfig.rank_stability_trials for "
        "the rolling top-K rank-failure diagnostic.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "PBO requested from fold-level Sharpes only — not computable "
        "(requires per-trial OOS evaluation); returning nan."
    )
    return float("nan")


# ── Diagnostic transfer-ratio distribution (plotting only) ────────────


def is_oos_transfer_distribution(
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


def pbo_logit_distribution(
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
    *,
    max_combinations: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Deprecated alias for :func:`is_oos_transfer_distribution`."""
    _warnings.warn(
        "pbo_logit_distribution is deprecated; use "
        "is_oos_transfer_distribution. This fold-level diagnostic is not "
        "a CSCV PBO logit distribution.",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_oos_transfer_distribution(
        is_sharpes,
        oos_sharpes,
        max_combinations=max_combinations,
        seed=seed,
    )
