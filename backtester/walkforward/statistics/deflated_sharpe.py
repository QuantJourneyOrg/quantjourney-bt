"""
Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting, and Non-Normality".

When an optimizer tests N parameter combinations and reports the best
Sharpe, the expected maximum under the null is strictly positive even
if every single combination has zero true alpha.  DSR corrects for
this selection bias by testing the observed SR against
SR₀ = E[max(SR|H₀)] via the Probabilistic Sharpe Ratio:

    DSR = PSR(SR₀)
        = Φ[ (SR̂ − SR₀) · √(T − 1) / √(1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂²) ]

where:
    SR̂  = observed per-period Sharpe of the candidate strategy
    SR₀  = E[max SR over N trials] = √V[SR]·((1−γ)·Φ⁻¹(1−1/N)
                                           + γ·Φ⁻¹(1−1/(N·e)))
    γ    = Euler–Mascheroni constant
    T    = number of return observations behind SR̂
    γ₃   = skewness of the candidate's returns
    γ₄   = RAW kurtosis of the candidate's returns (3.0 = normal)

IMPORTANT — units: all Sharpe inputs (``observed_sr``, ``trial_sharpes``,
``benchmark_sr``) must be expressed per observation period of the T
returns (e.g. *daily* Sharpe with daily T).  Passing annualized Sharpes
with daily T massively overstates significance.

Interpretation (DSR is a probability in [0, 1]):
    DSR ≥ 0.95   →  robust (SR survives multiple-testing deflation)
    0.80 – 0.95  →  marginal
    DSR < 0.80   →  likely false positive

Copyright (c) 2026 QuantJourney.
Updated: 07.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Euler–Mascheroni constant
_GAMMA = 0.5772156649015329


def _expected_max_sr(
    sr_std: float,
    n_trials: float,
) -> float:
    """
    E[max(SR)] under the null, using the Euler–Mascheroni approximation
    of the expected maximum of *n_trials* effectively independent
    standard normals,
    scaled by sr_std (= √V[SR] across trials).

    Bailey & López de Prado (2014):
        E[max SR] ≈ √V[SR] · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))
    """
    from scipy.stats import norm  # lazy import — only needed here

    if not math.isfinite(n_trials) or n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def probabilistic_sharpe(
    observed_sr: float,
    *,
    benchmark_sr: float = 0.0,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Probabilistic Sharpe Ratio — P[true SR > benchmark_sr].

    Bailey & López de Prado (2012/2014):
        PSR = Φ[ (SR̂ − SR₀)·√(T−1) / √(1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²) ]

    Args:
        observed_sr: Per-period Sharpe of the candidate strategy
                     (same periodicity as the T observations).
        benchmark_sr: Null-hypothesis Sharpe SR₀ (same units).
        n_obs: T — number of return observations behind observed_sr.
        skewness: Skewness γ₃ of the candidate's returns.
        kurtosis: RAW kurtosis γ₄ of the candidate's returns (3 = normal).
                  Fat tails (γ₄ > 3) widen the SR estimator's variance and
                  therefore LOWER the probability.

    Returns:
        Probability in [0, 1].
    """
    from scipy.stats import norm  # lazy import

    if n_obs is None or n_obs < 2:
        return 0.0  # cannot assess significance with < 2 observations

    denom_sq = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    if denom_sq <= 0.0:
        # Degenerate higher moments — fall back to the normal-returns
        # denominator (γ₃ = 0, γ₄ = 3) rather than fabricating certainty.
        denom_sq = 1.0 + 0.5 * observed_sr**2

    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1.0) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe(
    trial_sharpes: Sequence[float],
    n_trials: int | None = None,
    *,
    effective_n_trials: float | None = None,
    observed_sr: float | None = None,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (a probability in [0, 1]).

    The trial population (``trial_sharpes`` / ``n_trials``) and the
    candidate (``observed_sr`` / ``n_obs`` / moments) must describe
    consistent quantities in the SAME per-period units:

    - ``trial_sharpes``: finite completed-trial objective values used to
      estimate √V[SR] across trials for the E[max] threshold.
    - ``n_trials``: raw completed-trial count. Defaults to
      ``len(trial_sharpes)``; folds are not trials.
    - ``effective_n_trials``: optional effective number of independent
      trials used in E[max]. Strongly correlated parameter variants do
      not automatically count as independent trials. When dependence is
      not estimated, leaving this unset uses the raw count as a
      conservative approximation. Must be in [1, n_trials].
    - ``observed_sr``: per-period Sharpe of the selected candidate.
      Defaults to ``max(trial_sharpes)``.
    - ``n_obs``: T, the number of return observations behind
      ``observed_sr``.
    - ``skewness`` / ``kurtosis``: moments of the candidate strategy's
      RETURNS (kurtosis is RAW, 3 = normal).

    Degenerate inputs are handled honestly: with N ≤ 1 or zero variance
    across trials there is no selection bias to correct, so the result
    reduces to the PSR against ``benchmark_sr`` (NOT the raw Sharpe).

    Returns:
        DSR = Φ(z) ∈ [0, 1].  ≥ 0.95 robust, 0.80–0.95 marginal,
        < 0.80 likely false positive.
    """
    arr = np.asarray(list(trial_sharpes), dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if observed_sr is None:
        if arr.size == 0:
            return 0.0  # nothing to evaluate
        observed_sr = float(arr.max())

    raw_n = int(n_trials) if n_trials is not None else int(arr.size)
    if raw_n < 0:
        raise ValueError("n_trials must be >= 0")

    if effective_n_trials is None:
        effective_n = float(raw_n)
    else:
        effective_n = float(effective_n_trials)
        if not math.isfinite(effective_n) or effective_n < 1.0:
            raise ValueError("effective_n_trials must be finite and >= 1")
        if raw_n >= 1 and effective_n > raw_n:
            raise ValueError("effective_n_trials must be <= n_trials")

    sr_std = float(arr.std(ddof=0)) if arr.size > 1 else 0.0

    if effective_n <= 1.0 or sr_std == 0.0:
        # No multiple-testing to deflate — plain PSR vs the benchmark.
        sr0 = benchmark_sr
    else:
        sr0 = benchmark_sr + _expected_max_sr(sr_std, effective_n)

    return probabilistic_sharpe(
        float(observed_sr),
        benchmark_sr=sr0,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )
