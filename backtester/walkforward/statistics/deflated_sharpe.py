"""
Deflated Sharpe Ratio (DSR).

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting, and Non-Normality".

When an optimizer tests N parameter combinations and reports the best
Sharpe, the expected maximum under the null is strictly positive even
if every single combination has zero true alpha.  DSR corrects for
this selection bias by comparing observed SR* against E[max(SR|H₀)].

Interpretation:
    DSR > 2.0  →  robust (SR* far exceeds selection-bias threshold)
    1.0 – 2.0  →  marginal
    DSR < 1.0  →  likely false positive

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# Euler–Mascheroni constant
_GAMMA = 0.5772156649015329


def _expected_max_sr(
    sr_std: float,
    n_trials: int,
) -> float:
    """
    E[max(SR)] under the null, using the Euler–Mascheroni approximation
    of the expected maximum of *n_trials* i.i.d. standard normals,
    scaled by sr_std.

    Bailey & López de Prado (2014), Eq. 6.
    """
    from scipy.stats import norm  # lazy import — only needed here

    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def deflated_sharpe(
    sharpes: Sequence[float],
    n_trials: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute the Deflated Sharpe Ratio.

    Args:
        sharpes: Observed Sharpe ratios (e.g. per-fold IS Sharpes).
                 Only max(sharpes) and std(sharpes) are used.
        n_trials: Total number of parameter combinations explored
                  (sum of Optuna trials across folds, or len(sharpes)
                  when no optimizer is used).
        skewness: Skewness of the trial-Sharpe distribution (0 = normal).
        kurtosis: *Raw* kurtosis (3.0 = normal).  Excess kurtosis is
                  kurtosis − 3.

    Returns:
        DSR value.  Interpretable like a z-score:
        > 2.0 robust, 1.0–2.0 marginal, < 1.0 likely false positive.
    """
    if not sharpes or n_trials < 1:
        return 0.0

    arr = np.asarray(sharpes, dtype=np.float64)
    sr_max = float(arr.max())
    sr_std = float(arr.std(ddof=0))

    if sr_std == 0.0 or n_trials <= 1:
        return sr_max  # no variability → can't deflate

    try:
        e_max = _expected_max_sr(sr_std, n_trials)
    except ImportError:
        # scipy unavailable — return raw max as fallback
        return sr_max

    # Probabilistic Sharpe Ratio denominator with skew/kurtosis adjustment
    # Bailey & López de Prado (2014), Eq. 4
    excess_kurt = kurtosis - 3.0
    psr_var = 1.0 - skewness * sr_max + (excess_kurt / 4.0) * sr_max**2
    if psr_var <= 0:
        psr_var = 1.0  # degenerate — fall back to unadjusted

    dsr = (sr_max - e_max) / (sr_std / math.sqrt(psr_var))
    return float(dsr)
