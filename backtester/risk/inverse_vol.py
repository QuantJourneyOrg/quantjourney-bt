"""
Inverse-Volatility Weighting Risk Model
========================================

Reweights instruments so that each gets weight proportional to
``1 / σ_i`` (inverse of its realised volatility).  This is a
simple risk-budgeting approach: low-vol assets get more capital.

If incoming weights are all-or-nothing (binary signal), this is
equivalent to classic inverse-vol weighting.  If incoming weights
are continuous (alpha scores), the model combines alpha conviction
with vol adjustment::

    adjusted_w_i ∝ raw_w_i / σ_i

Then renormalised to preserve the original total exposure.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.risk.base import RiskModel


@dataclass
class InverseVolModel(RiskModel):
    """
    Weight each instrument inversely to its realised volatility.

    Parameters
    ----------
    lookback : int
        Rolling window (trading days) for per-asset vol.
    ann_factor : float
        Annualisation factor.
    min_vol : float
        Floor on vol estimate to prevent divide-by-zero blowup.
    blend_alpha : bool
        If True, multiply raw weights by inverse-vol *then* renormalise
        (conviction × risk adjustment).
        If False, ignore raw weights and allocate purely by inverse-vol
        among instruments that have non-zero raw weight.
    """

    lookback: int = 63
    ann_factor: float | None = None
    min_vol: float = 0.01
    blend_alpha: bool = True

    def adjust(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        metadata: dict | None = None,
    ) -> pd.DataFrame:
        n = len(weights)
        if n == 0:
            return weights

        out = weights.copy()
        periods_per_year = int((metadata or {}).get("periods_per_year", 252))
        ann_factor = (
            float(self.ann_factor)
            if self.ann_factor is not None
            else float(np.sqrt(max(periods_per_year, 1)))
        )

        # The public contract leaves the complete warm-up unchanged.  Shifted
        # rolling volatility matches the legacy strictly-prior window while
        # computing the full matrix in one native pandas pass.
        # ``DataFrame.std`` skips missing values in the legacy sliced window,
        # so two valid observations are sufficient even after warm-up.
        rolling = returns.rolling(window=self.lookback, min_periods=2).std().shift(1)
        rolling = rolling.iloc[:n].copy()
        rolling.index = weights.index[: len(rolling)]
        vols = pd.DataFrame(np.nan, index=weights.index, columns=rolling.columns, dtype=float)
        if len(rolling):
            vols.iloc[: len(rolling)] = rolling.to_numpy(dtype=float)
        vols = (vols * ann_factor).clip(lower=self.min_vol)
        inv_vol = 1.0 / vols
        active = weights.abs() > 1e-10

        if self.blend_alpha:
            blended = (weights * inv_vol).where(active, 0.0)
        else:
            signed_weights = pd.DataFrame(
                np.sign(weights.to_numpy(dtype=float)),
                index=weights.index,
                columns=weights.columns,
            )
            blended = (signed_weights * inv_vol).where(active, 0.0)

        total = blended.abs().sum(axis=1)
        original_exposure = weights.abs().sum(axis=1)
        scaled = blended.div(total, axis=0).mul(original_exposure, axis=0)
        scaled.loc[total < 1e-10, :] = 0.0
        scaled = scaled.reindex(columns=weights.columns)
        out.iloc[self.lookback :] = scaled.iloc[self.lookback :].to_numpy(dtype=float)

        return out

    def __repr__(self) -> str:
        mode = "blend" if self.blend_alpha else "pure"
        return f"InverseVolModel(lookback={self.lookback}, mode={mode})"
