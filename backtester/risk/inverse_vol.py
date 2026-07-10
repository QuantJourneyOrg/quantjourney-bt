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

        for i in range(self.lookback, n):
            row_w = weights.iloc[i]
            active = row_w.abs() > 1e-10

            if active.sum() == 0:
                out.iloc[i] = 0.0
                continue

            # Per-asset realised vol
            window = returns.iloc[max(0, i - self.lookback) : i]
            vols = window.std() * ann_factor
            vols = vols.clip(lower=self.min_vol)

            inv_vol = 1.0 / vols

            if self.blend_alpha:
                # Conviction × risk: w_i * (1/σ_i), then renormalise
                blended = row_w * inv_vol
                blended = blended.where(active, 0.0)
            else:
                # Pure inverse-vol among active instruments
                blended = inv_vol.where(active, 0.0)

            total = blended.abs().sum()
            if total < 1e-10:
                out.iloc[i] = 0.0
                continue

            # Preserve original total exposure
            original_exposure = row_w.abs().sum()
            out.iloc[i] = blended / total * original_exposure

        return out

    def __repr__(self) -> str:
        mode = "blend" if self.blend_alpha else "pure"
        return f"InverseVolModel(lookback={self.lookback}, mode={mode})"
