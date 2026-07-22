"""
Portfolio Vol-Target Risk Model
===============================

Scales ALL weights by a single scalar so that the *portfolio's*
realised volatility matches a target level.

This is the most common institutional risk overlay — used by
managed-futures funds, risk-parity strategies, and vol-targeting
wrappers.

Extracted from the inline vol-scaling logic that was previously
duplicated in every strategy's ``_compute_weights()``.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.risk.base import RiskModel


@dataclass
class VolTargetModel(RiskModel):
    """
    Scale weights so portfolio vol ≈ *target_vol*.

    Parameters
    ----------
    target_vol : float
        Annualised target volatility (e.g. 0.15 for 15%).
    lookback : int
        Rolling window (trading days) for realised vol estimate.
    max_leverage : float
        Hard cap on the scale factor (prevents blowup in low-vol regimes).
    rebalance_freq : str
        How often to recompute the scale factor.
        ``"BMS"`` = first business day of month (default).
        ``"D"`` = every day.
        ``"W-MON"`` = every Monday.
    ann_factor : float
        Annualisation factor (default √252 for daily data).
    """

    target_vol: float = 0.15
    lookback: int = 63
    max_leverage: float = 2.0
    rebalance_freq: str = "BMS"
    ann_factor: float | None = None

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

        # Detect rescoring dates
        if self.rebalance_freq == "D":
            is_rescore = pd.Series(True, index=weights.index)
        else:
            periods = pd.Series(
                weights.index.to_period(
                    {"BMS": "M", "W-MON": "W", "MS": "M", "QS": "Q"}.get(self.rebalance_freq, "M")
                ),
                index=weights.index,
            )
            is_rescore = periods != periods.shift(1)

        prev_scale = 1.0

        for i in range(self.lookback, n):
            row_w = weights.iloc[i]
            total_w = row_w.abs().sum()

            if total_w < 1e-8:
                out.iloc[i] = 0.0
                continue

            # Normalise to unit exposure for vol calc
            norm_w = row_w / total_w

            if is_rescore.iloc[i]:
                window = returns.iloc[max(0, i - self.lookback) : i]
                port_rets = (window * norm_w).sum(axis=1)
                realised_vol = port_rets.std() * ann_factor

                if realised_vol > 0.01:
                    prev_scale = min(
                        self.target_vol / realised_vol,
                        self.max_leverage,
                    )
                else:
                    prev_scale = 1.0

            out.iloc[i] = row_w * prev_scale

        return out

    def __repr__(self) -> str:
        return (
            f"VolTargetModel(target={self.target_vol:.0%}, "
            f"lookback={self.lookback}, max_lev={self.max_leverage})"
        )
