"""
Position-Limit Risk Model
=========================

Hard constraints on individual position sizes and optional
sector/group limits.  Applied *after* other risk models so that
the final weights respect compliance rules.

Multiple constraint types:

- **max_weight**: per-instrument cap (e.g. 0.40 = 40% max)
- **min_weight**: per-instrument floor (e.g. 0.02 = 2% min for active)
- **max_total_leverage**: sum(|w|) cap (e.g. 1.5 = 150%)
- **sector_limits**: dict of sector → max aggregate weight

After capping, excess is redistributed proportionally among
uncapped instruments to preserve total exposure.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtester.risk.base import RiskModel


@dataclass
class PositionLimitModel(RiskModel):
    """
    Enforce position-level and portfolio-level limits.

    Parameters
    ----------
    max_weight : float
        Maximum absolute weight for any single instrument.
    min_weight : float
        Minimum weight for active (non-zero) instruments.
        Set to 0 to disable.
    max_total_leverage : float
        Maximum sum of absolute weights.
    sector_limits : dict
        Mapping of sector_name → max aggregate weight.
        Requires ``metadata["sectors"]`` = dict(instrument → sector).
    """

    max_weight: float = 0.40
    min_weight: float = 0.0
    max_total_leverage: float = 2.0
    sector_limits: Dict[str, float] = field(default_factory=dict)

    def adjust(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        metadata: Optional[Dict] = None,
    ) -> pd.DataFrame:
        out = weights.copy()

        for i in range(len(out)):
            row = out.iloc[i].copy()

            if row.abs().sum() < 1e-10:
                continue

            # 1) Per-instrument cap
            row = self._apply_cap(row)

            # 2) Sector limits
            if self.sector_limits and metadata and "sectors" in metadata:
                row = self._apply_sector_limits(row, metadata["sectors"])

            # 3) Total leverage cap
            total = row.abs().sum()
            if total > self.max_total_leverage:
                row = row * (self.max_total_leverage / total)

            # 4) Per-instrument floor after leverage scaling.
            if self.min_weight > 0:
                row = self._apply_floor(row)
                row = self._apply_cap(row)
                total = row.abs().sum()
                if total > self.max_total_leverage:
                    row = row * (self.max_total_leverage / total)

            out.iloc[i] = row

        return out

    def _apply_cap(self, row: pd.Series) -> pd.Series:
        """Cap each weight at max_weight, redistributing until no cap is breached."""
        capped = row.astype(float).copy()
        target_abs_total = float(capped.abs().sum())
        if target_abs_total < 1e-10:
            return capped

        for _ in range(len(capped) + 1):
            over = capped.abs() > self.max_weight + 1e-10
            if not over.any():
                break
            capped[over] = np.sign(capped[over]) * self.max_weight
            excess = target_abs_total - float(capped.abs().sum())
            if excess <= 1e-10:
                break
            room = (self.max_weight - capped.abs()).clip(lower=0.0)
            candidates = room > 1e-10
            if not candidates.any():
                break
            weights = capped[candidates].abs()
            if float(weights.sum()) <= 1e-10:
                weights = room[candidates]
            redistribution = (weights / float(weights.sum())) * excess
            redistribution = np.minimum(redistribution, room[candidates])
            capped[candidates] += np.sign(capped[candidates].replace(0.0, 1.0)) * redistribution

        return capped.clip(lower=-self.max_weight, upper=self.max_weight)

    def _apply_floor(self, row: pd.Series) -> pd.Series:
        """Set active positions below min_weight to min_weight."""
        active = row.abs() > 1e-10
        below_floor = active & (row.abs() < self.min_weight)

        if below_floor.any():
            row[below_floor] = np.sign(row[below_floor]) * self.min_weight

        return row

    def _apply_sector_limits(
        self, row: pd.Series, sectors: Dict[str, str]
    ) -> pd.Series:
        """Cap aggregate weight per sector."""
        for sector, limit in self.sector_limits.items():
            sector_instruments = [
                inst for inst, s in sectors.items() if s == sector
            ]
            sector_mask = row.index.isin(sector_instruments)
            sector_total = row[sector_mask].abs().sum()

            if sector_total > limit:
                scale = limit / sector_total
                row[sector_mask] *= scale

        return row

    def __repr__(self) -> str:
        parts = [f"max={self.max_weight:.0%}"]
        if self.min_weight > 0:
            parts.append(f"min={self.min_weight:.0%}")
        if self.max_total_leverage < 10:
            parts.append(f"lev={self.max_total_leverage:.1f}x")
        if self.sector_limits:
            parts.append(f"sectors={len(self.sector_limits)}")
        return f"PositionLimitModel({', '.join(parts)})"
