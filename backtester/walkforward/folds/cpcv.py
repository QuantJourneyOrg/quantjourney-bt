"""
CPCV fold scheme — placeholder for Phase 10.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.folds.base import Fold


class CPCVFoldScheme:
    """Combinatorial Purged Cross-Validation — Phase 10 implementation."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self._cfg = config

    def generate_folds(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        trading_dates: pd.DatetimeIndex,
    ) -> List[Fold]:
        raise NotImplementedError(
            "CPCV fold scheme is planned for Phase 10. "
            "Use 'rolling', 'expanding', or 'anchored' for now."
        )
