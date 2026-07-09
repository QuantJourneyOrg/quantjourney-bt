"""
Base types for walk-forward fold generation.

``Fold`` is the immutable data contract describing one IS/OOS split.
``FoldScheme`` is the Protocol that all fold generators implement.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """
    Immutable description of a single walk-forward fold.

    Dates are inclusive calendar dates (``pd.Timestamp``).
    The *effective* IS window is ``[train_start, effective_is_end]``
    after purge/embargo have been applied.
    """

    fold_id: int
    scheme: str

    # Raw boundaries (before purge/embargo)
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    # After purge/embargo (None when no dates are excluded)
    effective_is_end: pd.Timestamp
    purge_start: Optional[pd.Timestamp]   # first excluded date
    purge_end: Optional[pd.Timestamp]     # last excluded date (= oos_start - 1 trading day)


@runtime_checkable
class FoldScheme(Protocol):
    """Protocol for fold generators (Open/Closed Principle)."""

    def generate_folds(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        trading_dates: pd.DatetimeIndex,
    ) -> List[Fold]:
        """
        Generate all folds for the given date range.

        Args:
            start: First available trading date.
            end: Last available trading date.
            trading_dates: Sorted ``DatetimeIndex`` of actual trading days.

        Returns:
            Ordered list of ``Fold`` objects.
        """
        ...
