"""
Universe — first-class grid of (dates × instruments).
=====================================================

Provides factory methods for pre-shaped DataFrames and cached
market-data derivatives so strategy authors never need to write:

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    months  = pd.Series(close.index.to_period("M"), index=close.index)
    rets    = close.pct_change()

Instead:

    weights = self.universe.zeros()
    months  = self.universe.periods("M")
    rets    = self.universe.returns

Zero breaking changes — ``self.universe`` is a new cached property
on ``Backtester``.  Existing strategies work without modification.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional


@dataclass(frozen=False)
class Universe:
    """Tradeable universe grid — dates × instruments.

    Parameters
    ----------
    _close : pd.DataFrame
        Adjusted close prices (dates × instruments).  Used as the
        canonical shape source for all factory methods.
    _sectors : dict[str, str]
        Instrument → sector mapping (e.g. ``{"AAPL": "Tech"}``).
        Used by ``PositionLimitModel`` and future sector-aware logic.
    """

    _close: pd.DataFrame
    _sectors: Dict[str, str] = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────
    # Shape properties
    # ─────────────────────────────────────────────────────────────

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Trading‑calendar date index."""
        return self._close.index

    @property
    def instruments(self) -> list[str]:
        """Ordered list of instrument tickers."""
        return self._close.columns.tolist()

    @property
    def n_dates(self) -> int:
        return len(self._close.index)

    @property
    def n_instruments(self) -> int:
        return len(self._close.columns)

    @property
    def shape(self) -> tuple[int, int]:
        """(n_dates, n_instruments)."""
        return self._close.shape

    # ─────────────────────────────────────────────────────────────
    # Factory methods — pre-shaped containers (numpy-style API)
    # ─────────────────────────────────────────────────────────────

    def zeros(self) -> pd.DataFrame:
        """DataFrame of zeros with shape ``(dates × instruments)``.

        Example::

            signals = self.universe.zeros()
            weights = self.universe.zeros()
        """
        return pd.DataFrame(0.0, index=self.dates, columns=self.instruments)

    def ones(self) -> pd.DataFrame:
        """DataFrame of ones with shape ``(dates × instruments)``.

        Example::

            signals = self.universe.ones()   # all instruments active
        """
        return pd.DataFrame(1.0, index=self.dates, columns=self.instruments)

    def full(self, fill: float) -> pd.DataFrame:
        """DataFrame of ``fill`` with shape ``(dates × instruments)``.

        Example::

            weights = self.universe.full(1.0 / self.universe.n_instruments)
        """
        return pd.DataFrame(fill, index=self.dates, columns=self.instruments)

    def empty_frame(self, fill: float = 0.0) -> pd.DataFrame:
        """Backward-compatible alias for ``zeros()`` / ``full(fill)``.

        .. deprecated:: 0.11
            Use ``zeros()``, ``ones()``, or ``full(fill)`` instead.
        """
        if fill == 0.0:
            return self.zeros()
        elif fill == 1.0:
            return self.ones()
        return self.full(fill)

    def zeros_series(self) -> pd.Series:
        """Series of zeros aligned to the trading‑date index."""
        return pd.Series(0.0, index=self.dates)

    def ones_series(self) -> pd.Series:
        """Series of ones aligned to the trading‑date index."""
        return pd.Series(1.0, index=self.dates)

    def empty_series(self, fill: float = 0.0) -> pd.Series:
        """Backward-compatible alias for ``zeros_series()``."""
        return pd.Series(fill, index=self.dates)

    def periods(self, freq: str = "M") -> pd.Series:
        """Calendar periods aligned to trading dates.

        Example::

            months = self.universe.periods("M")
            is_first = months != months.shift(1)  # 1st day of each month

        Common values: ``"M"`` (month), ``"Q"`` (quarter), ``"W"`` (week).
        """
        return pd.Series(self.dates.to_period(freq), index=self.dates)

    # ─────────────────────────────────────────────────────────────
    # Cached market‑data derivatives
    # ─────────────────────────────────────────────────────────────

    @cached_property
    def returns(self) -> pd.DataFrame:
        """Simple returns, preserving NaN gaps as unavailable bars."""
        returns = self._close.pct_change(fill_method=None)
        if len(returns) > 0:
            first_idx = returns.index[0]
            first_available = self._close.loc[first_idx].notna()
            returns.loc[first_idx, first_available] = 0.0
        return returns

    @cached_property
    def log_returns(self) -> pd.DataFrame:
        """Log returns, preserving NaN gaps as unavailable bars."""
        returns = np.log(self._close / self._close.shift(1))
        if len(returns) > 0:
            first_idx = returns.index[0]
            first_available = self._close.loc[first_idx].notna()
            returns.loc[first_idx, first_available] = 0.0
        return returns

    @cached_property
    def cumulative_returns(self) -> pd.DataFrame:
        """Cumulative simple returns: ``(1 + returns).cumprod()``."""
        return (1.0 + self.returns).cumprod()

    # ─────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────

    @property
    def sectors(self) -> Dict[str, str]:
        """Instrument → sector map (e.g. for PositionLimitModel)."""
        return self._sectors

    @sectors.setter
    def sectors(self, value: Dict[str, str]) -> None:
        self._sectors = value

    # ─────────────────────────────────────────────────────────────
    # Dunder
    # ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Universe(instruments={self.instruments}, "
            f"dates={self.dates[0].date()}..{self.dates[-1].date()}, "
            f"shape={self.shape})"
        )

    def __len__(self) -> int:
        """Number of trading dates."""
        return self.n_dates
