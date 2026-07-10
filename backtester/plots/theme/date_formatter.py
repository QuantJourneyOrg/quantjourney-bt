"""
    Date Formatter Utility
    ---------------------------------------------------------


    Changes vs original:
    - Consolidated three duplicate frequency maps into a single ``FreqConfig`` registry
    - Locators created via factory lambdas (no wasted instantiations)
    - Fixed closure capture bug in FuncFormatter lambda
    - Updated deprecated pandas freq aliases (M→ME, Q→QE, A→YE, etc.)
    - Fixed daily-label logic at month boundaries
    - Added fallback for single-data-point edge case
    - Removed redundant %W custom handler (standard strftime already handles it)

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

import datetime
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

__all__ = ["DateFormatter"]


# ---------------------------------------------------------------------------
# Frequency registry — single source of truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreqConfig:
    """Bundles a matplotlib locator factory, default date format, and pandas freq alias."""

    locator_factory: Callable[[], mdates.DateLocator]
    default_format: str
    pandas_freq: str  # modern pandas ≥2.2 alias


def _auto() -> mdates.AutoDateLocator:
    return mdates.AutoDateLocator()


_FREQ_REGISTRY: dict[str, FreqConfig] = {
    # Daily / business daily
    "D": FreqConfig(lambda: _auto(), "%Y-%m-%d", "D"),
    "B": FreqConfig(lambda: _auto(), "%Y-%m-%d", "B"),
    # Weekly
    "W": FreqConfig(lambda: mdates.WeekdayLocator(byweekday=mdates.MO), "%Y-%m-%d", "W"),
    "W-MON": FreqConfig(lambda: mdates.WeekdayLocator(byweekday=mdates.MO), "%Y-%m-%d", "W-MON"),
    "W-FRI": FreqConfig(lambda: mdates.WeekdayLocator(byweekday=mdates.FR), "%Y-%m-%d", "W-FRI"),
    "2W": FreqConfig(
        lambda: mdates.WeekdayLocator(byweekday=mdates.MO, interval=2), "%Y-%m-%d", "2W"
    ),
    # Monthly
    "M": FreqConfig(lambda: mdates.MonthLocator(), "%Y-%m", "ME"),
    "MS": FreqConfig(lambda: mdates.MonthLocator(), "%Y-%m", "MS"),
    "BM": FreqConfig(lambda: mdates.MonthLocator(), "%Y-%m", "BME"),
    # Quarterly
    "Q": FreqConfig(lambda: mdates.MonthLocator(bymonth=[1, 4, 7, 10]), "%Y-Q%Q", "QE"),
    "QS": FreqConfig(lambda: mdates.MonthLocator(bymonth=[1, 4, 7, 10]), "%Y-Q%Q", "QS"),
    "BQ": FreqConfig(lambda: mdates.MonthLocator(bymonth=[3, 6, 9, 12]), "%Y-Q%Q", "BQE"),
    # Yearly
    "Y": FreqConfig(lambda: mdates.YearLocator(), "%Y", "YE"),
    "YE": FreqConfig(lambda: mdates.YearLocator(), "%Y", "YE"),
    "A": FreqConfig(lambda: mdates.YearLocator(), "%Y", "YE"),
    "YS": FreqConfig(lambda: mdates.YearLocator(), "%Y", "YS"),
    "AS": FreqConfig(lambda: mdates.YearLocator(), "%Y", "YS"),
    "BY": FreqConfig(lambda: mdates.YearLocator(), "%Y", "BYE"),
    "BA": FreqConfig(lambda: mdates.YearLocator(), "%Y", "BYE"),
}

# Groups used by ``map_dates_index_to_str`` to decide label strategy
_YEARLY_FREQS = {"YE", "BYE", "YS", "Y", "A", "BA", "BY"}
_QUARTERLY_FREQS = {"QE", "BQE", "QS", "Q", "BQ"}
_MONTHLY_FREQS = {"ME", "BME", "MS", "M", "BM"}
_WEEKLY_FREQS = {"W", "W-MON", "W-FRI", "2W"}
_DAILY_FREQS = {"D", "B"}


# ---------------------------------------------------------------------------
# DateFormatter
# ---------------------------------------------------------------------------


class DateFormatter:
    """Utility for intelligent date-axis formatting on matplotlib charts."""

    # -- Locator / formatter ------------------------------------------------

    @staticmethod
    def get_locator_and_formatter(
        freq: str,
        date_format: str | None = None,
    ) -> tuple[mdates.DateLocator, mticker.Formatter]:
        """
        Return a (locator, formatter) pair appropriate for *freq*.

        Parameters
        ----------
        freq : str
            Frequency string (e.g. ``'D'``, ``'W'``, ``'M'``, ``'Q'``, ``'Y'``).
        date_format : str, optional
            Custom strftime format.  Falls back to the registry default.

        Returns
        -------
        tuple[DateLocator, Formatter]
        """
        cfg = _FREQ_REGISTRY.get(freq)
        locator = cfg.locator_factory() if cfg else _auto()
        fmt = date_format or (cfg.default_format if cfg else "%Y-%m-%d")

        # Capture *fmt* by value (default-argument trick) to avoid closure bug
        formatter = mticker.FuncFormatter(
            lambda x, pos, _fmt=fmt: DateFormatter._format_date(mdates.num2date(x), _fmt)
        )
        return locator, formatter

    # -- Custom format handlers ---------------------------------------------

    @staticmethod
    def _format_date(date: datetime.datetime, format_str: str) -> str:
        """
        Format *date* with *format_str*, supporting the custom ``%Q`` token
        for fiscal quarter numbers.
        """
        if "%Q" in format_str:
            quarter = (date.month - 1) // 3 + 1
            return date.strftime(format_str.replace("%Q", str(quarter)))
        return date.strftime(format_str)

    # -- Bar-plot date mapping ----------------------------------------------

    @staticmethod
    def map_dates_index_to_str(
        data: pd.DataFrame | pd.Series,
        **kwargs,
    ) -> tuple[pd.DataFrame | pd.Series, list[str]]:
        """
        Re-index *data* with formatted date strings and generate tick labels
        suitable for bar charts.

        Parameters
        ----------
        data : DataFrame or Series
            Must have a ``DatetimeIndex``.
        x_date_freq : str, default ``'YE'``
            Desired tick frequency (uses modern pandas aliases).
        x_date_format : str, default ``'%b-%y'``
            ``strftime`` format for labels.

        Returns
        -------
        tuple[DataFrame | Series, list[str]]
            Re-indexed data and corresponding tick labels (empty strings for
            positions that should not display a label).
        """
        x_date_freq: str = kwargs.get("x_date_freq", "YE")
        x_date_format: str = kwargs.get("x_date_format", "%b-%y")

        re_indexed_data = data.copy()

        if x_date_freq is None or not isinstance(data.index, pd.DatetimeIndex):
            return re_indexed_data, list(data.index)

        dates_index = pd.to_datetime(data.index)

        # Resolve to modern pandas freq alias
        cfg = _FREQ_REGISTRY.get(x_date_freq)
        pd_freq = cfg.pandas_freq if cfg else x_date_freq

        # Generate tick positions; fall back to finer granularity if needed
        ticks = pd.date_range(start=dates_index[0], end=dates_index[-1], freq=pd_freq)

        if len(ticks) <= 1:
            for fallback in ["ME", "W", "D"]:
                ticks = pd.date_range(start=dates_index[0], end=dates_index[-1], freq=fallback)
                if len(ticks) > 1:
                    pd_freq = fallback
                    break

        # Edge case: single data point — just format and return
        if len(ticks) <= 1:
            re_indexed_data.index = dates_index.strftime(x_date_format)
            return re_indexed_data, [t.strftime(x_date_format) for t in dates_index]

        # Re-index with formatted strings
        re_indexed_data.index = dates_index.strftime(x_date_format)

        # Build labels based on resolved frequency
        datalabels = DateFormatter._build_labels(dates_index, pd_freq, x_date_format)

        # Suppress consecutive duplicates
        prev = ""
        for i, label in enumerate(datalabels):
            if label == prev:
                datalabels[i] = ""
            else:
                prev = label

        return re_indexed_data, datalabels

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _build_labels(
        dates_index: pd.DatetimeIndex,
        pd_freq: str,
        fmt: str,
    ) -> list[str]:
        """Choose label strategy based on resolved frequency."""

        if pd_freq in _YEARLY_FREQS:
            return [t.strftime(fmt) if t.month == 12 else "" for t in dates_index]

        if pd_freq in _QUARTERLY_FREQS:
            return [t.strftime(fmt) if t.month % 3 == 0 else "" for t in dates_index]

        if pd_freq in _MONTHLY_FREQS:
            return [t.strftime(fmt) for t in dates_index]

        if pd_freq in _WEEKLY_FREQS:
            return [t.strftime("%d-%b") for t in dates_index]

        if pd_freq in _DAILY_FREQS:
            labels: list[str] = []
            for i, t in enumerate(dates_index):
                if i == 0:
                    labels.append(t.strftime("%d-%b"))
                # Show label when entering a new month
                elif t.month != dates_index[i - 1].month:
                    labels.append(t.strftime("%d-%b"))
                else:
                    labels.append("")
            return labels

        # Unknown freq — label everything
        return [t.strftime(fmt) for t in dates_index]
