"""
Compatibility stubs for deep quantjourney dependencies.

Provides lightweight replacements for:
- ReturnTypes (from quantjourney.reporting.reporting)
- Reporting / ReportingParams (from quantjourney.reporting.reporting)
- DataFrequency (from quantjourney.utils.data_freq)
- TimePeriod (from quantjourney.utils.data_period)
- quantstats_stats functions (from quantjourney.reporting.quantstats_stats)

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ReturnTypes
# ---------------------------------------------------------------------------
class ReturnTypes(Enum):
    RELATIVE = "Relative"
    LOG = "Log"
    DIFFERENCE = "Diff"
    LEVEL = "Level"
    LEVEL0 = "Level0"


# ---------------------------------------------------------------------------
# Reporting enum stub — only the members used by instr_calc.py
# ---------------------------------------------------------------------------
class _ReportingMember:
    """Stub for ColumnDescriptor-based Reporting enum members."""

    def __init__(self, name: str):
        self._name = name

    def to_string(self) -> str:
        return self._name


class Reporting:
    TOTAL_RETURN = _ReportingMember("Total Return")
    ANNUAL_RETURN = _ReportingMember("Annual Return")
    NUM_YEARS = _ReportingMember("Num Years")
    NUM_OBSERVATIONS = _ReportingMember("Num Obs")
    START_DATE = _ReportingMember("Start Date")
    END_DATE = _ReportingMember("End Date")


@dataclass
class ReportingParams:
    freq: Optional[str] = None
    freq_volatility: str = "ME"
    freq_drawdown: str = "D"
    freq_regression: str = "QE"
    freq_excess_return: str = "ME"
    return_type: ReturnTypes = ReturnTypes.RELATIVE
    rates_data: Optional[pd.Series] = None
    alpha_annual_factor: float = 252.0


# ---------------------------------------------------------------------------
# DataFrequency stub — only resample_to_frequency is used
# ---------------------------------------------------------------------------
class DataFrequency:

    @staticmethod
    def resample_to_frequency(
        df: pd.DataFrame,
        freq: str,
        include_start_date: bool = False,
        include_end_date: bool = False,
        fill_na_method: Optional[str] = None,
    ) -> pd.DataFrame:
        freq_norm = {"M": "ME", "Q": "QE"}.get(freq, freq)
        resampled = df.resample(freq_norm).last()
        if include_start_date and df.index[0] not in resampled.index:
            first = df.iloc[[0]]
            resampled = pd.concat([first, resampled])
        if include_end_date and df.index[-1] not in resampled.index:
            last = df.iloc[[-1]]
            resampled = pd.concat([resampled, last])
        if fill_na_method == "ffill":
            resampled = resampled.ffill()
        return resampled


# ---------------------------------------------------------------------------
# DataDFS stub — only get_first_or_last_non_nan_index and
# get_index_before_first_non_nan are used by sampling.py
# ---------------------------------------------------------------------------
class DataDFS:

    @staticmethod
    def get_first_or_last_non_nan_index(df: pd.DataFrame, get_first: bool = True):
        if get_first:
            return [df[col].first_valid_index() or df.index[0] for col in df.columns]
        else:
            return [df[col].last_valid_index() or df.index[-1] for col in df.columns]

    @staticmethod
    def get_index_before_first_non_nan(df: pd.DataFrame):
        result = []
        for col in df.columns:
            first = df[col].first_valid_index()
            if first is not None:
                loc = df.index.get_loc(first)
                result.append(df.index[max(0, loc)])
            else:
                result.append(df.index[0])
        return result


# ---------------------------------------------------------------------------
# TimePeriod stub
# ---------------------------------------------------------------------------
@dataclass
class TimePeriod:
    start: Optional[str] = None
    end: Optional[str] = None


# ---------------------------------------------------------------------------
# quantstats_stats stubs — functions used by calc/risk.py and calc/outliers.py
# Provide simple implementations that don't require the quantstats package.
# ---------------------------------------------------------------------------
def ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index: RMS of drawdowns."""
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    dd = ((nav - peak) / peak) * 100
    return float(np.sqrt((dd ** 2).mean()))


def serenity_index(returns: pd.Series, rf: float = 0.0) -> float:
    """Serenity Index: annualized excess return / ulcer index."""
    ui = ulcer_index(returns)
    if ui == 0:
        return float("nan")
    ann_ret = (1 + returns.mean()) ** 252 - 1
    return float((ann_ret - rf) / ui)


def risk_of_ruin(returns: pd.Series) -> float:
    """Simplified risk of ruin estimate."""
    if returns.empty:
        return float("nan")
    wins = (returns > 0).sum()
    losses = (returns <= 0).sum()
    total = wins + losses
    if total == 0:
        return float("nan")
    win_rate = wins / total
    if win_rate >= 1.0:
        return 0.0
    if win_rate <= 0.0:
        return 1.0
    return float(((1 - win_rate) / win_rate) ** total)


def outliers(returns: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    """Return outlier values (beyond quantile threshold)."""
    upper = returns.quantile(quantile)
    lower = returns.quantile(1 - quantile)
    mask = (returns > upper) | (returns < lower)
    return returns.where(mask)


def remove_outliers(returns: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    """Replace outliers with NaN."""
    upper = returns.quantile(quantile)
    lower = returns.quantile(1 - quantile)
    mask = (returns > upper) | (returns < lower)
    return returns.where(~mask)
