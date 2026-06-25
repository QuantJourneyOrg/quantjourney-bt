"""
Reporting frequency configuration for performance reports.

Reporting frequency is a presentation and diagnostics cadence. It must not
change the backtest path, fills, NAV accounting, or execution timing. It only
controls frequency-dependent report statistics such as rolling volatility,
rolling beta, tail-risk windows, and correlation snapshots.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class ReportingFrequency(str, Enum):
    """Supported report calculation cadences."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

    @classmethod
    def parse(cls, value: "ReportingFrequency | str | None") -> "ReportingFrequency":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            return cls.DAILY
        raw = str(value).strip().lower().replace("-", "_")
        aliases = {
            "d": cls.DAILY,
            "day": cls.DAILY,
            "daily": cls.DAILY,
            "b": cls.DAILY,
            "business_daily": cls.DAILY,
            "w": cls.WEEKLY,
            "week": cls.WEEKLY,
            "weekly": cls.WEEKLY,
            "m": cls.MONTHLY,
            "month": cls.MONTHLY,
            "monthly": cls.MONTHLY,
            "me": cls.MONTHLY,
            "bme": cls.MONTHLY,
            "q": cls.QUARTERLY,
            "quarter": cls.QUARTERLY,
            "quarterly": cls.QUARTERLY,
            "qe": cls.QUARTERLY,
            "bqe": cls.QUARTERLY,
        }
        if raw not in aliases:
            valid = ", ".join(f.value for f in cls)
            raise ValueError(f"Unsupported reporting_frequency={value!r}. Use one of: {valid}.")
        return aliases[raw]


@dataclass(frozen=True)
class ReportingFrequencyConfig:
    """Resolved report cadence, windows, annualisation and labels."""

    frequency: ReportingFrequency
    pandas_freq: Optional[str]
    label: str
    short_label: str
    periods_per_year: int
    rank: int
    is_long_period: bool
    rolling_window: int
    beta_window: int
    turnover_window: int
    correlation_window: int
    regime_frequency: str
    min_trailing_obs: int = 12

    @property
    def observation_label(self) -> str:
        return {
            ReportingFrequency.DAILY: "daily",
            ReportingFrequency.WEEKLY: "weekly",
            ReportingFrequency.MONTHLY: "monthly",
            ReportingFrequency.QUARTERLY: "quarterly",
        }[self.frequency]

    @property
    def stats_label(self) -> str:
        return f"{self.label} stats"


_BASE = {
    ReportingFrequency.DAILY: (None, "Daily", "B", 252, 0),
    ReportingFrequency.WEEKLY: ("W-FRI", "Weekly", "W-FRI", 52, 1),
    ReportingFrequency.MONTHLY: ("BME", "Monthly", "BME", 12, 2),
    ReportingFrequency.QUARTERLY: ("BQE", "Quarterly", "BQE", 4, 3),
}


def infer_native_frequency(index: pd.Index) -> ReportingFrequency:
    """Infer native sampling frequency from the median distance between observations."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    idx = pd.DatetimeIndex(index).dropna().sort_values().unique()
    if len(idx) < 3:
        return ReportingFrequency.DAILY

    deltas = pd.Series(idx).diff().dropna().dt.total_seconds() / 86400.0
    median_days = float(deltas.median())
    if median_days <= 3.5:
        return ReportingFrequency.DAILY
    if median_days <= 10.5:
        return ReportingFrequency.WEEKLY
    if median_days <= 45.0:
        return ReportingFrequency.MONTHLY
    return ReportingFrequency.QUARTERLY


def resolve_reporting_frequency(
    value: ReportingFrequency | str | None,
    *,
    index: Optional[pd.Index] = None,
    long_threshold_years: float = 5.0,
) -> ReportingFrequencyConfig:
    """Resolve a requested reporting cadence into windows and annualisation."""
    freq = ReportingFrequency.parse(value)
    pandas_freq, label, short_label, periods_per_year, rank = _BASE[freq]

    if index is not None and len(index) >= 2:
        dt_index = pd.DatetimeIndex(index)
        years = max((dt_index[-1] - dt_index[0]).days / 365.25, 0.0)
    else:
        years = 0.0
    is_long_period = years >= long_threshold_years

    if is_long_period:
        rolling_window = periods_per_year if freq == ReportingFrequency.DAILY else periods_per_year * 3
        beta_window = periods_per_year * 3
        regime_frequency = "QE"
    else:
        rolling_window = periods_per_year
        beta_window = periods_per_year
        regime_frequency = "ME"

    turnover_window = periods_per_year
    correlation_window = max(periods_per_year, 12)

    return ReportingFrequencyConfig(
        frequency=freq,
        pandas_freq=pandas_freq,
        label=label,
        short_label=short_label,
        periods_per_year=periods_per_year,
        rank=rank,
        is_long_period=is_long_period,
        rolling_window=max(2, int(rolling_window)),
        beta_window=max(2, int(beta_window)),
        turnover_window=max(2, int(turnover_window)),
        correlation_window=max(2, int(correlation_window)),
        regime_frequency=regime_frequency,
    )


def validate_reporting_frequency(
    index: pd.Index,
    requested: ReportingFrequencyConfig | ReportingFrequency | str,
) -> None:
    """Refuse silent up-sampling from coarse data to finer report cadence."""
    config = (
        requested
        if isinstance(requested, ReportingFrequencyConfig)
        else resolve_reporting_frequency(requested, index=index)
    )
    native = infer_native_frequency(index)
    native_rank = _BASE[native][4]
    if config.rank < native_rank:
        raise ValueError(
            "Requested reporting_frequency="
            f"{config.frequency.value!r} is finer than native data frequency "
            f"{native.value!r}. Reporting cannot up-sample missing observations."
        )


def resample_return_series(
    returns: pd.Series,
    config: ReportingFrequencyConfig,
) -> pd.Series:
    """Compound returns to the configured report cadence."""
    r = returns.dropna()
    if config.frequency == ReportingFrequency.DAILY:
        return r
    nav = (1.0 + r).cumprod()
    sampled = nav.resample(config.pandas_freq).last().dropna()
    out = sampled.pct_change().fillna(0.0)
    out.name = returns.name
    return out


def resample_return_frame(
    returns: pd.DataFrame,
    config: ReportingFrequencyConfig,
) -> pd.DataFrame:
    """Compound a returns frame to the configured report cadence."""
    if config.frequency == ReportingFrequency.DAILY:
        return returns.dropna(how="all")
    compounded = []
    for col in returns.columns:
        compounded.append(resample_return_series(returns[col], config).rename(col))
    return pd.concat(compounded, axis=1).dropna(how="all")


def resample_nav_series(
    nav: pd.Series,
    config: ReportingFrequencyConfig,
) -> pd.Series:
    """Sample a NAV path at report frequency, preserving the native start point."""
    clean = nav.dropna()
    if config.frequency == ReportingFrequency.DAILY:
        return clean
    sampled = clean.resample(config.pandas_freq).last().dropna()
    if sampled.empty:
        return clean
    if sampled.index[0] != clean.index[0]:
        sampled = pd.concat([clean.iloc[[0]], sampled]).sort_index()
        sampled = sampled.loc[~sampled.index.duplicated(keep="first")]
    return sampled


def resample_optional_frame(
    frame: Optional[pd.DataFrame],
    config: ReportingFrequencyConfig,
) -> Optional[pd.DataFrame]:
    """Sample state-like frames such as weights/positions at report frequency."""
    if frame is None or config.frequency == ReportingFrequency.DAILY:
        return frame
    sampled = frame.resample(config.pandas_freq).last().dropna(how="all")
    return sampled.ffill()


def make_reporting_portfolio_data(portfolio_data, config: ReportingFrequencyConfig):
    """Return a PortfolioData view sampled at reporting frequency."""
    if config.frequency == ReportingFrequency.DAILY:
        return portfolio_data

    from backtester.portfolio.portf_data import PortfolioData

    nav = resample_nav_series(portfolio_data.net_asset_value, config)
    weights = resample_optional_frame(portfolio_data.weights, config)
    positions = resample_optional_frame(portfolio_data.positions, config)

    cash = portfolio_data.cash
    if isinstance(cash, pd.Series):
        cash = resample_nav_series(cash, config).reindex(nav.index).ffill()

    out = PortfolioData(
        instruments=portfolio_data.instruments,
        net_asset_value=nav,
        input_weights=portfolio_data.input_weights,
        rebalance_flags=portfolio_data.rebalance_flags,
        asset_name_map=portfolio_data.asset_name_map,
        trading_calendar=portfolio_data.trading_calendar,
        weights=weights,
        positions=positions,
        cash_buffer=portfolio_data.cash_buffer,
        cash=cash,
        _skip_initial_metrics=False,
    )
    return out


class ReportingInstrumentView:
    """Lightweight InstrumentCalculations-compatible view with resampled returns."""

    def __init__(self, source, config: ReportingFrequencyConfig) -> None:
        self._source = source
        self._config = config
        self._returns_cache: Optional[pd.DataFrame] = None

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_cache is None:
            returns = self._source.returns
            if isinstance(returns, pd.Series):
                returns = returns.to_frame()
            self._returns_cache = resample_return_frame(returns, self._config)
        return self._returns_cache

    @property
    def prices(self):
        return self._source.prices

    @property
    def units(self):
        units = self._source.units
        if self._config.frequency == ReportingFrequency.DAILY:
            return units
        return units.resample(self._config.pandas_freq).last().ffill()

    @property
    def instruments(self):
        return self._source.instruments

    def __getattr__(self, name: str):
        return getattr(self._source, name)


def safe_annualize_return(total_return: float, observations: int, periods_per_year: int) -> float:
    """Annualise a compounded total return using the reporting periods/year."""
    if observations <= 0 or not np.isfinite(total_return):
        return float("nan")
    years = max(observations / float(periods_per_year), 1e-9)
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)
