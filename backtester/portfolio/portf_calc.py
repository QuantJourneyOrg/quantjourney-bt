"""
Portfolio Calculations Facade - Thin Adapter Over calc/*
--------------------------------------------------------

This module provides a thin facade over PortfolioData, delegating analytics
to pure functions in quantjourney.portfolio.calc. It keeps orchestration
and config handling near data containers and leaves math to the calc layer.

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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from backtester.portfolio.portf_data import PortfolioData
from backtester.portfolio.config import CalcConfig, get_default_config
from backtester.portfolio.calc import returns as calc_returns
from backtester.portfolio.calc import risk as calc_risk
from backtester.portfolio.instr_calc import InstrumentCalculations


class MetricStatus(Enum):
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"
    ERROR = "ERROR"


@dataclass
class ValidationResult:
    status: MetricStatus
    message: str
    data: Any
    details: Optional[Dict[str, Any]] = None


class ReturnMethod(Enum):
    SIMPLE = "simple"
    LOG = "log"
    EXCESS = "excess"


class TimeFrame(Enum):
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "ME"
    QUARTERLY = "QE"
    YEARLY = "YE"
    MTD = "MTD"
    QTD = "QTD"
    YTD = "YTD"


class PortfolioCalculations:
    """Thin facade over PortfolioData delegating to calc.* modules."""

    def __init__(self, portfolio_data: PortfolioData, *, config: CalcConfig | None = None) -> None:
        self._portfolio_data = portfolio_data
        self._config: CalcConfig = config or get_default_config()
        self.trading_days = int(self._config.days_per_year)
        self.risk_free_rate = float(self._config.risk_free_rate_annual or 0.0)

    # Accessors --------------------------------------------------------
    @property
    def returns(self) -> pd.Series:
        return self._portfolio_data.returns

    @property
    def metric_returns(self) -> pd.Series:
        """Observed returns used for metrics, excluding display-only first zero."""
        r = getattr(self._portfolio_data, "returns_for_metrics", None)
        if r is None:
            r = self._portfolio_data.returns
        return r.replace([np.inf, -np.inf], np.nan).dropna()

    @property
    def weights(self) -> Optional[pd.DataFrame | pd.Series]:
        return self._portfolio_data.weights

    # Back-compat helpers for plotting layer ---------------------------
    @property
    def portfolio_data(self) -> PortfolioData:
        """Expose underlying data for code expecting .portfolio_data."""
        return self._portfolio_data

    @property
    def instrument_calculations(self) -> InstrumentCalculations:
        """Provide InstrumentCalculations built from underlying instruments."""
        return InstrumentCalculations(self._portfolio_data.instruments)

    @property
    def drawdowns(self) -> pd.Series:
        """Expose drawdowns series for compatibility with plotting code."""
        return self.compute_drawdowns()

    @staticmethod
    def _normalize_time_index_like(
        obj: pd.Series | pd.DataFrame,
        target_index: pd.Index,
    ) -> pd.Series | pd.DataFrame:
        out = obj.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        if isinstance(target_index, pd.DatetimeIndex) and target_index.tz is not None:
            out.index = (
                out.index.tz_convert(target_index.tz)
                if out.index.tz is not None
                else out.index.tz_localize(target_index.tz)
            )
        elif isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        return out

    # Validation -------------------------------------------------------
    def _validate_portfolio_data(self) -> ValidationResult:
        if self._portfolio_data is None:
            return ValidationResult(MetricStatus.FAILED, "Missing portfolio data", {"error": "Portfolio data not provided"})
        if self.returns is None or len(self.returns) == 0:
            return ValidationResult(MetricStatus.FAILED, "Empty return series", {"error": "No return data"})
        if len(self.returns) < self.trading_days:
            return ValidationResult(MetricStatus.WARNING, "Limited data history", {"warning": "Less than 1 year of data"})
        if self.weights is not None:
            w = self.weights
            if isinstance(w, pd.DataFrame):
                s = w.sum(axis=1).iloc[-1]
            else:
                s = w.sum()
            if not np.isclose(s, 1.0, rtol=1e-3):
                return ValidationResult(MetricStatus.WARNING, "Weights don't sum to 1", {"warning": f"Sum: {s}"})
        return ValidationResult(MetricStatus.SUCCESS, "Validation passed", {"message": "All checks completed"})

    # Returns ----------------------------------------------------------
    def compute_returns(self, method: str = "simple") -> pd.Series:
        r = self.metric_returns
        if method == "simple":
            return r
        elif method == "log":
            return np.log1p(r)
        else:
            raise ValueError(f"Invalid return method: {method}")

    def compute_cumulative_returns(self, starting_value: float = 1.0) -> Dict[str, Any]:
        if len(self.returns) == 0:
            return {"status": MetricStatus.ERROR.value, "message": "Insufficient data", "data": None}
        cum = starting_value * (1 + self.returns).cumprod()
        return {
            "status": MetricStatus.SUCCESS.value,
            "cumulative_returns": cum,
            "total_return": cum.iloc[-1] - starting_value,
            "annualized_return": calc_returns.compute_annualized_returns(self.returns.to_frame(), days_per_year=self.trading_days).iloc[0],
        }

    def compute_periodic_returns(self, period: str = "ME", method: str = "compound") -> Dict[str, Any]:
        if len(self.returns) == 0:
            return {"status": MetricStatus.ERROR.value, "message": "Insufficient data", "data": None}
        r = self.returns.dropna().sort_index()
        if method == "compound":
            periodic = (1 + r).resample(period).prod() - 1
        else:
            periodic = r.resample(period).sum()

        def _period_return(start=None, periods: Optional[int] = None) -> float:
            if start is not None:
                window = r.loc[start:]
            elif periods is not None:
                if len(r) < periods:
                    return np.nan
                window = r.iloc[-periods:]
            else:
                window = r
            if window.empty:
                return np.nan
            return float((1 + window).prod() - 1)

        def _annualized_trailing(periods: int) -> float:
            trailing = _period_return(periods=periods)
            if not np.isfinite(trailing):
                return np.nan
            return float((1 + trailing) ** (self.trading_days / periods) - 1)

        latest = r.index[-1]
        current_month_start = latest.replace(day=1)
        current_quarter_month = ((latest.month - 1) // 3) * 3 + 1
        current_quarter_start = latest.replace(month=current_quarter_month, day=1)
        current_year_start = latest.replace(month=1, day=1)
        nav = (1 + r).cumprod()
        ath = float(nav.max())
        drawdown_from_ath = float(nav.iloc[-1] / ath - 1) if ath > 0 else np.nan
        periods_per_year = max(int(self.trading_days), 1)

        statistics = {
            "MTD": _period_return(start=current_month_start),
            "QTD": _period_return(start=current_quarter_start),
            "YTD": _period_return(start=current_year_start),
            "1Y": _period_return(periods=periods_per_year),
            "3Y": _annualized_trailing(periods_per_year * 3),
            "5Y": _annualized_trailing(periods_per_year * 5),
            "ITD": _period_return(),
            "ATH Value": ath,
            "Drawdown from ATH (%)": drawdown_from_ath,
        }

        return {
            "status": MetricStatus.SUCCESS.value,
            "periodic_returns": periodic,
            "statistics": statistics,
        }

    # Risk & Ratios ----------------------------------------------------
    def compute_drawdowns(self) -> pd.Series:
        dd = calc_risk.compute_drawdowns(self.returns.to_frame())
        return dd.iloc[:, 0]

    def compute_max_drawdown(self) -> float:
        return calc_risk.compute_max_drawdown(self.returns.to_frame()).iloc[0]

    def compute_sharpe_ratio(self, risk_free_rate: Optional[float] = None, annualize: bool = True) -> float:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        sr = calc_risk.sharpe_ratio(self.returns.to_frame(), risk_free_rate=risk_free_rate, days_per_year=self.trading_days, annualize=annualize)
        return float(sr.iloc[0])

    def compute_sortino_ratio(
        self,
        risk_free_rate: Optional[float] = None,
        target_return: float = 0.0,
        annualize: bool = True,
    ) -> float:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        adjusted = self.returns - ((risk_free_rate + target_return) / self.trading_days)
        downside = adjusted[adjusted < 0]
        if len(adjusted) == 0 or len(downside) == 0:
            return np.nan
        downside_std = np.sqrt((downside ** 2).sum() / len(adjusted))
        if downside_std == 0:
            return np.nan
        ratio = adjusted.mean() / downside_std
        if annualize:
            ratio *= np.sqrt(self.trading_days)
        return float(ratio)

    # Turnover & Exposure ----------------------------------------------
    def compute_gross_weight_change(self) -> pd.Series:
        """Gross weight churn: sum(abs(diff(weights))) per date."""
        w = self.weights
        if w is None or len(w) == 0:
            return pd.Series(dtype=float)
        df = w.to_frame() if isinstance(w, pd.Series) else w
        return df.diff().abs().sum(axis=1).fillna(0.0)

    def compute_turnover(self) -> pd.Series:
        """Institutional half-turnover from weights.

        Turnover is defined as ``sum(abs(diff(weights))) / 2`` so buys and
        sells are not double-counted.  Use ``compute_gross_weight_change`` for
        raw gross churn.
        """
        return self.compute_gross_weight_change() / 2.0

    # Rolling ----------------------------------------------------------
    def compute_rolling_sortino(self, window: int = 252, risk_free_rate: float = 0.0, target_return: float = 0.0) -> pd.Series:
        """Rolling Sortino ratio using downside deviation within the window.

        Filters to only negative returns for the denominator (avoids zero-padding
        inflation) and annualises the result with sqrt(trading_days).
        """
        r = self.compute_returns() - (risk_free_rate / self.trading_days)
        if len(r) == 0:
            return pd.Series(dtype=float)
        td = self.trading_days
        def sortino_win(x: pd.Series) -> float:
            neg = x[x < target_return]
            if len(neg) < 2 or neg.std() == 0:
                return np.nan
            return (x.mean() / neg.std()) * np.sqrt(td)
        return r.rolling(window=window).apply(sortino_win, raw=False)

    def compute_annualized_return(self, returns: Optional[pd.Series] = None) -> float:
        if returns is None:
            returns = self.returns
        total_return = (1 + returns).prod() - 1
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        if years <= 0:
            return np.nan
        return (1 + total_return) ** (1 / years) - 1

    # ── Monthly stats ─────────────────────────────────────────────────
    # ── Period stats ──────────────────────────────────────────────────
    def compute_period_stats(self) -> Dict[str, float]:
        daily_wins = (self.returns > 0).mean() * 100
        monthly = self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        quarterly = self.returns.resample("QE").apply(lambda x: (1 + x).prod() - 1)
        yearly = self.returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        return {
            "win_days": daily_wins,
            "win_month": (monthly > 0).mean() * 100,
            "win_quarter": (quarterly > 0).mean() * 100,
            "win_year": (yearly > 0).mean() * 100,
        }

    # ── Expected returns ──────────────────────────────────────────────
    # ── Advanced annualised volatility ────────────────────────────────
    def compute_advanced_annualized_volatility(
        self,
        short_window: Optional[int] = None,
        long_window: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        r = self.returns.dropna()
        annual = np.sqrt(self.trading_days)
        std_vol = r.std() * annual
        short_window = int(short_window or (30 if self.trading_days >= 252 else max(2, self.trading_days)))
        long_window = int(long_window or max(2, self.trading_days))
        short_window = max(2, short_window)
        long_window = max(2, long_window)
        rolling_short = r.rolling(short_window).std() * annual
        rolling_long = r.rolling(long_window).std() * annual
        return {
            "standard": std_vol * 100,
            "current_30d": rolling_short.iloc[-1] * 100 if len(rolling_short.dropna()) > 0 else np.nan,
            "historical_252d": rolling_long.iloc[-1] * 100 if len(rolling_long.dropna()) > 0 else np.nan,
            "peak_95th": rolling_long.quantile(0.95) * 100 if len(rolling_long.dropna()) > 0 else np.nan,
            "short_window": short_window,
            "long_window": long_window,
            "summary_stats": {
                "min_vol": rolling_long.min() * 100 if len(rolling_long.dropna()) > 0 else np.nan,
                "max_vol": rolling_long.max() * 100 if len(rolling_long.dropna()) > 0 else np.nan,
                "avg_vol": rolling_long.mean() * 100 if len(rolling_long.dropna()) > 0 else np.nan,
            },
        }

    def compute_advanced_calmar_ratio(self, **kwargs) -> Dict[str, Any]:
        ann_ret = self.compute_annualized_return()
        max_dd = abs(self.compute_max_drawdown())
        calmar = ann_ret / max_dd if max_dd != 0 else np.inf
        return {"base_calmar": calmar, "status": "success"}

    def compute_recovery_factor(self) -> float:
        max_dd = abs(self.compute_max_drawdown())
        if max_dd == 0:
            return np.inf
        return self.compute_annualized_return() / max_dd

    def compute_advanced_turnover(
        self,
        trades_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Public/light turnover analytics from target-weight changes."""
        if self.weights is None:
            return {"status": "error", "message": "No weights data"}
        turnover = self.compute_turnover()
        if len(turnover) == 0:
            return {"status": "error", "message": "No turnover data"}
        avg = float(turnover.mean())
        total_turnover_ratio = float(turnover.sum())
        return {
            "total_turnover_ratio": total_turnover_ratio,
            "total_turnover_pct": total_turnover_ratio * 100,
            "total_traded_notional": np.nan,
            "total_turnover": total_turnover_ratio * 100,
            "average_turnover": avg * 100,
            "avg_daily_turnover_pct": avg * 100,
            "annualized_turnover": avg * self.trading_days * 100,
            "annualized_turnover_pct": avg * self.trading_days * 100,
            "max_turnover": float(turnover.max()) * 100,
            "daily_turnover": avg * 100,
            "turnover_std": float(turnover.std()) * 100 if len(turnover) > 1 else 0,
            "source": "weights_diff",
        }
