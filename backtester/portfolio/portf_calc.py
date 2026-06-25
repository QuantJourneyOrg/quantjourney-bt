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
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from backtester.portfolio.portf_data import PortfolioData
from backtester.portfolio.config import CalcConfig, get_default_config
from backtester.portfolio.calc import returns as calc_returns
from backtester.portfolio.calc import risk as calc_risk
from backtester.portfolio.calc import rolling_stats as calc_roll
from backtester.portfolio.instr_calc import InstrumentCalculations
from backtester.portfolio.calc import liquidity as calc_liq


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
    def compute_volatility(self, window: int = 252) -> pd.Series:
        return calc_risk.compute_volatility(self.returns.to_frame(), window=window, days_per_year=self.trading_days).iloc[:, 0]

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

    def compute_information_ratio(self, benchmark_returns: pd.Series) -> float:
        ir = calc_risk.information_ratio(self.returns.to_frame(), benchmark_returns, days_per_year=self.trading_days)
        return float(ir.iloc[0])

    def compute_var(self, confidence: float = 0.95) -> float:
        return float(calc_risk.compute_var(self.returns.to_frame(), confidence=confidence).iloc[0])

    def compute_cvar(self, confidence: float = 0.95) -> float:
        return float(calc_risk.compute_cvar(self.returns.to_frame(), confidence=confidence).iloc[0])

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
    def compute_rolling_returns(self, window: int = 252, annualize: bool = True) -> pd.Series:
        rr = calc_roll.rolling_mean(self.returns.to_frame(), window=window).iloc[:, 0]
        return rr * (self.trading_days if annualize else 1.0)

    def compute_rolling_beta(self, benchmark_returns: pd.Series, window: int = 252) -> pd.Series:
        """Rolling beta vs a benchmark using cov/var over a sliding window."""
        r = self.compute_returns()
        benchmark_returns = self._normalize_time_index_like(benchmark_returns, r.index)
        r, b = r.align(benchmark_returns, join="inner")
        if len(r) == 0:
            return pd.Series(dtype=float)
        rolling_cov = r.rolling(window=window).cov(b)
        rolling_var = b.rolling(window=window).var()
        beta = rolling_cov / rolling_var
        return beta.dropna()

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

    def compute_rolling_calmar(self, window: int = 252) -> pd.Series:
        """Rolling Calmar ratio = annualised return / |rolling max drawdown|.

        Delegates to ``rolling_stats.rolling_calmar_ratio`` which benefits
        from Numba-accelerated max-DD kernels when ``QJ_USE_NUMBA=1``.
        """
        from backtester.portfolio.calc.rolling_stats import rolling_calmar_ratio

        r = self.compute_returns()
        if len(r) == 0:
            return pd.Series(dtype=float)
        df = rolling_calmar_ratio(r.to_frame("portfolio"), window=window)
        return df.iloc[:, 0]

    # Liquidity ---------------------------------------------------------
    def compute_nav_based_liquidity_proxy(self) -> Dict[str, Any]:
        """Return a diagnostic liquidity proxy based on NAV, not market liquidity."""
        nav = self._portfolio_data.net_asset_value
        returns = self.compute_returns()
        result = calc_liq.compute_liquidity_summary(
            returns=returns,
            volume_proxy=nav,
            high=None,
            low=None,
            nav=nav,
        )
        result.update({
            "source": "nav_proxy",
            "status": "proxy",
            "warning": "NAV is not market volume; do not treat this as institutional liquidity analytics.",
        })
        return result

    def compute_liquidity_metrics(
        self,
        *,
        adv: Optional[pd.DataFrame | pd.Series] = None,
        bid_ask_spread: Optional[pd.DataFrame | pd.Series] = None,
        volume: Optional[pd.DataFrame | pd.Series] = None,
        max_participation_rate: float = 0.10,
    ) -> Dict[str, Any]:
        """Compute liquidity analytics when market liquidity inputs are provided.

        If ADV is omitted, returns an explicitly labelled NAV-based proxy for
        backward compatibility.
        """
        if adv is None:
            return self.compute_nav_based_liquidity_proxy()

        if self.weights is None:
            return {"status": "error", "message": "weights are required for liquidity metrics"}

        nav = self._portfolio_data.net_asset_value
        weights = self.weights.to_frame() if isinstance(self.weights, pd.Series) else self.weights
        adv_df = adv.to_frame() if isinstance(adv, pd.Series) else adv
        adv_df = self._normalize_time_index_like(adv_df, weights.index)
        weights, adv_df = weights.align(adv_df, join="inner", axis=None)
        nav = nav.reindex(weights.index)
        position_notional = weights.abs().multiply(nav, axis=0)
        adv_safe = adv_df.replace(0.0, np.nan)
        adv_usage = (position_notional / adv_safe).replace([np.inf, -np.inf], np.nan)
        days_to_liquidate = adv_usage / max_participation_rate

        result: Dict[str, Any] = {
            "status": "success",
            "source": "adv",
            "max_position_adv_ratio": float(adv_usage.max().max()),
            "avg_position_adv_ratio": float(adv_usage.stack().mean()),
            "max_days_to_liquidate": float(days_to_liquidate.max().max()),
            "avg_days_to_liquidate": float(days_to_liquidate.stack().mean()),
            "max_participation_rate": float(max_participation_rate),
        }

        if volume is not None:
            vol_df = volume.to_frame() if isinstance(volume, pd.Series) else volume
            vol_df = self._normalize_time_index_like(vol_df, position_notional.index)
            position_notional_al, vol_df = position_notional.align(vol_df, join="inner", axis=None)
            participation = (position_notional_al / vol_df.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
            result["max_observed_participation_rate"] = float(participation.max().max())
            result["avg_observed_participation_rate"] = float(participation.stack().mean())

        if bid_ask_spread is not None:
            spread_df = bid_ask_spread.to_frame() if isinstance(bid_ask_spread, pd.Series) else bid_ask_spread
            spread_df = self._normalize_time_index_like(spread_df, position_notional.index)
            position_notional_al, spread_df = position_notional.align(spread_df, join="inner", axis=None)
            spread_cost = (position_notional_al * spread_df / 2.0).replace([np.inf, -np.inf], np.nan)
            result["estimated_spread_cost"] = float(spread_cost.sum().sum())

        return result

    # "Smart" ratios ----------------------------------------------------
    def compute_smart_sharpe_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        s = calc_risk.smart_sharpe_ratio(self.returns.to_frame(), risk_free_rate=risk_free_rate, days_per_year=self.trading_days)
        return float(s.iloc[0]) if hasattr(s, "iloc") else float(s)

    def compute_smart_sortino_ratio(self, risk_free_rate: Optional[float] = None, target_return: float = 0.0) -> float:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        s = calc_risk.smart_sortino_ratio(self.returns.to_frame(), risk_free_rate=risk_free_rate, target_return=target_return, days_per_year=self.trading_days)
        return float(s.iloc[0]) if hasattr(s, "iloc") else float(s)

    def compute_smart_calmar_ratio(self) -> float:
        s = calc_risk.smart_calmar_ratio(self.returns.to_frame(), days_per_year=self.trading_days)
        return float(s.iloc[0]) if hasattr(s, "iloc") else float(s)

    # ── Excess returns helper ─────────────────────────────────────────
    @property
    def excess_returns(self) -> pd.Series:
        return self.returns - self.risk_free_rate / self.trading_days

    # ── Annualised return ─────────────────────────────────────────────
    def compute_annualized_return(self, returns: Optional[pd.Series] = None) -> float:
        if returns is None:
            returns = self.returns
        total_return = (1 + returns).prod() - 1
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        if years <= 0:
            return np.nan
        return (1 + total_return) ** (1 / years) - 1

    # ── Monthly stats ─────────────────────────────────────────────────
    def compute_monthly_stats(self) -> Dict[str, float]:
        monthly = self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        up = monthly[monthly > 0]
        down = monthly[monthly < 0]
        return {
            "avg_up_month": up.mean() * 100 if len(up) > 0 else 0,
            "avg_down_month": down.mean() * 100 if len(down) > 0 else 0,
            "win_month_ratio": len(up) / len(monthly) * 100 if len(monthly) > 0 else 0,
        }

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
    def compute_expected_returns(self) -> Dict[str, float]:
        daily = self.returns.dropna().mean()
        return {
            "expected_daily": daily * 100,
            "expected_weekly": ((1 + daily) ** 5 - 1) * 100,
            "expected_monthly": ((1 + daily) ** 21 - 1) * 100,
            "expected_quarterly": ((1 + daily) ** 63 - 1) * 100,
            "expected_yearly": ((1 + daily) ** 252 - 1) * 100,
        }

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

    # ── Statistical moments helper ────────────────────────────────────
    def _compute_statistical_moments(self) -> Dict[str, float]:
        """Unbiased skewness and excess kurtosis via scipy."""
        r = self.metric_returns.to_numpy()
        n = len(r)
        if n < 3 or np.std(r, ddof=1) == 0:
            return {"skewness": 0.0, "kurtosis": 0.0}
        try:
            from scipy import stats as sp_stats
            skew = float(sp_stats.skew(r, bias=False))
            kurt = float(sp_stats.kurtosis(r, fisher=True, bias=False))
        except ImportError:
            mean = np.mean(r)
            std = np.std(r, ddof=1)
            m3 = np.mean((r - mean) ** 3)
            m4 = np.mean((r - mean) ** 4)
            skew = m3 / (std ** 3) * (n * (n - 1)) ** 0.5 / (n - 2) if n > 2 else 0.0
            kurt = ((m4 / (std ** 4)) - 3) * (n - 1) / ((n - 2) * (n - 3)) * (n + 1) + 6 / (n - 1) if n > 3 else 0.0
        return {"skewness": skew, "kurtosis": kurt}

    # ── Advanced Sharpe ───────────────────────────────────────────────
    def compute_advanced_sharpe_ratio(self, annualize: bool = True, benchmark_sharpe: float = 0.0) -> Dict[str, Any]:
        base = self.compute_sharpe_ratio(annualize=False)
        moments = self._compute_statistical_moments()
        skew = np.clip(moments["skewness"], -3, 3)
        kurt = np.clip(moments["kurtosis"], -10, 10)
        # Pézier-White: kurtosis term is SUBTRACTED (fat tails penalise)
        adj = 1 + (skew * base) / 6 - (kurt * base ** 2) / 24
        smart = base * adj
        ann = np.sqrt(self.trading_days) if annualize else 1.0
        return {
            "smart_sharpe": smart * ann,
            "base_sharpe": base,
            "statistics": {
                "observations": len(self.returns),
                "skewness": skew,
                "kurtosis": kurt,
                "original_sharpe_ratio": base * ann,
                "annualized": annualize,
                "adjustment_factor": float(adj),
            },
            "status": "success",
        }

    # ── Advanced Sortino ──────────────────────────────────────────────
    def compute_advanced_sortino_ratio(self, annualize: bool = True, target_return: float = 0.0, min_periods: int = 252) -> Dict[str, Any]:
        adjusted = self.returns - ((self.risk_free_rate + target_return) / self.trading_days)
        downside = adjusted[adjusted < 0]
        if len(downside) < max(min_periods // 4, 5):
            return {"status": "error", "message": "Insufficient downside observations"}
        sortino = self.compute_sortino_ratio(target_return=target_return, annualize=False)
        if not np.isfinite(sortino):
            return {"smart_sortino": np.nan, "base_sortino": np.nan, "status": "warning"}
        moments = self._compute_statistical_moments()
        skew = np.clip(moments["skewness"], -3, 3)
        kurt = np.clip(moments["kurtosis"], -10, 10)
        # Pézier-White: kurtosis term is SUBTRACTED (fat tails penalise)
        smart = sortino * (1 + (skew * sortino) / 6 - (kurt * sortino ** 2) / 24)
        ann = np.sqrt(self.trading_days) if annualize else 1.0
        if annualize:
            smart *= ann
        return {
            "smart_sortino": smart,
            "base_sortino": sortino * ann,
            "raw_base_sortino": sortino,
            "downside_risk": {
                "threshold": target_return,
                "observations": len(downside),
            },
            "status": "success",
        }

    # ── Advanced Omega ────────────────────────────────────────────────
    def compute_advanced_omega_ratio(self, threshold: float = 0.0, **kwargs) -> Dict[str, Any]:
        excess = self.returns - threshold / self.trading_days
        gains = excess[excess > 0]
        losses = excess[excess < 0]
        if len(losses) == 0:
            omega = np.inf if len(gains) > 0 else 1.0
        else:
            omega = (gains.mean() * len(gains)) / (abs(losses.mean()) * len(losses))
        return {"base_omega": omega, "status": "success"}

    # ── Advanced Calmar ───────────────────────────────────────────────
    def compute_advanced_calmar_ratio(self, **kwargs) -> Dict[str, Any]:
        ann_ret = self.compute_annualized_return()
        max_dd = abs(self.compute_max_drawdown())
        calmar = ann_ret / max_dd if max_dd != 0 else np.inf
        return {"base_calmar": calmar, "status": "success"}

    # ── Advanced Ulcer Index ──────────────────────────────────────────
    def compute_advanced_ulcer_index(self, window: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        dd = self.drawdowns
        squared_dd = np.square(dd)
        if window:
            base = np.sqrt(squared_dd.rolling(window=window, min_periods=1).mean())
        else:
            base = np.sqrt(squared_dd.mean())
        return {"base_ulcer": base, "status": "success"}

    # ── VaR ratio (Cornish-Fisher) ────────────────────────────────────
    def compute_var_ratio(self, confidence: float = 0.95, method: str = "cornish_fisher") -> Dict[str, Any]:
        r = self.metric_returns
        if len(r) == 0:
            return {
                "stats": {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan},
                "var_metrics": {},
                "regime_info": {"distribution": {"normal_skew": np.nan, "normal_kurt": np.nan}},
            }
        try:
            from scipy import stats as sp_stats
            skew_val = float(sp_stats.skew(r))
            kurt_val = float(sp_stats.kurtosis(r, fisher=True))
        except ImportError:
            moments = self._compute_statistical_moments()
            skew_val = moments["skewness"]
            kurt_val = moments["kurtosis"]

        # VaR convention: positive = loss.
        losses = -r
        hist_var = float(losses.quantile(confidence))
        gauss_var = -float(r.mean() + r.std() * np.sqrt(2) * self._erfinv(2 * (1 - confidence) - 1))

        # Cornish-Fisher — use LEFT tail (loss quantile), not right tail
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)  # e.g. norm.ppf(0.05) = -1.6449
        cf = z + (z ** 2 - 1) * skew_val / 6 + (z ** 3 - 3 * z) * kurt_val / 24
        cf_var = -(r.mean() + cf * r.std())  # positive = loss

        stress_window = min(max(20, int(self.trading_days / 4)), len(r))
        stress_var = np.nan
        stress_observations = 0
        if len(r) >= stress_window and stress_window >= 2:
            min_periods = min(stress_window, max(2, stress_window // 2))
            rolling_vol = r.rolling(stress_window, min_periods=min_periods).std()
            high_vol_cutoff = rolling_vol.quantile(0.80)
            stress_mask = rolling_vol >= high_vol_cutoff
            stress_losses = losses.loc[stress_mask.fillna(False)]
            stress_observations = int(len(stress_losses))
            if stress_observations >= 2:
                stress_var = float(stress_losses.quantile(confidence))

        def _stress_or_normal(normal_var: float) -> float:
            if not np.isfinite(normal_var):
                return float(stress_var) if np.isfinite(stress_var) else np.nan
            if np.isfinite(stress_var):
                return max(float(normal_var), float(stress_var))
            return float(normal_var)

        worst_loss = float(losses.max())
        return {
            "stats": {"mean": r.mean(), "std": r.std(), "min": r.min(), "max": r.max()},
            "var_metrics": {
                "historical": {"normal": hist_var, "stress": _stress_or_normal(hist_var), "extreme": worst_loss},
                "gaussian": {"normal": gauss_var, "stress": _stress_or_normal(gauss_var)},
                "cornish_fisher": {"normal": cf_var, "stress": _stress_or_normal(cf_var), "extreme": worst_loss},
            },
            "regime_info": {
                "distribution": {"normal_skew": skew_val, "normal_kurt": kurt_val},
                "stress_var_observations": stress_observations,
                "stress_var_basis": "top_quintile_rolling_volatility",
            },
        }

    @staticmethod
    def _erfinv(x: float) -> float:
        """Approximate inverse error function (avoids scipy for simple cases)."""
        try:
            from scipy.special import erfinv
            return float(erfinv(x))
        except ImportError:
            # Winitzki approximation
            a = 0.147
            ln1mx2 = np.log(1 - x * x)
            t = 2 / (np.pi * a) + ln1mx2 / 2
            sign = 1 if x >= 0 else -1
            return sign * np.sqrt(np.sqrt(t * t - ln1mx2 / a) - t)

    # ── CVaR ratio ────────────────────────────────────────────────────
    def compute_cvar_ratio(self, confidence: float = 0.95, method: str = "historical") -> Dict[str, Any]:
        r = self.metric_returns
        losses = -r
        var_results = self.compute_var_ratio(confidence, method)
        cvar_metrics: Dict[str, Any] = {}
        tail_observations: Dict[str, Any] = {}
        for method_name in ["historical", "gaussian", "cornish_fisher"]:
            cvar_metrics[method_name] = {}
            tail_observations[method_name] = {}
            var_vals = var_results.get("var_metrics", {}).get(method_name, {})
            for regime, var_val in var_vals.items():
                tail_losses = losses[losses >= var_val]
                cvar = float(tail_losses.mean()) if len(tail_losses) >= 2 else np.nan
                cvar_metrics[method_name][regime] = cvar
                tail_observations[method_name][regime] = int(len(tail_losses))
        return {
            "cvar_metrics": cvar_metrics,
            "tail_observations": tail_observations,
            "status": "success",
            "warning": "CVaR is NaN when there are fewer than two tail observations.",
            "tail_metrics": {
                "worst_loss": float(losses.max()) if len(losses) else np.nan,
                "worst_5_avg": float(losses.nlargest(min(5, len(losses))).mean()) if len(losses) else np.nan,
                "tail_ratio": abs(r.nsmallest(min(100, len(r))).mean() / r.nlargest(min(100, len(r))).mean()) if r.nlargest(min(100, len(r))).mean() != 0 else np.nan,
            },
        }

    # ── Drawdown durations ────────────────────────────────────────────
    def compute_drawdown_durations(self) -> Dict[str, Any]:
        dd = self.drawdowns
        is_dd = dd < 0
        durations = []
        cur = 0
        in_dd = False
        for v in is_dd:
            if v:
                cur += 1
                in_dd = True
            elif in_dd:
                durations.append(cur)
                cur = 0
                in_dd = False
        if in_dd:
            durations.append(cur)
        return {
            "avg_duration": np.mean(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "current_duration": cur,
            "total_drawdowns": len(durations),
        }

    # ── Recovery factor ───────────────────────────────────────────────
    def compute_recovery_factor(self) -> float:
        max_dd = abs(self.compute_max_drawdown())
        if max_dd == 0:
            return np.inf
        return self.compute_annualized_return() / max_dd

    # ── Serenity index ────────────────────────────────────────────────
    def compute_serenity_index(self) -> float:
        excess_return = self.returns.mean() * self.trading_days
        pain = abs(self.drawdowns.mean())
        return excess_return / pain if pain != 0 else np.nan

    # ── Win percentages ───────────────────────────────────────────────
    def compute_win_percentages(self) -> Dict[str, float]:
        r = self.metric_returns
        total = len(r)
        if total == 0:
            return {
                "win_rate": np.nan, "loss_rate": np.nan, "flat_rate": np.nan,
                "win_days": 0, "loss_days": 0, "flat_days": 0, "total_days": 0,
            }
        wins = (r > 0).sum()
        losses = (r < 0).sum()
        flats = (r == 0).sum()
        return {
            "win_rate": wins / total, "loss_rate": losses / total, "flat_rate": flats / total,
            "win_days": wins, "loss_days": losses, "flat_days": flats, "total_days": total,
        }

    # ── Gain to pain ratio ────────────────────────────────────────────
    def compute_gain_to_pain_ratio(self) -> float:
        pos = self.returns[self.returns > 0].sum()
        neg = abs(self.returns[self.returns < 0].sum())
        if neg == 0:
            return np.inf if pos > 0 else np.nan
        return pos / neg

    # ── Payoff ratio ──────────────────────────────────────────────────
    def compute_payoff_ratio(self) -> float:
        avg_win = self.returns[self.returns > 0].mean()
        avg_loss = abs(self.returns[self.returns < 0].mean())
        if avg_loss == 0:
            return np.inf if avg_win > 0 else np.nan
        return avg_win / avg_loss

    # ── Profit factor ─────────────────────────────────────────────────
    def compute_profit_factor(self) -> float:
        gross_profit = (self.returns[self.returns > 0] * 100).sum()
        gross_loss = abs((self.returns[self.returns < 0] * 100).sum())
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else np.nan
        return gross_profit / gross_loss

    # ── Drawdown-vol adjusted return ──────────────────────────────────
    def compute_drawdown_vol_adjusted_return(self) -> float:
        r = self.metric_returns
        ann_ret = r.mean() * self.trading_days
        cum = (1 + r).cumprod()
        dd = cum / cum.expanding().max() - 1
        dd_vol = dd.std() * np.sqrt(self.trading_days)
        return ann_ret / dd_vol if dd_vol != 0 else np.nan

    def compute_common_sense_ratio(self) -> float:
        """Backward-compatible alias for a non-standard drawdown-vol ratio."""
        return self.compute_drawdown_vol_adjusted_return()

    # ── Max win streak per win cluster ────────────────────────────────
    def compute_max_win_streak_per_win_cluster(self) -> float:
        streak = 0
        max_streak = 0
        total = 0
        for r in self.metric_returns:
            if r > 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                if streak > 0:
                    total += 1
                streak = 0
        return max_streak / total if total > 0 else 0

    def compute_cpc_index(self) -> float:
        """Backward-compatible alias for win-streak concentration diagnostic."""
        return self.compute_max_win_streak_per_win_cluster()

    # ── Kelly criterion ───────────────────────────────────────────────
    def compute_kelly_criterion(self) -> float:
        r = self.metric_returns
        if len(r) == 0:
            return np.nan
        w = (r > 0).mean()
        avg_win = r[r > 0].mean()
        avg_loss = abs(r[r < 0].mean())
        if avg_loss == 0:
            return np.nan
        return w - ((1 - w) / (avg_win / avg_loss))

    # ── Consecutive wins / losses ─────────────────────────────────────
    def compute_max_consecutive_wins_losses(self) -> Dict[str, int]:
        cw = cl = mw = ml = 0
        for r in self.returns:
            if r > 0:
                cw += 1; cl = 0; mw = max(mw, cw)
            elif r < 0:
                cl += 1; cw = 0; ml = max(ml, cl)
            else:
                cw = cl = 0
        return {
            "max_consecutive_wins": mw, "max_consecutive_losses": ml,
            "current_win_streak": cw, "current_loss_streak": cl,
        }

    # ── Average win / loss ────────────────────────────────────────────
    def compute_average_win_loss(self) -> Dict[str, float]:
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        return {
            "avg_win": wins.mean() if len(wins) > 0 else 0,
            "avg_loss": losses.mean() if len(losses) > 0 else 0,
        }

    # ── Time in market ────────────────────────────────────────────────
    def compute_time_in_market(self) -> Dict[str, float]:
        threshold = 1e-8
        if self.weights is not None:
            w = self.weights.to_frame() if isinstance(self.weights, pd.Series) else self.weights
            gross_exposure = w.abs().sum(axis=1)
        elif self._portfolio_data.positions is not None:
            pos = self._portfolio_data.positions
            nav = self._portfolio_data.net_asset_value.reindex(pos.index)
            gross_exposure = pos.abs().sum(axis=1).divide(nav.replace(0.0, np.nan))
        else:
            return {
                "status": "error",
                "message": "No weights or position values available",
                "time_invested": 0.0,
                "time_in_cash": 0.0,
                "invested_days": 0,
                "cash_days": 0,
                "total_days": 0,
            }

        gross_exposure = gross_exposure.replace([np.inf, -np.inf], np.nan).dropna()
        total = len(gross_exposure)
        if total == 0:
            return {
                "status": "error",
                "message": "No exposure observations available",
                "time_invested": 0.0,
                "time_in_cash": 0.0,
                "invested_days": 0,
                "cash_days": 0,
                "total_days": 0,
            }
        invested = int((gross_exposure > threshold).sum())
        cash = int(total - invested)
        return {
            "time_invested": invested / total, "time_in_cash": cash / total,
            "invested_days": invested, "cash_days": cash, "total_days": total,
            "source": "gross_exposure",
        }

    # ── Directionality ────────────────────────────────────────────────
    def compute_directionality(self) -> Dict[str, float]:
        r = self.returns
        ma50 = r.rolling(window=50).mean()
        ma200 = r.rolling(window=200).mean()
        pos = (r > 0) & (r.shift(1) > 0)
        neg = (r < 0) & (r.shift(1) < 0)
        return {
            "trend_strength": (ma50 > ma200).mean(),
            "directional_consistency": (pos | neg).mean(),
            "positive_momentum": pos.mean(),
            "negative_momentum": neg.mean(),
            "direction_changes": (r.shift(1) * r < 0).mean(),
        }

    # ── Parity distance ───────────────────────────────────────────────
    def compute_parity_distance(self) -> Dict[str, float]:
        cum = (1 + self.returns).cumprod()
        dist = abs(cum - 1)
        return {
            "avg_distance": dist.mean(), "max_distance": dist.max(),
            "current_distance": dist.iloc[-1], "distance_volatility": dist.std(),
        }

    # ── Average holding period ────────────────────────────────────────
    def compute_average_holding_period(
        self,
        trades_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Average holding period from blotter round-trips.

        Holding period is a lot/accounting concept; returns-based fallbacks are
        intentionally not used because they confuse flat P&L days with cash.
        """
        # Preferred path: actual round-trip matching
        if trades_df is not None and not trades_df.empty:
            from backtester.portfolio.calc.round_trips import RoundTripAnalyzer
            analyzer = RoundTripAnalyzer(
                trades_df,
                self.returns,
                initial_capital=float(self._portfolio_data.net_asset_value.iloc[0]),
            )
            hp = analyzer._holding_period_stats()
            return {
                "avg_holding_period": hp["avg_holding_days"],
                "median_holding_period": hp["median_holding_days"],
                "max_holding_period": hp["max_holding_days"],
                "min_holding_period": hp["min_holding_days"],
                "avg_holding_winners": hp["avg_holding_winners"],
                "avg_holding_losers": hp["avg_holding_losers"],
                "source": "blotter_fifo",
            }

        return {
            "status": "unavailable",
            "message": "Holding period requires blotter/FIFO round-trip data",
            "avg_holding_period": 0.0,
            "median_holding_period": 0.0,
            "max_holding_period": 0.0,
            "min_holding_period": 0.0,
            "avg_holding_winners": 0,
            "avg_holding_losers": 0,
            "source": "none",
        }

    # ── Weight summary ────────────────────────────────────────────────
    def compute_weight_summary(self) -> Dict[str, Any]:
        if self.weights is None:
            return {}
        weights_frame = self.weights.to_frame() if isinstance(self.weights, pd.Series) else self.weights
        w = weights_frame.iloc[-1]
        nz = w[w != 0]
        gross_exposure = w.abs().sum()
        avg_abs = weights_frame.abs().mean(axis=0)
        denom = float(avg_abs.sum())
        effective_positions = np.nan
        if denom > 0:
            norm = avg_abs / denom
            concentration = float((norm ** 2).sum())
            effective_positions = (1.0 / concentration) if concentration > 0 else np.nan
        return {
            "avg_weight": w.mean(),
            "active_positions": (w != 0).sum(),
            "concentration": (w ** 2).sum(),
            "raw_concentration": (w.abs() / gross_exposure).pow(2).sum() if gross_exposure > 0 else np.nan,
            "min_weight": nz.min() if len(nz) > 0 else 0,
            "max_weight": w.max(),
            "weight_std": w.std(),
            "long_exposure": w[w > 0].sum(),
            "short_exposure": abs(w[w < 0].sum()),
            "gross_exposure": gross_exposure,
            "net_exposure": w.sum(),
            "cash_sleeve": 1.0 - w.sum(),
            "effective_positions": effective_positions,
        }

    def compute_exposure_path_checks(self) -> Dict[str, Any]:
        """Exposure diagnostics across the full weight path."""
        if self.weights is None:
            return {
                "status": "unavailable",
                "message": "No weights available",
                "observations": 0,
            }

        w = self.weights.to_frame() if isinstance(self.weights, pd.Series) else self.weights
        w = w.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
        if w.empty:
            return {
                "status": "unavailable",
                "message": "No finite weight observations",
                "observations": 0,
            }

        gross = w.abs().sum(axis=1)
        long_exposure = w.clip(lower=0).sum(axis=1)
        short_exposure = w.clip(upper=0).abs().sum(axis=1)
        net_exposure = w.sum(axis=1)
        cash_sleeve = 1.0 - net_exposure

        return {
            "status": "success",
            "observations": int(len(w)),
            "max_gross_exposure": float(gross.max()),
            "avg_gross_exposure": float(gross.mean()),
            "max_long_exposure": float(long_exposure.max()),
            "max_short_exposure": float(short_exposure.max()),
            "min_cash_sleeve": float(cash_sleeve.min()),
            "max_cash_sleeve": float(cash_sleeve.max()),
            "min_net_exposure": float(net_exposure.min()),
            "max_net_exposure": float(net_exposure.max()),
            "cash_sleeve_nonnegative": bool(cash_sleeve.min() >= -1e-8),
            "finite_weights": bool(np.isfinite(w.to_numpy()).all()),
            "source": "portfolio_weights",
        }

    # ── Statistical metrics ───────────────────────────────────────────
    def compute_statistical_metrics(self) -> Dict[str, Any]:
        r = self.metric_returns
        moments = self._compute_statistical_moments()
        sorted_r = r.sort_values()
        n100 = min(100, len(r))
        tail_ratio = abs(sorted_r.tail(n100).mean() / sorted_r.head(n100).mean()) if sorted_r.head(n100).mean() != 0 else np.nan
        effective_exposure_breadth = np.nan
        if self.weights is not None:
            w = self.weights.to_frame() if isinstance(self.weights, pd.Series) else self.weights
            avg_abs = w.abs().mean(axis=0)
            denom = float(avg_abs.sum())
            if denom > 0:
                norm = avg_abs / denom
                concentration = float((norm ** 2).sum())
                effective_exposure_breadth = (1.0 / concentration) if concentration > 0 else np.nan
        return {
            "skewness": moments["skewness"],
            "kurtosis": moments["kurtosis"],
            "tail_ratio": tail_ratio,
            "effective_exposure_breadth": (
                float(effective_exposure_breadth)
                if np.isfinite(effective_exposure_breadth)
                else np.nan
            ),
            # Backward-compatible key; this is HHI-based exposure breadth,
            # not correlation-adjusted independent bets.
            "effective_bets": (
                float(effective_exposure_breadth)
                if np.isfinite(effective_exposure_breadth)
                else np.nan
            ),
            "observations": float(len(r)),
        }

    # ── Advanced turnover ─────────────────────────────────────────────
    def compute_advanced_turnover(
        self,
        trades_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Turnover analytics.

        If *trades_df* is provided the turnover is **dollar-based**
        (total notional traded / 2 / avg NAV), which is the correct
        institutional definition.  Falls back to weight-diff method
        when no blotter is available.
        """
        # Preferred: dollar-based from actual trades
        if trades_df is not None and not trades_df.empty:
            from backtester.portfolio.calc.round_trips import RoundTripAnalyzer
            analyzer = RoundTripAnalyzer(
                trades_df,
                self.returns,
                initial_capital=float(self._portfolio_data.net_asset_value.iloc[0]),
            )
            vol = analyzer._volume_and_turnover()
            total_notional = float(vol["total_volume"])
            avg_nav = float(self._portfolio_data.net_asset_value.mean())
            total_turnover_ratio = (total_notional / 2.0 / avg_nav) if avg_nav > 0 else 0.0
            return {
                "total_traded_notional": total_notional,
                "total_turnover_ratio": total_turnover_ratio,
                "total_turnover_pct": total_turnover_ratio * 100,
                "average_turnover": vol["daily_turnover_pct"],
                "avg_daily_turnover_pct": vol["daily_turnover_pct"],
                "annualized_turnover": vol["annualized_turnover_pct"],
                "annualized_turnover_pct": vol["annualized_turnover_pct"],
                "max_turnover": np.nan,
                "daily_turnover": vol["daily_turnover_pct"],
                "turnover_std": np.nan,
                "avg_trade_size": vol["avg_trade_size"],
                # Backward-compatible key, now explicitly a percent ratio.
                "total_turnover": total_turnover_ratio * 100,
                "source": "blotter_notional",
            }

        # Fallback: weight-diff half-turnover
        if self.weights is None:
            return {"status": "error", "message": "No weights data"}
        w = self.weights
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

    # ── Rolling turnover ──────────────────────────────────────────────
    def compute_rolling_turnover(self, window: int = 252) -> Dict[str, pd.Series]:
        if self.weights is None:
            return {"status": "error", "message": "No weights data"}
        w = self.weights
        if isinstance(w, pd.DataFrame):
            turnover = w.diff().abs().sum(axis=1) / 2
        else:
            turnover = w.diff().abs() / 2
        turnover = turnover.fillna(0.0)
        return {
            "rolling_turnover": turnover.rolling(window=window, min_periods=1).sum() * 100,
            "rolling_turnover_rate": turnover.rolling(window=window, min_periods=1).mean() * 100,
        }

    # ── Benchmark metrics ─────────────────────────────────────────────
    def align_benchmark_returns(
        self,
        benchmark_returns: pd.Series,
        *,
        min_overlap: float = 0.0,
        require_same_frequency: bool = False,
        benchmark_return_type: Literal["price_index", "total_return", "excess_return"] = "price_index",
    ) -> Dict[str, Any]:
        """Align portfolio and benchmark returns with explicit overlap metadata."""
        portfolio_returns = self.returns.copy()
        benchmark_returns = self._normalize_time_index_like(
            benchmark_returns,
            portfolio_returns.index,
        )
        combined = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        })
        union_count = int(len(combined))
        aligned = combined.dropna()
        overlap_count = int(len(aligned))
        overlap_ratio = (overlap_count / union_count) if union_count else 0.0
        if overlap_ratio < min_overlap:
            raise ValueError(
                f"Benchmark overlap {overlap_ratio:.2%} is below required {min_overlap:.2%}"
            )

        frequency_match = True
        if require_same_frequency and overlap_count >= 3:
            p_diffs = aligned["portfolio"].index.to_series().diff().dropna()
            b_diffs = benchmark_returns.dropna().index.to_series().diff().dropna()
            if len(p_diffs) and len(b_diffs):
                frequency_match = p_diffs.mode().iloc[0] == b_diffs.mode().iloc[0]
            if not frequency_match:
                raise ValueError("Portfolio and benchmark return frequency mismatch")

        return {
            "portfolio_returns": aligned["portfolio"],
            "benchmark_returns": aligned["benchmark"],
            "overlap_count": overlap_count,
            "union_count": union_count,
            "overlap_ratio": overlap_ratio,
            "frequency_match": frequency_match,
            "benchmark_return_type": benchmark_return_type,
        }

    def compute_alpha(self, benchmark_returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        aligned_meta = self.align_benchmark_returns(
            benchmark_returns,
            min_overlap=0.95,
            require_same_frequency=True,
        )
        r = aligned_meta["portfolio_returns"]
        b = aligned_meta["benchmark_returns"]
        if len(r) < 2:
            return np.nan
        excess_r = r - risk_free_rate / self.trading_days
        excess_b = b - risk_free_rate / self.trading_days
        cov_rb = np.cov(excess_r, excess_b)
        beta = cov_rb[0, 1] / cov_rb[1, 1] if cov_rb[1, 1] != 0 else 0
        alpha = excess_r.mean() - beta * excess_b.mean()
        return alpha * self.trading_days

    def compute_beta(self, benchmark_returns: pd.Series) -> float:
        aligned_meta = self.align_benchmark_returns(
            benchmark_returns,
            min_overlap=0.95,
            require_same_frequency=True,
        )
        r = aligned_meta["portfolio_returns"]
        b = aligned_meta["benchmark_returns"]
        if len(r) < 2:
            return np.nan
        cov = np.cov(r, b)
        return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan

    def compute_r_squared(self, benchmark_returns: pd.Series) -> float:
        aligned_meta = self.align_benchmark_returns(
            benchmark_returns,
            min_overlap=0.95,
            require_same_frequency=True,
        )
        r = aligned_meta["portfolio_returns"]
        b = aligned_meta["benchmark_returns"]
        if len(r) < 2:
            return np.nan
        corr = np.corrcoef(r, b)[0, 1]
        return corr ** 2

    def compute_tracking_error(self, benchmark_returns: pd.Series) -> float:
        aligned_meta = self.align_benchmark_returns(
            benchmark_returns,
            min_overlap=0.95,
            require_same_frequency=True,
        )
        r = aligned_meta["portfolio_returns"]
        b = aligned_meta["benchmark_returns"]
        if len(r) < 2:
            return np.nan
        return np.std(r - b) * np.sqrt(self.trading_days)

    def compute_up_capture_ratio(self, benchmark_returns: pd.Series) -> float:
        aligned_meta = self.align_benchmark_returns(
            benchmark_returns,
            min_overlap=0.95,
            require_same_frequency=True,
        )
        r = aligned_meta["portfolio_returns"]
        b = aligned_meta["benchmark_returns"]
        if len(r) < 2:
            return np.nan
        up = b > 0
        if not up.any():
            return np.nan
        n_up = up.sum()
        port_up = (1 + r[up]).prod() ** (self.trading_days / n_up) - 1
        bench_up = (1 + b[up]).prod() ** (self.trading_days / n_up) - 1
        return port_up / bench_up if bench_up != 0 else np.nan

    def compute_down_capture_ratio(self, benchmark_returns: pd.Series) -> float:
        aligned_meta = self.align_benchmark_returns(
            benchmark_returns,
            min_overlap=0.95,
            require_same_frequency=True,
        )
        r = aligned_meta["portfolio_returns"]
        b = aligned_meta["benchmark_returns"]
        if len(r) < 2:
            return np.nan
        down = b < 0
        if not down.any():
            return np.nan
        n_down = down.sum()
        port_down = (1 + r[down]).prod() ** (self.trading_days / n_down) - 1
        bench_down = (1 + b[down]).prod() ** (self.trading_days / n_down) - 1
        return port_down / bench_down if bench_down != 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════
# Standalone trade-level analytics (uses blotter data, not returns)
# Delegates to RoundTripAnalyzer for canonical FIFO matching.
# ═══════════════════════════════════════════════════════════════════════

def compute_trade_analytics(
    trades_df: pd.DataFrame,
    returns: pd.Series,
    initial_capital: float,
) -> Dict[str, Any]:
    """Compute trade-level metrics from blotter data.

    Delegates to ``RoundTripAnalyzer`` so that round-trips, holding
    periods, turnover and win/loss stats all come from the same FIFO
    matching engine — guaranteeing mutual consistency.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Blotter trades with columns: Instrument, Side, Quantity, Price,
        TradeValue, TransactionCost, Timestamp.
    returns : pd.Series
        Daily portfolio returns (for NAV / drawdown in $).
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    dict with keys usable via dot-path in PORTFOLIO_PERF_METRICS.
    """
    from backtester.portfolio.calc.round_trips import RoundTripAnalyzer

    analyzer = RoundTripAnalyzer(trades_df, returns, initial_capital)
    return analyzer.summary()
