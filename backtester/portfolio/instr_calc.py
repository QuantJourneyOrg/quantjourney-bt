# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

from typing import Optional, Dict, Union
import warnings

import numpy as np
import pandas as pd

from backtester.portfolio.instr_data import InstrumentData
from backtester.portfolio.config import CalcConfig, get_default_config
from backtester.portfolio._compat import TimePeriod

from backtester.portfolio.calc import returns as calc_returns
from backtester.portfolio.calc import risk as calc_risk
from backtester.portfolio.calc import exposures as calc_exposures
from backtester.portfolio.calc import attribution as calc_attr
from backtester.portfolio.calc import sampling as calc_sampling
from backtester.portfolio.calc import outliers as calc_outliers
from backtester.portfolio.calc import rolling_stats as calc_roll
from backtester.portfolio.calc import scenario as calc_scenario
from backtester.portfolio.calc import pnl as calc_pnl
from backtester.portfolio.calc import metrics as calc_metrics
from backtester.portfolio._compat import ReturnTypes, Reporting, ReportingParams
from backtester.portfolio.models.results import ReturnsSummary


class InstrumentCalculations:
    """
    Facade class for instrument analytics. Performs alignment/orchestration and
    delegates numerical work to calc modules.
    """

    def __init__(self, instrument_data: InstrumentData, config: Optional[CalcConfig] = None):
        self._instrument_data = instrument_data
        self._config: CalcConfig = config or get_default_config()

    # Accessors --------------------------------------------------------
    @property
    def prices(self) -> pd.DataFrame:
        return self._instrument_data.prices

    @property
    def returns(self) -> pd.DataFrame:
        return self._instrument_data.get_feature("metrics", level="returns")

    @property
    def units(self) -> pd.DataFrame:
        """Backward-compatible alias for position_units.

        In QuantJourney data contracts ``units`` means executed position
        quantity/shares/contracts. It must not be used as portfolio weights.
        """
        return self.position_units

    @property
    def position_units(self) -> pd.DataFrame:
        """Executed position quantities per instrument."""
        return self._instrument_data.get_feature("parameters", level="units")

    @property
    def weights(self) -> pd.DataFrame:
        """Portfolio weights per instrument.

        Weights are distinct from position units. They may be stored as an
        optional ``parameters/weights`` frame by callers that want instrument
        analytics to compute weight-based attribution.
        """
        params = getattr(self._instrument_data, "parameters", pd.DataFrame())
        if (
            params is not None
            and isinstance(params, pd.DataFrame)
            and isinstance(params.columns, pd.MultiIndex)
            and "weights" in params.columns.get_level_values(-1)
        ):
            return params.xs("weights", axis=1, level=-1)
        raise ValueError(
            "weights are required for this calculation; pass weights explicitly "
            "or store them under InstrumentData.parameters level='weights'. "
            "Do not use units as weights."
        )

    @property
    def instruments(self) -> list[str]:
        return self._instrument_data.group_data.index.tolist()

    @property
    def data_index(self) -> pd.DatetimeIndex:
        return self.prices.index

    # Generic ----------------------------------------------------------
    def compute_position_values(self) -> pd.DataFrame:
        """Return per-instrument market value: position units/shares times price."""
        prices_close = self.prices.xs("adj_close", axis=1, level=1)
        return calc_exposures.compute_exposures(prices_close, self.position_units)

    def compute_nav(self) -> pd.DataFrame:
        """Deprecated alias for compute_position_values().

        This does not include cash, fees, financing, FX, or short proceeds, so
        it is not portfolio NAV. Portfolio-level NAV lives on PortfolioData.
        """
        warnings.warn(
            "InstrumentCalculations.compute_nav() is deprecated because it "
            "returns per-instrument position values, not full portfolio NAV. "
            "Use compute_position_values() or PortfolioData.net_asset_value.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_position_values()

    def compute_short_long_exposure(self) -> pd.DataFrame:
        prices_close = self.prices.xs("adj_close", axis=1, level=1)
        return calc_exposures.compute_short_long_exposure(prices_close, self.position_units)

    def compute_exposures(self, time_period: Optional[TimePeriod] = None, add_total: bool = True) -> pd.DataFrame:
        prices_close = self.prices.xs("adj_close", axis=1, level=1)
        exposures = calc_exposures.compute_exposures(prices_close, self.position_units)
        if time_period:
            exposures = time_period.locate(exposures)
        if add_total:
            exposures["Total"] = exposures.sum(axis=1)
        return exposures

    def compute_cumulative_returns(self) -> pd.DataFrame:
        # Cumulative daily contributions ret * lagged units
        return calc_pnl.compute_cumulative_returns_from_units(self.returns, self.position_units)

    def compute_cumulative_pnl(self) -> pd.DataFrame:
        return calc_pnl.compute_cumulative_pnl(self.returns, self.position_units)

    def compute_realized_pnl(self, time_period: Optional[TimePeriod] = None) -> pd.DataFrame:
        pnl = self.compute_cumulative_pnl()
        return time_period.locate(pnl) if time_period else pnl

    def compute_transaction_costs(self, time_period: Optional[TimePeriod] = None) -> pd.DataFrame:
        costs = self._instrument_data.get_feature("metrics", level="transaction_costs")
        return time_period.locate(costs) if time_period else costs

    # Returns & NAV ----------------------------------------------------
    def compute_returns(self, time_period: Optional[TimePeriod] = None) -> pd.DataFrame:
        r = self.returns
        return time_period.locate(r) if time_period else r

    def compute_periodic_returns(
        self,
        *,
        is_log_returns: bool = False,
        return_type: ReturnTypes = ReturnTypes.RELATIVE,
        freq: Optional[str] = None,
        include_start_date: bool = False,
        include_end_date: bool = False,
        ffill_nans: bool = True,
        drop_first: bool = False,
        is_first_zero: bool = False,
    ) -> pd.DataFrame:
        prices = self.resample_prices_at_frequency(
            freq=freq,
            include_start_date=include_start_date,
            include_end_date=include_end_date,
            ffill_nans=ffill_nans,
        )
        return calc_returns.compute_periodic_returns(
            prices,
            is_log_returns=is_log_returns,
            return_type=return_type,
            freq=None,
            include_start_date=False,
            include_end_date=False,
            ffill_nans=ffill_nans,
            drop_first=drop_first,
            is_first_zero=is_first_zero,
        )

    def compute_total_returns(self) -> pd.Series:
        return calc_returns.compute_total_returns(self.returns)

    def compute_annualized_returns(self, annualize_less_1y: bool = False) -> pd.Series:
        # annualize_less_1y preserved for signature; compute_annualized_returns handles guard
        return calc_returns.compute_annualized_returns(self.returns, days_per_year=self._config.days_per_year)

    def compute_excess_returns(self, benchmark_returns: pd.DataFrame) -> pd.DataFrame:
        return calc_metrics.excess_returns(self.returns, benchmark_returns)

    def compute_annualized_excess_returns(
        self,
        rates_data: Optional[pd.Series],
        first_date: Optional[pd.Timestamp] = None,
        annualize_less_1y: bool = False,
    ) -> pd.Series:
        if rates_data is None or len(rates_data) == 0:
            return self.compute_annualized_returns(annualize_less_1y=annualize_less_1y)
        rf = rates_data.reindex(self.returns.index).ffill()
        rf_daily = rf / float(self._config.days_per_year)
        excess = self.returns.subtract(rf_daily, axis=0)
        return calc_returns.compute_annualized_returns(excess, days_per_year=self._config.days_per_year)

    def compute_pnl(self) -> pd.DataFrame:
        return calc_pnl.compute_pnl(self.returns, self.position_units)

    def compute_active_return(self, benchmark_returns: pd.Series) -> pd.Series:
        return calc_metrics.active_return(self.returns, benchmark_returns)

    def compute_returns_summary(
        self, reporting_params: Optional[ReportingParams] = None, annualize_less_1y: bool = False
    ) -> Dict[str, np.ndarray]:
        # Minimal, preserving expected keys used by plots/reporting
        if self.prices.empty:
            n = len(self.instruments)
            return {
                Reporting.TOTAL_RETURN.to_string(): np.full(n, np.nan),
                Reporting.NUM_YEARS.to_string(): np.full(n, np.nan),
            }
        ann = self.compute_annualized_returns(annualize_less_1y=annualize_less_1y)
        total = self.compute_total_returns()
        years = self.compute_duration_in_years(self._config.days_per_year)
        return {
            Reporting.ANNUAL_RETURN.to_string(): ann.to_numpy(),
            Reporting.TOTAL_RETURN.to_string(): total.to_numpy(),
            Reporting.NUM_YEARS.to_string(): np.full(len(self.instruments), years),
        }

    # Strongly-typed variant
    def summarize_returns(self) -> ReturnsSummary:
        ann = self.compute_annualized_returns()
        total = self.compute_total_returns()
        years = self.compute_duration_in_years(self._config.days_per_year)
        return ReturnsSummary(annualized_return=ann, total_return=total, num_years=years)

    def summarize_risk(self, confidence: float = 0.95):
        """
        Return a typed RiskSummary for common risk stats per instrument.
        """
        from backtester.portfolio.models.results import RiskSummary

        vol_ann = self.compute_annualized_volatility()
        var = self.compute_var(confidence=confidence)
        cvar = self.compute_cvar(confidence=confidence)
        mdd = self.compute_max_drawdown()
        dd = self.compute_downside_deviation()
        return RiskSummary(
            volatility_annualized=vol_ann,
            var=var,
            cvar=cvar,
            max_drawdown=mdd,
            downside_deviation=dd,
        )

    def rolling_bundle(
        self,
        *,
        periods: int = 60,
        risk_free_rate: float = 0.0,
    ):
        """
        Return a typed RollingBundle with rolling mean/volatility/sharpe.
        """
        from backtester.portfolio.models.results import RollingBundle

        rmean = calc_roll.rolling_mean(self.returns, window=periods)
        rvol = calc_roll.rolling_volatility(self.returns, window=periods)
        rsr = calc_roll.rolling_sharpe_ratio(
            self.returns,
            risk_free_rate=risk_free_rate,
            window=periods,
            days_per_year=self._config.days_per_year,
        )
        return RollingBundle(
            rolling_mean=rmean,
            rolling_volatility=rvol,
            rolling_sharpe=rsr,
        )

    # Volatility and Risk ---------------------------------------------
    def compute_volatility(self, window: int = 252) -> pd.DataFrame:
        return calc_risk.compute_volatility(self.returns, window=window, days_per_year=self._config.days_per_year)

    def compute_sampled_volatility(
        self,
        freq_vol: str = "M",
        freq_return: Optional[str] = None,
        include_start_date: bool = False,
        include_end_date: bool = False,
    ) -> pd.DataFrame:
        # Delegate to risk.sampled_volatility (compounds if freq_return provided)
        return calc_risk.sampled_volatility(
            self.returns,
            freq_vol=freq_vol,
            freq_return=freq_return,
            days_per_year=self._config.days_per_year,
        )

    def compute_annualized_volatility(self) -> pd.Series:
        return calc_metrics.annualized_volatility(self.returns, days_per_year=self._config.days_per_year)

    def compute_unit_turnover(self, add_total: bool = False) -> pd.DataFrame:
        """Position-unit turnover: absolute change in executed quantities."""
        return self.position_units.diff().abs().fillna(0.0).pipe(
            lambda df: df.assign(Total=df.sum(axis=1)) if add_total else df
        )

    def compute_notional_turnover(self, add_total: bool = False) -> pd.DataFrame:
        """Dollar turnover per instrument: abs(delta units) times price."""
        prices_close = self.prices.xs("adj_close", axis=1, level=1)
        return calc_exposures.compute_turnover(
            self.position_units,
            instruments=self.instruments,
            add_total=add_total,
            prices=prices_close,
        ).fillna(0.0)

    def compute_weight_turnover(self, weights: Optional[pd.DataFrame] = None, add_total: bool = False) -> pd.DataFrame:
        """Weight turnover per instrument: abs(delta weights) / 2."""
        w = self.weights if weights is None else weights
        w = w.reindex(index=self.returns.index, columns=self.returns.columns)
        turnover = w.diff().abs().fillna(0.0) / 2.0
        if add_total:
            turnover["Total"] = turnover.sum(axis=1)
        return turnover

    def compute_turnover(self, add_total: bool = False) -> pd.DataFrame:
        """Backward-compatible unit turnover.

        For institutional portfolio turnover use PortfolioCalculations or
        compute_notional_turnover()/compute_weight_turnover() explicitly.
        """
        return self.compute_unit_turnover(add_total=add_total)

    def compute_downside_deviation(self, target_return: float = 0) -> pd.Series:
        return calc_risk.downside_deviation(self.returns, target_return=target_return)

    def compute_var(self, sigma: float = 1, confidence: float = 0.95) -> pd.Series:
        return calc_risk.compute_var(self.returns, confidence=confidence)

    def compute_cvar(self, sigma: float = 1, confidence: float = 0.95) -> pd.Series:
        return calc_risk.compute_cvar(self.returns, confidence=confidence)

    def compute_expected_shortfall(self, confidence: float = 0.95) -> pd.Series:
        return calc_risk.compute_expected_shortfall(self.returns, confidence=confidence)

    def compute_drawdowns(self) -> pd.DataFrame:
        return calc_risk.compute_drawdowns(self.returns)

    def compute_max_drawdown(self) -> pd.Series:
        return calc_risk.compute_max_drawdown(self.returns)

    def compute_max_drawdown_duration(self) -> pd.Series:
        return calc_risk.max_drawdown_duration(self.returns)

    def compute_conditional_drawdown_at_risk(self, confidence: float = 0.95) -> float:
        return calc_risk.conditional_drawdown_at_risk(self.returns, confidence=confidence)

    def compute_pain_index(self) -> pd.Series:
        return calc_risk.pain_index(self.returns)

    def _resolve_risk_free_rate(self, risk_free_rate: Optional[float]) -> float:
        if risk_free_rate is not None:
            return float(risk_free_rate)
        if self._config.risk_free_rate_annual is not None:
            return float(self._config.risk_free_rate_annual)
        return 0.0

    def compute_sharpe_ratio(self, risk_free_rate: Optional[float] = None, annualize: bool = True) -> pd.Series:
        rf = self._resolve_risk_free_rate(risk_free_rate)
        return calc_risk.sharpe_ratio(self.returns, risk_free_rate=rf, days_per_year=self._config.days_per_year, annualize=annualize)

    def compute_sortino_ratio(self, risk_free_rate: Optional[float] = None, target_return: float = 0) -> pd.Series:
        rf = self._resolve_risk_free_rate(risk_free_rate)
        return calc_risk.sortino_ratio(self.returns, risk_free_rate=rf, target_return=target_return, days_per_year=self._config.days_per_year, annualize=True)

    def compute_information_ratio(self, benchmark_returns: pd.Series) -> pd.Series:
        return calc_risk.information_ratio(self.returns, benchmark_returns, days_per_year=self._config.days_per_year)

    def compute_calmar_ratio(self) -> pd.Series:
        return calc_risk.calmar_ratio(self.returns, days_per_year=self._config.days_per_year)

    def compute_omega_ratio(self, threshold: float = 0) -> pd.Series:
        return calc_risk.omega_ratio(self.returns, threshold=threshold)

    def compute_ulcer_index(self) -> pd.Series:
        return calc_risk.ulcer_index(self.returns)

    def compute_serenity_index(self, rf: float = 0) -> pd.Series:
        return calc_risk.serenity_index(self.returns, rf=rf)

    def compute_gain_to_pain_ratio(self) -> pd.Series:
        return calc_risk.gain_to_pain_ratio(self.returns)

    def compute_upside_potential_ratio(self, target_return: float = 0) -> pd.Series:
        return calc_risk.upside_potential_ratio(self.returns, target_return=target_return)

    def compute_risk_of_ruin(self) -> pd.Series:
        return calc_risk.risk_of_ruin(self.returns)

    def compute_information_coefficient(self, predicted_returns) -> float:
        return calc_metrics.information_coefficient(self.returns, predicted_returns)

    # Attribution & Participation -------------------------------------
    def compute_factor_exposures(self, factor_returns) -> pd.DataFrame:
        ret = self.returns
        return calc_attr.compute_factor_exposures_ols(ret, factor_returns, add_intercept=False)

    def compute_factor_attribution(self, factor_returns, factor_exposures) -> pd.DataFrame:
        return calc_attr.compute_factor_attribution(factor_returns, factor_exposures)

    def compute_marginal_var(self, confidence: float = 0.95) -> pd.Series:
        warnings.warn(
            "compute_marginal_var() is deprecated; the legacy implementation "
            "is a tail mean contribution diagnostic, not true marginal VaR. "
            "Use compute_tail_mean_contribution().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_tail_mean_contribution(confidence=confidence)

    def compute_tail_mean_contribution(self, confidence: float = 0.95) -> pd.Series:
        """Legacy tail diagnostic: breach rate times average return in tail."""
        var = self.compute_var(confidence=confidence)
        return (self.returns <= -var).mean() * self.returns[self.returns <= -var].mean()

    def compute_market_cap_participation(self, market_cap: pd.DataFrame, trade_value: float = 100_000_000) -> pd.DataFrame:
        prices_close = self.prices.xs("adj_close", axis=1, level=1)
        return calc_exposures.market_cap_participation(self.position_units, prices_close, market_cap, trade_value=trade_value)

    def compute_volume_participation(
        self,
        volumes: pd.DataFrame,
        *,
        traded_units: Optional[pd.DataFrame] = None,
        traded_notional: Optional[pd.DataFrame] = None,
        trade_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """Shares/contracts traded divided by market volume.

        If explicit traded units are not supplied, the method infers them from
        absolute changes in position_units. This is a participation proxy, not
        portfolio turnover.
        """
        if trade_value is not None:
            warnings.warn(
                "trade_value is ignored by compute_volume_participation(); "
                "pass traded_units or traded_notional for explicit participation.",
                DeprecationWarning,
                stacklevel=2,
            )
        vols = volumes.reindex(index=self.returns.index, columns=self.returns.columns)
        if traded_units is None:
            if traded_notional is not None:
                prices_close = self.prices.xs("adj_close", axis=1, level=1)
                px = prices_close.reindex(index=self.returns.index, columns=self.returns.columns).ffill()
                traded_units = traded_notional.reindex(index=self.returns.index, columns=self.returns.columns).divide(px)
            else:
                traded_units = self.position_units.diff().abs()
        tu = traded_units.reindex(index=vols.index, columns=vols.columns).fillna(0.0)
        return calc_exposures.volume_participation(tu, vols)

    def compute_cumulative_attribution(self, weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Cumulative return attribution from lagged portfolio weights."""
        lagged_weights = self.weights if weights is None else weights
        lagged_weights = lagged_weights.reindex(
            index=self.returns.index,
            columns=self.returns.columns,
        ).shift(1)
        attribution = self.returns.multiply(lagged_weights).fillna(0.0)
        return attribution.cumsum()

    def compute_cumulative_pnl_attribution_from_units(self) -> pd.DataFrame:
        """Cumulative dollar P&L attribution from lagged position units."""
        prices_close = self.prices.xs("adj_close", axis=1, level=1)
        price_diff = prices_close.diff()
        pnl_attribution = price_diff.multiply(self.position_units.shift(1))
        return pnl_attribution.fillna(0.0).cumsum()

    def compute_performance_attribution(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        excess = self.returns.sub(benchmark_returns, axis=0)
        return calc_attr.compute_performance_attribution(excess)

    def compute_factor_exposures_ols(self, factor_returns: pd.DataFrame, add_intercept: bool = True) -> pd.DataFrame:
        return calc_attr.compute_factor_exposures_ols(self.returns, factor_returns, add_intercept=add_intercept)

    def compute_factor_alpha(self, factor_returns: pd.DataFrame, add_intercept: bool = True) -> pd.Series:
        return calc_attr.compute_factor_alpha(self.returns, factor_returns, add_intercept=add_intercept)

    # NAV and higher-level returns ------------------------------------
    def compute_relative_nav_from_weights(self, weights: pd.DataFrame) -> pd.Series:
        """Build a relative NAV series from lagged portfolio weights.

        Weights are shifted by one bar so weights observed on date t earn
        returns from t+1. This avoids same-bar look-ahead in analytics.
        """
        if weights is None:
            raise ValueError("weights must be provided explicitly; units are position quantities, not weights")
        returns = self.returns.fillna(0.0)
        weights_for_return = weights.reindex(
            index=returns.index,
            columns=returns.columns,
        ).shift(1).fillna(0.0)
        portfolio_returns = returns.multiply(weights_for_return).sum(axis=1)
        return (1.0 + portfolio_returns).cumprod()

    def compute_relative_nav(self) -> pd.Series:
        warnings.warn(
            "compute_relative_nav() is deprecated; use "
            "compute_relative_nav_from_weights() so the weight-based contract "
            "and one-bar execution lag are explicit.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_relative_nav_from_weights(self.weights)

    def convert_returns_to_nav(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        init_period: Optional[int] = 1,
        terminal_value: Optional[np.ndarray] = None,
        init_value: Optional[np.ndarray | float] = None,
        first_date: Optional[pd.Timestamp] = None,
        freq: Optional[str] = None,
        constant_trade_level: bool = False,
        ffill_between_nans: bool = True,
    ) -> pd.DataFrame:
        return calc_returns.convert_returns_to_nav(
            returns,
            init_period=init_period,
            terminal_value=terminal_value,
            init_value=init_value,
            freq=freq,
            ffill_between_nans=ffill_between_nans,
            constant_trade_level=constant_trade_level,
        )

    def convert_log_returns_to_nav(
        self,
        log_returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        init_period: Optional[int] = None,
        terminal_value: Optional[np.ndarray] = None,
        init_value: Optional[np.ndarray | float] = None,
    ) -> pd.DataFrame:
        return calc_returns.convert_log_returns_to_nav(
            log_returns,
            init_period=init_period,
            terminal_value=terminal_value,
            init_value=init_value,
        )

    # Sampling & Outliers ----------------------------------------------
    def resample_prices_at_frequency(
        self,
        *,
        freq: Optional[str] = None,
        include_start_date: bool = False,
        include_end_date: bool = False,
        ffill_nans: bool = True,
    ) -> pd.DataFrame:
        return calc_sampling.resample_prices_at_frequency(
            self.prices,
            freq=freq,
            include_start_date=include_start_date,
            include_end_date=include_end_date,
            ffill_nans=ffill_nans,
        )

    def ffill_prices_between_nans(self, method: Optional[str] = "ffill") -> pd.DataFrame:
        return calc_sampling.ffill_prices_between_nans(self.prices, method=method)

    def set_first_non_nan_returns_to_zero(
        self, returns: Union[pd.Series, pd.DataFrame], init_period: Union[int, None] = 1
    ) -> Union[pd.Series, pd.DataFrame]:
        return calc_sampling.set_first_non_nan_returns_to_zero(returns, init_period=init_period)

    def get_outliers(self, quantile: float = 0.95) -> pd.DataFrame:
        return calc_outliers.outliers(self.returns, quantile)

    def remove_return_outliers(self, quantile: float = 0.95) -> pd.DataFrame:
        return calc_outliers.remove_outliers(self.returns, quantile)

    # Rolling -----------------------------------------------------------
    def compute_rolling_returns(self, periods: int = 7) -> pd.DataFrame:
        return calc_roll.rolling_mean(self.returns, window=periods)

    def compute_rolling_volatility(self, periods: int = 7) -> pd.DataFrame:
        return calc_roll.rolling_volatility(self.returns, window=periods)

    def compute_rolling_sharpe_ratio(self, periods: int = 7, risk_free_rate: Optional[float] = None, **kwargs) -> pd.DataFrame:
        window = kwargs.get("window", periods)
        rf = self._resolve_risk_free_rate(risk_free_rate)
        return calc_roll.rolling_sharpe_ratio(self.returns, risk_free_rate=rf, window=window, days_per_year=self._config.days_per_year)

    def compute_rolling_max_drawdown(self, periods: int = 7) -> pd.DataFrame:
        prices_close = self.prices.xs("adj_close", axis=1, level=1)
        return calc_roll.rolling_max_drawdown(prices_close, window=periods)

    def compute_rolling_beta(self, benchmark_returns: pd.Series, periods: int = 60, **kwargs) -> pd.DataFrame:
        window = kwargs.get("window", periods)
        bench = benchmark_returns.reindex(self.returns.index)
        return calc_roll.rolling_beta(self.returns, bench, window=window)

    def compute_rolling_alpha(self, benchmark_returns: pd.Series, periods: int = 60, **kwargs) -> pd.DataFrame:
        window = kwargs.get("window", periods)
        bench = benchmark_returns.reindex(self.returns.index)
        rf = self._resolve_risk_free_rate(kwargs.get("risk_free_rate"))
        return calc_roll.rolling_alpha(self.returns, bench, window=window, risk_free_rate=rf, days_per_year=self._config.days_per_year)

    def compute_rolling_correlation(self, window: int = 30) -> pd.DataFrame:
        return calc_roll.rolling_correlation(self.returns, window=window)

    def compute_correlation_matrix(self) -> pd.DataFrame:
        return calc_metrics.correlation_matrix(self.returns)

    # Holding-period returns ------------------------------------------
    def compute_holding_period_return_distribution(self) -> pd.DataFrame:
        return (1 + self.returns).cumprod() - 1

    # Duration ----------------------------------------------------------
    def compute_duration_in_days(self) -> int:
        dates = self.data_index
        if len(dates) < 2:
            return 0
        if dates[0] > dates[-1]:
            raise ValueError(f"Inconsistent dates: start date={dates[0]}, end date={dates[-1]}")
        return max((dates[-1] - dates[0]).days, 1)

    def compute_duration_in_years(self, days_per_year: float | None = None) -> float:
        dpy = float(days_per_year or self._config.days_per_year)
        return (self.data_index[-1] - self.data_index[0]).days / dpy
