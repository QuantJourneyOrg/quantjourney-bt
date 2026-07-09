"""
PortfolioData - Institutional Portfolio State Container
=======================================================

PortfolioData stores auditable portfolio state: NAV, realized cash, positions,
target weights, realized weights, returns, costs, drawdowns, and derived
analytics. It is designed for deterministic backtest replay, portfolio
accounting, and institutional-grade report generation.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Union, Dict, Optional, Tuple, List
from enum import Enum
import warnings

from backtester.portfolio.instr_data import InstrumentData
from backtester.utils.decorators import error_logger
from backtester.utils.logger import logger
from backtester.portfolio.schemas import (
    validate_nav_series,
    validate_weights_frame,
)


# PortfolioData class ------------------------------------------------------------------
@dataclass
class PortfolioData:
    """
    Class to store portfolio data and compute various portfolio metrics

    Timezone convention (important):
        ``__post_init__`` silently converts ALL datetime indices (NAV,
        weights, positions, cash, flags, calendar) to tz-aware UTC —
        tz-naive input indices are localized to UTC, tz-aware ones are
        converted. This happens even if the series you passed in was
        tz-naive. Consequence: joining/aligning any PortfolioData series
        (e.g. ``portfolio_data.net_asset_value``) with an external
        tz-naive series raises
        ``TypeError: Cannot join tz-naive with tz-aware DatetimeIndex``.
        Remedy: strip the timezone first, e.g.
        ``nav = pd.Series(pd_data.net_asset_value); nav.index = nav.index.tz_localize(None)``
        (or localize your external series to UTC instead).

    Attributes:
        instruments (InstrumentData): InstrumentData object, containing instrument data
        net_asset_value (pd.Series): Net Asset Value, per date.
            NOTE: after construction its index is ALWAYS tz-aware UTC
            (converted in ``__post_init__``), regardless of the timezone
            of the series passed in. Use ``.index.tz_localize(None)`` on a
            copy before joining with tz-naive data.
        input_weights (Optional[Union[np.ndarray, pd.DataFrame, Dict[str, float]]): Initial weights, e.g. {'AAPL': 0.5, 'MSFT': 0.5}
        rebalance_flags (Optional[pd.Series]): Rebalance flags, e.g True if rebalancing is needed
        asset_name_map (Optional[Dict[str, str]]): Asset name mapping e.g. {'AAPL': 'Apple Inc.'}
        trading_calendar (Optional[pd.Series]): Trading calendar
        weights (Optional[pd.DataFrame]): Weights, per date
        positions (Optional[pd.DataFrame]): Portfolio positions, per date
        cash_buffer (Optional[pd.Series]): Cash buffer, per date
        returns (Optional[pd.Series]): Daily returns
        cumulative_returns (Optional[pd.Series]): Cumulative returns
        volatility (Optional[pd.Series]): Daily volatility
        total_pnl (Optional[pd.Series]): Total PnL
        total_transaction_costs (Optional[pd.Series]): Total transaction costs
        sharpe_ratio (Optional[pd.Series]): Sharpe ratio
        drawdown (Optional[pd.Series]): Drawdown
    """

    instruments: InstrumentData
    net_asset_value: pd.Series

    input_weights: Optional[Union[np.ndarray, pd.DataFrame, Dict[str, float]]] = (None)
    rebalance_flags: Optional[pd.Series] = (None)
    asset_name_map: Optional[Dict[str, str]] = (None)
    trading_calendar: Optional[pd.Series] = None

    weights: Optional[pd.DataFrame] = None
    positions: Optional[pd.DataFrame] = None   
    position_values: Optional[pd.DataFrame] = None
    cash_buffer: Union[float, pd.Series] = 0.05             # Default to 5% cash buffer
    # Optional actual cash time series (distinct from configuration buffer)
    cash: Optional[pd.Series] = None

    # It is useful to store the following metrics for quick access in the backtesting engine:
    returns: Optional[pd.Series] = None
    returns_for_metrics: Optional[pd.Series] = None
    cumulative_returns: Optional[pd.Series] = (None)
    volatility: Optional[pd.Series] = None
    total_pnl: Optional[pd.Series] = None
    total_transaction_costs: Optional[pd.Series] = (None)
    sharpe_ratio: Optional[pd.Series] = None
    drawdown: Optional[pd.Series] = None
    _version: int = field(default=0, repr=False)
    _skip_initial_metrics: bool = field(default=False, repr=False)

    def __post_init__(self):
        """
        Post-initialization method to validate the data
        """
        self._convert_timezone("UTC")
        self._validate_index_integrity()
        self._validate_data()
        if not self._skip_initial_metrics:
            self._calculate_aggregate_metrics()

    @error_logger("Error validating data")
    def _validate_data(self):
        """
        Validate the data DataFrame to ensure it has the required columns
        """
        # Required attributes validation
        if self.instruments is None:
            raise ValueError("PortfolioData is missing the required attribute: instruments")
        
        if not isinstance(self.instruments, InstrumentData):
            raise ValueError("instruments must be an InstrumentData object")

        if self.net_asset_value is None:
            raise ValueError("PortfolioData is missing the required attribute: net_asset_value")
        
        if not isinstance(self.net_asset_value, pd.Series):
            raise ValueError("net_asset_value must be a pandas Series")
        # Schema validation for core frames
        validate_nav_series(self.net_asset_value)
        if self.weights is not None:
            validate_weights_frame(self.weights, self.instrument_names)

    @error_logger("Error converting timezone")
    def _convert_timezone(self, timezone: str = "UTC"):
        """
        Convert all datetime indices to the specified timezone.
        """

        def ensure_datetime_and_convert(index):
            if not isinstance(index, pd.DatetimeIndex):
                index = pd.to_datetime(index)
            return (
                index.tz_convert(timezone) if index.tz else index.tz_localize(timezone)
            )

        # Convert the timezone for each attribute safely
        if self.net_asset_value is not None:
            self.net_asset_value.index = ensure_datetime_and_convert(
                self.net_asset_value.index
            )
        if self.weights is not None:
            self.weights.index = ensure_datetime_and_convert(self.weights.index)
        if self.positions is not None:
            self.positions.index = ensure_datetime_and_convert(self.positions.index)
        if self.position_values is not None:
            self.position_values.index = ensure_datetime_and_convert(self.position_values.index)
        if self.cash is not None:
            self.cash.index = ensure_datetime_and_convert(self.cash.index)
        if self.rebalance_flags is not None:
            self.rebalance_flags.index = ensure_datetime_and_convert(self.rebalance_flags.index)
        if self.trading_calendar is not None:
            self.trading_calendar.index = ensure_datetime_and_convert(self.trading_calendar.index)

    def _validate_index_integrity(self):
        """Ensure time indices are tz-aware, monotonic, and unique."""
        def normalize(obj, name: str):
            if obj is None:
                return None
            if not isinstance(obj.index, pd.DatetimeIndex):
                obj = obj.copy()
                obj.index = pd.to_datetime(obj.index)
            if obj.index.tz is None:
                obj = obj.copy()
                obj.index = obj.index.tz_localize("UTC")
            if obj.index.has_duplicates:
                dup_count = int(obj.index.duplicated().sum())
                raise ValueError(f"{name}: index contains {dup_count} duplicate timestamp(s)")
            if not obj.index.is_monotonic_increasing:
                obj = obj.sort_index()
            return obj

        if self.net_asset_value is not None:
            self.net_asset_value = normalize(self.net_asset_value, "net_asset_value")
        if self.weights is not None:
            self.weights = normalize(self.weights, "weights")
        if self.positions is not None:
            self.positions = normalize(self.positions, "positions")
        if self.position_values is not None:
            self.position_values = normalize(self.position_values, "position_values")
        if self.cash is not None:
            self.cash = normalize(self.cash, "cash")
        if self.rebalance_flags is not None:
            self.rebalance_flags = normalize(self.rebalance_flags, "rebalance_flags")
        if self.trading_calendar is not None:
            self.trading_calendar = normalize(self.trading_calendar, "trading_calendar")

    @staticmethod
    def _coerce_timestamp_to_index_tz(
        value: Union[str, pd.Timestamp, None],
        index: pd.Index,
    ) -> Optional[pd.Timestamp]:
        """Coerce a user-supplied timestamp to the timezone of a DatetimeIndex."""
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if not isinstance(index, pd.DatetimeIndex):
            return ts
        if index.tz is None:
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        return ts.tz_convert(index.tz) if ts.tzinfo is not None else ts.tz_localize(index.tz)

    @staticmethod
    def _index_as_utc(index: pd.Index) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(index)
        return idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")

    @error_logger("Error calculating aggregate metrics")
    def _calculate_aggregate_metrics(self):
        """
        Calculate aggregate metrics for the portfolio.
        Uses more stable calculations for returns and drawdowns.
        """
        # Keep display returns separate from metric returns: the first row is a
        # reporting convention, not an observed portfolio return.
        raw_returns = self.net_asset_value.pct_change()
        self.returns_for_metrics = raw_returns.dropna()
        self.returns = raw_returns.fillna(0.0)
        self.cumulative_returns = (1 + self.returns).cumprod() - 1
        self.volatility = self.returns_for_metrics.rolling(window=20).std() * np.sqrt(252)  # Annualized

        # Improved drawdown calculation using cumulative returns
        wealth_index = 1 + self.cumulative_returns
        previous_peaks = wealth_index.expanding(min_periods=1).max()
        self.drawdown = (wealth_index - previous_peaks) / previous_peaks

        # Initialize optional metrics
        self.total_pnl = self.net_asset_value - self.net_asset_value.iloc[0]
        
        # Calculate Sharpe ratio if returns and volatility are available
        metric_returns = self.returns_for_metrics
        if not metric_returns.isna().all() and not self.volatility.isna().all():
            risk_free_rate = 0.02  # 2% annual — consistent with PortfolioCalculations
            excess_returns = metric_returns - risk_free_rate / 252
            std_dev = excess_returns.std()
            if std_dev > 0:  # Prevent division by zero
                self.sharpe_ratio = excess_returns.mean() / std_dev * np.sqrt(252)
            else:
                self.sharpe_ratio = None
                
    # Properties -----------------------------------------------------------------

    @property
    def time_period(self) -> Tuple[str, str]:
        """
        Return the start and end dates of the data as strings.
        """
        return (
            str(self.net_asset_value.index[0].date()),
            str(self.net_asset_value.index[-1].date()),
        )

    @property
    def instrument_names(self) -> List[str]:
        """
        Return the list of instrument names in the portfolio.
        """
        # Delegate to InstrumentData accessor
        try:
            return self.instruments.get_instruments()
        except AttributeError:
            # Fallback: attempt to infer from prices columns if needed
            if hasattr(self.instruments, "prices"):
                cols = self.instruments.prices.columns
                return list(dict.fromkeys(cols.get_level_values(0)))
            raise

    @property
    def dates(self) -> pd.DatetimeIndex:
        """
        Return the list of dates in the portfolio data.
        """
        return self.net_asset_value.index

    @property
    def num_instruments(self) -> int:
        """
        Return the number of instruments in the portfolio.
        """
        return len(self.instrument_names)

    @property
    def num_dates(self) -> int:
        """
        Return the number of dates in the portfolio data.
        """
        return len(self.dates)

    @property
    def total_value(self) -> pd.Series:
        """
        Return the total portfolio value over time.
        """
        return self.net_asset_value

    def assert_accounting_identity(
        self,
        *,
        position_values: Optional[pd.DataFrame] = None,
        cash: Optional[pd.Series] = None,
        rtol: float = 1e-10,
        atol: float = 1e-8,
    ) -> None:
        """Validate NAV = cash + sum(position_values) over aligned dates."""
        pv = position_values if position_values is not None else self.position_values
        cash_series = cash if cash is not None else self.cash
        if pv is None:
            raise ValueError("position_values are required for accounting identity validation")
        if cash_series is None:
            raise ValueError("cash series is required for accounting identity validation")

        lhs, rhs_cash = self.net_asset_value.align(cash_series, join="inner")
        rhs_positions = pv.sum(axis=1).reindex(lhs.index)
        rhs = rhs_cash.reindex(lhs.index) + rhs_positions
        if len(lhs) == 0:
            raise ValueError("No overlapping dates for accounting identity validation")
        if rhs.isna().any() or lhs.isna().any():
            raise ValueError("Accounting identity inputs contain missing aligned values")
        if not np.allclose(lhs.to_numpy(), rhs.to_numpy(), rtol=rtol, atol=atol):
            max_abs = float(np.max(np.abs(lhs.to_numpy() - rhs.to_numpy())))
            raise AssertionError(f"NAV accounting identity failed; max_abs_diff={max_abs:.12g}")

    @property
    def average_weights(self) -> pd.Series:
        """
        Return the average weights of each instrument over the entire period.
        """
        if self.weights is None:
            raise ValueError("weights are required to compute average_weights")
        return self.weights.mean()

    # Methods for Weights, Positions, Value  ---------------------------------------------------------

    def generate_positions(self, strategies: pd.DataFrame):
        """
        Generate positions based on provided strategies and constraints.

        Args:
                strategies (pd.DataFrame): DataFrame of trading strategies to be used for generating positions.
        """
        if strategies is None or strategies.empty:
            raise ValueError("Strategies are required to generate positions.")

        warnings.warn(
            "generate_positions(strategies) is a legacy helper. Strategy data "
            "represents signal intent, not executed positions. Prefer the "
            "Backtester execution pipeline: signals -> target_weights -> "
            "orders -> fills -> positions.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.positions = strategies

    def generate_weights(self, *, positions_are_market_values: bool = True):
        """
        Generate weights from position market values and NAV.

        Quantity positions cannot be converted into weights without prices,
        multipliers and FX rates; callers must mark those explicitly elsewhere.
        """
        if self.positions is None:
            raise ValueError("Positions are required to generate weights.")
        if not positions_are_market_values:
            raise ValueError(
                "Cannot generate weights from quantity positions without prices; "
                "provide position market values or use the execution engine."
            )

        pos = self.positions.apply(pd.to_numeric, errors="coerce")
        nav = self.net_asset_value.reindex(pos.index)
        if nav.isna().any():
            raise ValueError("NAV is missing for one or more position dates")
        if (nav == 0).any():
            raise ValueError("Cannot generate weights with zero NAV")
        self.weights = pos.divide(nav, axis=0)
        self._version += 1

    def update_positions(self, new_positions: pd.DataFrame):
        """
        Update the positions DataFrame.
        """
        self.positions = new_positions.copy()
        self._convert_timezone("UTC")
        self._validate_index_integrity()
        self._version += 1

    def update_weights(self, new_weights: pd.DataFrame):
        """
        Update the weights DataFrame.
        """
        self.weights = new_weights.copy()
        self._convert_timezone("UTC")
        self._validate_index_integrity()
        validate_weights_frame(self.weights, self.instrument_names)
        self._version += 1
        
    def update_cash(self, cash_series: pd.Series) -> None:
        """Update the realized cash time series (not the buffer setting)."""
        self.cash = cash_series.copy()
        self._convert_timezone("UTC")
        self._validate_index_integrity()
        self._version += 1

    def update_net_asset_value(self, new_nav: pd.Series):
        """
        Update the net asset value Series.
        """
        self.net_asset_value = new_nav.copy()
        self._convert_timezone("UTC")
        self._validate_index_integrity()
        self._validate_data()
        self._calculate_aggregate_metrics()
        self._version += 1

    # Immutable-style helpers ---------------------------------------------------
    def with_weights(self, weights: pd.DataFrame) -> "PortfolioData":
        return PortfolioData(
            instruments=self.instruments,
            net_asset_value=self.net_asset_value,
            input_weights=self.input_weights,
            rebalance_flags=self.rebalance_flags,
            asset_name_map=self.asset_name_map,
            trading_calendar=self.trading_calendar,
            weights=weights,
            positions=self.positions,
            position_values=self.position_values,
            cash_buffer=self.cash_buffer,
            cash=self.cash,
            returns=self.returns,
            returns_for_metrics=self.returns_for_metrics,
            cumulative_returns=self.cumulative_returns,
            volatility=self.volatility,
            total_pnl=self.total_pnl,
            total_transaction_costs=self.total_transaction_costs,
            sharpe_ratio=self.sharpe_ratio,
            drawdown=self.drawdown,
            _version=self._version + 1,
        )

    def with_positions(self, positions: pd.DataFrame) -> "PortfolioData":
        return PortfolioData(
            instruments=self.instruments,
            net_asset_value=self.net_asset_value,
            input_weights=self.input_weights,
            rebalance_flags=self.rebalance_flags,
            asset_name_map=self.asset_name_map,
            trading_calendar=self.trading_calendar,
            weights=self.weights,
            positions=positions,
            position_values=self.position_values,
            cash_buffer=self.cash_buffer,
            cash=self.cash,
            returns=self.returns,
            returns_for_metrics=self.returns_for_metrics,
            cumulative_returns=self.cumulative_returns,
            volatility=self.volatility,
            total_pnl=self.total_pnl,
            total_transaction_costs=self.total_transaction_costs,
            sharpe_ratio=self.sharpe_ratio,
            drawdown=self.drawdown,
            _version=self._version + 1,
        )

    def with_net_asset_value(self, nav: pd.Series) -> "PortfolioData":
        pd_obj = PortfolioData(
            instruments=self.instruments,
            net_asset_value=nav,
            input_weights=self.input_weights,
            rebalance_flags=self.rebalance_flags,
            asset_name_map=self.asset_name_map,
            trading_calendar=self.trading_calendar,
            weights=self.weights,
            positions=self.positions,
            position_values=self.position_values,
            cash_buffer=self.cash_buffer,
            cash=self.cash,
            _version=self._version + 1,
        )
        return pd_obj

    @property
    def version(self) -> int:
        return self._version

    # Generic Calculations ------------------------------------------------------------------

    def get_instrument_weights(self, instrument: str) -> pd.Series:
        """
        Return the weights for a specific instrument over time.
        """
        if self.weights is None:
            raise ValueError("weights are required to get instrument weights")
        return self.weights[instrument]

    def get_instrument_value(self, instrument: str) -> pd.Series:
        """
        Return the value of a specific instrument over time.
        """
        if self.weights is None:
            raise ValueError("weights are required to get instrument value")
        return self.net_asset_value * self.weights[instrument]

    def get_portfolio_snapshot(self, date: Union[str, pd.Timestamp]) -> pd.Series:
        """
        Return a snapshot of the portfolio weights on a specific date.
        """
        if self.weights is None:
            raise ValueError("weights are required to get a portfolio snapshot")
        return self.weights.loc[date]

    def is_trading_day(self, date: Union[str, pd.Timestamp]) -> bool:
        """
        Check if a given date is a trading day.
        """
        if self.trading_calendar is None:
            return True  # Assume all days are trading days if no calendar is provided
        return self.trading_calendar.loc[date, "is_trading_day"]

    def get_asset_name(self, instrument: str) -> str:
        """
        Get the full asset name for a given instrument symbol.
        """
        if self.asset_name_map is None:
            return instrument
        return self.asset_name_map.get(instrument, instrument)

    # For CrossValidation & Walk-Forward -------------------------------------------

    def slice_data(self, date_range) -> "PortfolioData":
        """
        Slice the data based on the given date range, used for cross-validation

        Args:
                date_range: slice or list of dates

        Returns:
                PortfolioData object with sliced data
        """
        if isinstance(date_range, slice):
            start = self._coerce_timestamp_to_index_tz(
                date_range.start,
                self.net_asset_value.index,
            )
            stop = self._coerce_timestamp_to_index_tz(
                date_range.stop,
                self.net_asset_value.index,
            )
            sliced_nav = self.net_asset_value.loc[start:stop]
            sliced_weights = (
                self.weights.loc[start:stop]
                if self.weights is not None
                else None
            )
            sliced_positions = (
                self.positions.loc[start:stop]
                if self.positions is not None
                else None
            )
            sliced_position_values = (
                self.position_values.loc[start:stop]
                if self.position_values is not None
                else None
            )
            sliced_cash = (
                self.cash.loc[start:stop]
                if self.cash is not None
                else None
            )
            sliced_rebalance_flags = (
                self.rebalance_flags.loc[start:stop]
                if self.rebalance_flags is not None
                else None
            )
            # InstrumentData expects a tuple (start, end), not a slice
            sliced_instruments = self.instruments.slice_data((start, stop))
        else:
            sliced_nav = self.net_asset_value.loc[date_range]
            sliced_weights = self.weights.loc[date_range] if self.weights is not None else None
            sliced_positions = (
                self.positions.loc[date_range] if self.positions is not None else None
            )
            sliced_position_values = (
                self.position_values.loc[date_range]
                if self.position_values is not None
                else None
            )
            sliced_cash = self.cash.loc[date_range] if self.cash is not None else None
            sliced_rebalance_flags = (
                self.rebalance_flags.loc[date_range]
                if self.rebalance_flags is not None
                else None
            )
            sliced_instruments = self.instruments.slice_data(date_range)

        sliced_nav_index = self._index_as_utc(sliced_nav.index)
        instrument_index = self._index_as_utc(sliced_instruments.get_dates())
        index_checks = [sliced_nav_index.equals(instrument_index)]
        if sliced_weights is not None:
            index_checks.append(sliced_nav_index.equals(self._index_as_utc(sliced_weights.index)))
        if sliced_positions is not None:
            index_checks.append(sliced_nav_index.equals(self._index_as_utc(sliced_positions.index)))
        if sliced_position_values is not None:
            index_checks.append(sliced_nav_index.equals(self._index_as_utc(sliced_position_values.index)))
        if sliced_cash is not None:
            index_checks.append(sliced_nav_index.equals(self._index_as_utc(sliced_cash.index)))
        if not all(index_checks):
            raise ValueError("Inconsistent date ranges after slicing")

        return PortfolioData(
            instruments=sliced_instruments,
            net_asset_value=sliced_nav,
            weights=sliced_weights,
            positions=sliced_positions,
            position_values=sliced_position_values,
            input_weights=self.input_weights,
            rebalance_flags=sliced_rebalance_flags,
            asset_name_map=self.asset_name_map,
            trading_calendar=self.trading_calendar,
            cash_buffer=self.cash_buffer,
            cash=sliced_cash,
        )


"""
Note: In-file UnitTests have been removed. Tests now live under
quantjourney/portfolio/_tests.
"""


# Builder ----------------------------------------------------------------------
class PortfolioDataBuilder:
    @staticmethod
    def from_parts(
        *,
        instruments: "InstrumentData",
        nav: pd.Series,
        weights: Optional[pd.DataFrame] = None,
        positions: Optional[pd.DataFrame] = None,
        position_values: Optional[pd.DataFrame] = None,
        input_weights: Optional[Union[np.ndarray, pd.DataFrame, Dict[str, float]]] = None,
        rebalance_flags: Optional[pd.Series] = None,
        asset_name_map: Optional[Dict[str, str]] = None,
        trading_calendar: Optional[pd.Series] = None,
        cash_buffer: Union[float, pd.Series] = 0.05,
        cash: Optional[pd.Series] = None,
    ) -> "PortfolioData":
        return PortfolioData(
            instruments=instruments,
            net_asset_value=nav,
            input_weights=input_weights,
            rebalance_flags=rebalance_flags,
            asset_name_map=asset_name_map,
            trading_calendar=trading_calendar,
            weights=weights,
            positions=positions,
            position_values=position_values,
            cash_buffer=cash_buffer,
            cash=cash,
        )
