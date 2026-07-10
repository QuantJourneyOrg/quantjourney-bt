"""
        InstrumentData - Vectorized Storage for Individual Instruments
        ------------------------------------------------------------

        This module provides a data structure for storing and managing individual
        instrument data in a vectorized approach. It is designed to facilitate efficient
        access and manipulation of data for each instrument, supporting streamlined calculations
        and analysis within the QuantJourney Framework.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtester.portfolio.schemas import (
    validate_metrics_frame,
    validate_parameters_frame,
    validate_prices_frame,
    validate_strategies_frame,
)
from backtester.utils.decorators import error_logger


# InstrumentData class ---------------------------------------------------------
@dataclass
class InstrumentData:
    """
    Class to store instrument data and perform various calculations

    This class organizes the data into four main components:
            1. prices - Raw price data (OHLCAV).
            2. Strategies - Data related to strategy outputs (signals, positions, etc.).
            3. metrics - Derived metrics such as returns, volatility, and PnL.
               Metrics evaluate how well strategies are performing.
            4. parameters - Calculated attributes such as eligibility, forecasts,
               and trading days that guide strategy execution.

    Each component is stored in a separate structure:
            - prices: Multi-index DataFrame with instruments as Level 0 and price
              attributes (for example, 'open' and 'close') as Level 1.
            - strategies: Dictionary of DataFrames, keyed by strategy name.
            - metrics: Dictionary of DataFrames for different metrics, keyed by metric name.
            - parameters: DataFrame or dictionary to handle parameters attributes.
    """

    group_data: pd.Series  # Related to grouping
    group_order: list[str]  # Related to grouping

    strategies: pd.DataFrame  # Multi-index columns: strategy, field, instrument
    # Multi-index frame: instrument at Level 0 and price field at Level 1.
    prices: pd.DataFrame
    metrics: (
        pd.DataFrame
    )  # Multi-index DataFrame with instruments as Level 0 and metric names as Level 1
    parameters: (
        pd.DataFrame
    )  # Multi-index DataFrame with instruments as Level 0 and parameters attributes as Level 1
    fundamentals: pd.DataFrame | None = (
        None  # Multi-index DataFrame with instruments as Level 0 and fundamental data as Level 1)
    )
    _version: int = 0  # cache invalidation version
    _skip_validation: bool = False  # skip heavy validation/normalization for speed

    def __post_init__(self):
        """
        Post-initialization method to validate the data.
        When _skip_validation=True, only essential setup is performed
        (column naming + adj_close detection) — saves ~400ms.
        """
        self.strategies = self._canonical_strategy_frame(self.strategies)

        if self._skip_validation:
            # Fast path: minimal setup only
            self.use_adjusted_close = "adj_close" in self.prices.columns.get_level_values(1)
            try:
                self.prices.columns.names = ["instrument", "price"]
            except Exception:
                pass
            try:
                self.metrics.columns.names = ["instrument", "metric"]
            except Exception:
                pass
            try:
                self.parameters.columns.names = ["instrument", "parameter"]
            except Exception:
                pass
            return

        # Full validation path (original)
        self._validate_data()
        self._convert_timezone("UTC")
        self._align_indexes()
        # Check if adj_close exists and set use_adjusted_close accordingly
        self.use_adjusted_close = "adj_close" in self.prices.columns.get_level_values(1)
        self._validate_index_integrity()
        # Name MultiIndex levels for consistency
        try:
            self.prices.columns.names = ["instrument", "price"]
        except Exception:
            pass
        try:
            self.metrics.columns.names = ["instrument", "metric"]
        except Exception:
            pass
        try:
            self.parameters.columns.names = ["instrument", "parameter"]
        except Exception:
            pass
        # Coerce parameter dtypes to expected types
        self._coerce_parameter_dtypes_inplace()
        # Validate frames (post-normalization)
        validate_prices_frame(self.prices)
        validate_metrics_frame(self.metrics)
        validate_parameters_frame(self.parameters)
        validate_strategies_frame(self.strategies)

    def _canonical_strategy_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return strategy data with ``(strategy, field, instrument)`` columns.

        Named legacy frames using ``(instrument, strategy, field)`` are
        reordered explicitly.  For unnamed three-level frames, the instrument
        level is inferred from the current universe; this keeps old serialized
        frames readable without guessing during every accessor call.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("strategies must be a pandas DataFrame")
        if frame.empty and not isinstance(frame.columns, pd.MultiIndex):
            return frame.copy()
        if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 3:
            raise ValueError(
                "strategies: expected three column levels: strategy, field, instrument"
            )

        result = frame.copy()
        canonical_names = ["strategy", "field", "instrument"]
        names = list(result.columns.names)
        if len(set(names)) == 3 and set(names) == set(canonical_names):
            result = result.reorder_levels(canonical_names, axis=1)
        else:
            known_instruments = set(map(str, self.group_data.index))
            if isinstance(self.prices.columns, pd.MultiIndex):
                known_instruments.update(map(str, self.prices.columns.get_level_values(0)))

            overlap_by_level = {
                level: len(
                    set(map(str, result.columns.get_level_values(level))) & known_instruments
                )
                for level in range(3)
            }
            if "instrument" in names:
                instrument_level = names.index("instrument")
            else:
                best_overlap = max(overlap_by_level.values())
                candidates = [
                    level for level, overlap in overlap_by_level.items() if overlap == best_overlap
                ]
                if best_overlap == 0:
                    # The unnamed frame cannot be inferred from the universe;
                    # retain compatibility with canonical legacy writer output.
                    instrument_level = 2
                elif len(candidates) == 1:
                    instrument_level = candidates[0]
                else:
                    raise ValueError(
                        "strategies: ambiguous unnamed instrument level; name the levels explicitly"
                    )

            if instrument_level == 0:
                # Legacy: (instrument, strategy, field).
                strategy_level, field_level = 1, 2
            elif instrument_level == 2:
                # Canonical: (strategy, field, instrument).
                strategy_level, field_level = 0, 1
            else:
                raise ValueError(
                    "strategies: unsupported unnamed layout with instrument "
                    "at level 1; use (strategy, field, instrument)"
                )
            if "strategy" in names and names.index("strategy") != strategy_level:
                raise ValueError("strategies: inconsistent strategy level name")
            if "field" in names and names.index("field") != field_level:
                raise ValueError("strategies: inconsistent field level name")
            result = result.reorder_levels([strategy_level, field_level, instrument_level], axis=1)

        result.columns = result.columns.set_names(canonical_names)
        if result.columns.has_duplicates:
            duplicates = result.columns[result.columns.duplicated()].tolist()
            raise ValueError(
                f"strategies: duplicate strategy/field/instrument columns: {duplicates[:5]}"
            )
        return result

    # Explicit accessors --------------------------------------------------------
    def get_prices(self, field: str | None = None) -> pd.DataFrame:
        if field is None:
            return self.prices
        return self.prices.xs(field, axis=1, level=1)

    def get_metrics(self, field: str | None = None) -> pd.DataFrame:
        if field is None:
            return self.metrics
        return self.metrics.xs(field, axis=1, level=1)

    def get_parameters(self, field: str | None = None) -> pd.DataFrame:
        if field is None:
            return self.parameters
        return self.parameters.xs(field, axis=1, level=1)

    def get_strategy(
        self,
        strategy: str | None = None,
        field: str | None = None,
        orientation: str = "instrument_first",
    ) -> pd.DataFrame:
        if orientation not in {"instrument_first", "strategy_first"}:
            raise ValueError("orientation must be 'instrument_first' or 'strategy_first'")
        if self.strategies.empty:
            return self.strategies.copy()

        out = self.strategies
        if strategy is not None:
            out = out.xs(strategy, axis=1, level="strategy", drop_level=False)
        if field is not None:
            out = out.xs(field, axis=1, level="field", drop_level=False)
        if orientation == "instrument_first":
            out = out.reorder_levels(["instrument", "strategy", "field"], axis=1)
        return out.sort_index(axis=1).copy()

    def _coerce_parameter_dtypes_inplace(self) -> None:
        lvl = self.parameters.columns.get_level_values(1)
        bool_cols = lvl.isin(["eligibility", "active", "is_trading_day"])  # booleans
        unit_cols = lvl == "units"
        daytype_cols = lvl == "day_type"
        exch_cols = lvl == "exchange"
        if bool_cols.any():
            bool_df = self.parameters.loc[:, bool_cols].astype("boolean").fillna(False)
            for col in bool_df.columns:
                self.parameters[col] = bool_df[col]
        if unit_cols.any():
            units_df = self.parameters.xs("units", axis=1, level=1, drop_level=False)
            units_df = units_df.apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)
            # Assign back per-column to force dtype
            for col in units_df.columns:
                self.parameters[col] = units_df[col]
        if daytype_cols.any():
            day_df = self.parameters.xs("day_type", axis=1, level=1, drop_level=False)
            for col in day_df.columns:
                self.parameters[col] = day_df[col].astype("category")
        if exch_cols.any():
            exch_df = self.parameters.xs("exchange", axis=1, level=1, drop_level=False)
            for col in exch_df.columns:
                self.parameters[col] = exch_df[col].astype("category")
        # Ensure units are float64 explicitly
        if unit_cols.any():
            units_df2 = self.parameters.xs("units", axis=1, level=1, drop_level=False)
            for col in units_df2.columns:
                self.parameters[col] = units_df2[col].astype("float64")

    @error_logger("Error validating data")
    def _validate_data(self):
        """
        Validate the data DataFrames to ensure they have the required columns.
        Used for all DataFrames: prices, metrics, parameters.

        Note: Strategies data is validated separately because its schema is
        strategy-specific.
        """
        expected_prices_columns = [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
        missing_prices = [
            col
            for col in expected_prices_columns
            if col not in self.prices.columns.get_level_values(1)
        ]
        if missing_prices:
            raise ValueError(
                f"Prices DataFrame is missing the following required columns: {missing_prices}"
            )

        expected_metrics_columns = [
            "returns",
            "volatility",
            "daily_pnl",
            "transaction_costs",
            "net_asset_value",
            "gross_asset_value",
            "daily_net_return",
            "drawdown",
        ]
        missing_metrics = [
            col
            for col in expected_metrics_columns
            if col not in self.metrics.columns.get_level_values(1)
        ]
        if missing_metrics:
            raise ValueError(
                f"Metrics DataFrame is missing the following columns: {missing_metrics}"
            )

        expected_parameters_columns = [
            "exchange",
            "units",
            "eligibility",
            "active",
            "forecasts",
            "is_trading_day",
            "day_type",
        ]
        missing_parameters = [
            col
            for col in expected_parameters_columns
            if col not in self.parameters.columns.get_level_values(1)
        ]
        if missing_parameters:
            raise ValueError(
                f"Operational data DataFrame is missing the following columns: {missing_parameters}"
            )

    @error_logger("Error converting timezone")
    def _convert_timezone(self, timezone: str = "UTC"):
        """
        Convert all datetime indices to the specified timezone.
        """

        def ensure_datetime_and_convert(index):
            if not isinstance(index, pd.DatetimeIndex):
                index = pd.to_datetime(index)
            return index.tz_convert(timezone) if index.tz else index.tz_localize(timezone)

        # Convert the timezone for each attribute safely
        self.prices.index = ensure_datetime_and_convert(self.prices.index)
        self.metrics.index = ensure_datetime_and_convert(self.metrics.index)
        self.parameters.index = ensure_datetime_and_convert(self.parameters.index)

        if not self.strategies.empty:
            self.strategies.index = ensure_datetime_and_convert(self.strategies.index)

        if self.fundamentals is not None and not self.fundamentals.empty:
            self.fundamentals.index = ensure_datetime_and_convert(self.fundamentals.index)

    @error_logger("Error converting timezone")
    def X_convert_timezone(self, timezone: str = "UTC"):
        """
        Convert all datetime indices to the specified timezone.
        """
        # self.prices.index = self.prices.index.tz_convert(timezone)
        # self.metrics.index = self.metrics.index.tz_convert(timezone)
        # self.parameters.index = self.parameters.index.tz_convert(timezone)
        # if not self.strategies.empty:
        # 	self.strategies.index = self.strategies.index.tz_convert(timezone)

    @error_logger("Error aligning indexes")
    def _align_indexes(self):
        """
        Ensure all DataFrames have the same index (timestamps)
        """
        # Align prices, metrics, and parameters first
        common_index = self.prices.index.intersection(self.metrics.index).intersection(
            self.parameters.index
        )

        self.prices = self.prices.loc[common_index]
        self.metrics = self.metrics.loc[common_index]
        self.parameters = self.parameters.loc[common_index]

        # Align strategies if it's not empty
        if not self.strategies.empty:
            self.strategies = self.strategies.loc[common_index]

    @error_logger("Error validating index integrity")
    def _validate_index_integrity(self):
        """Ensure indices are tz-aware, monotonic, and unique across frames."""

        def normalize_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
            if frame is None or frame.empty:
                return frame
            out = frame.copy()
            if not isinstance(out.index, pd.DatetimeIndex):
                out.index = pd.to_datetime(out.index)
            if out.index.tz is None:
                out.index = out.index.tz_localize("UTC")
            if out.index.has_duplicates:
                dup_count = int(out.index.duplicated().sum())
                raise ValueError(f"{name}: index contains {dup_count} duplicate timestamp(s)")
            if not out.index.is_monotonic_increasing:
                out = out.sort_index()
            return out

        # Apply and realign after fixing
        self.prices = normalize_frame(self.prices, "prices")
        self.metrics = normalize_frame(self.metrics, "metrics")
        self.parameters = normalize_frame(self.parameters, "parameters")
        if not self.strategies.empty:
            self.strategies = normalize_frame(self.strategies, "strategies")
        if self.fundamentals is not None and not self.fundamentals.empty:
            self.fundamentals = normalize_frame(self.fundamentals, "fundamentals")
        common_index = self.prices.index.intersection(self.metrics.index).intersection(
            self.parameters.index
        )
        self.prices = self.prices.loc[common_index]
        self.metrics = self.metrics.loc[common_index]
        self.parameters = self.parameters.loc[common_index]
        if not self.strategies.empty:
            self.strategies = self.strategies.reindex(common_index)

    # Properties -----------------------------------------------------------------

    def __getattr__(self, name: str) -> pd.DataFrame:
        """
        Dynamic attribute getter to handle common feature access by leaf name.

        Returns a DataFrame with instrument symbols as columns for recognized
        price, metric, or parameter leaf names.
        """
        prices_attributes = ["open", "high", "low", "close", "adj_close", "volume"]
        metrics_attributes = [
            "returns",
            "volatility",
            "daily_pnl",
            "transaction_costs",
            "net_asset_value",
            "gross_asset_value",
            "daily_net_return",
            "drawdown",
        ]
        parameters_attributes = [
            "exchange",
            "units",
            "eligibility",
            "forecasts",
            "is_trading_day",
            "day_type",
        ]

        if name in prices_attributes:
            # Return instrument x selected price field
            return self.prices.xs(name, axis=1, level=1)
        elif name in metrics_attributes:
            if name in self.metrics.columns.get_level_values(1):
                return self.metrics.xs(name, axis=1, level=1)
            raise AttributeError(f"'metrics' has no attribute '{name}'")
        elif name in parameters_attributes:
            if name in self.parameters.columns.get_level_values(1):
                return self.parameters.xs(name, axis=1, level=1)
            raise AttributeError(f"'parameters' has no attribute '{name}'")

        # Dynamic fallback — indicators (SMA_50_close, RSI_14_close, …)
        # and any other leaf name stored in prices/metrics/parameters/fundamentals
        for attr in ["prices", "metrics", "parameters", "fundamentals"]:
            try:
                df = object.__getattribute__(self, attr)
                if df is not None and not df.empty and name in df.columns.get_level_values(-1):
                    return df.xs(name, axis=1, level=-1)
            except AttributeError:
                continue
        raise AttributeError(f"'InstrumentData' object has no attribute '{name}'")

    def add_strategy_data(self, strategy_name: str, data_type: str, data: pd.DataFrame) -> None:
        """Add one strategy field without mutating or truncating input data.

        Strategy storage always uses ``(strategy, field, instrument)``
        columns and the same date index as market data.  A mismatched index is
        rejected instead of intersected because intersection could silently
        remove already stored decisions and shorten the backtest.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("strategy data must be a pandas DataFrame")
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string")
        if not isinstance(data_type, str) or not data_type.strip():
            raise ValueError("data_type must be a non-empty string")
        if data.empty:
            raise ValueError("strategy data must not be empty")

        working = data.copy(deep=True)
        if isinstance(working.columns, pd.MultiIndex):
            if working.columns.nlevels != 3:
                raise ValueError("strategy data MultiIndex columns must have three levels")
            working = self._canonical_strategy_frame(working)
            instruments = list(working.columns.get_level_values("instrument"))
        else:
            instruments = list(working.columns)

        if pd.Index(instruments).has_duplicates:
            duplicates = pd.Index(instruments)[pd.Index(instruments).duplicated()].tolist()
            raise ValueError(f"strategy data contains duplicate instruments: {duplicates}")

        expected_index = self.prices.index
        if not working.index.equals(expected_index):
            raise ValueError(
                "strategy data index must exactly match the market-data index "
                f"({len(expected_index)} rows expected, got {len(working.index)})"
            )
        known_instruments = set(map(str, self.get_instruments()))
        unknown = sorted(set(map(str, instruments)) - known_instruments)
        if unknown:
            raise ValueError(f"strategy data contains instruments outside the universe: {unknown}")

        working.columns = pd.MultiIndex.from_arrays(
            [
                [strategy_name] * len(instruments),
                [data_type] * len(instruments),
                instruments,
            ],
            names=["strategy", "field", "instrument"],
        )

        if self.strategies.empty:
            self.strategies = working
        else:
            if not self.strategies.index.equals(expected_index):
                raise ValueError("existing strategy data index does not match market data")
            columns = self.strategies.columns
            replace_mask = (columns.get_level_values("strategy") == strategy_name) & (
                columns.get_level_values("field") == data_type
            )
            retained = self.strategies.loc[:, ~replace_mask]
            self.strategies = pd.concat([retained, working], axis=1)
        self.strategies = self.strategies.sort_index(axis=1)
        self.on_data_change()

    # New immutable-style helpers ----------------------------------------------
    def with_prices(self, prices: pd.DataFrame) -> "InstrumentData":
        return InstrumentData(
            prices=prices,
            strategies=self.strategies,
            metrics=self.metrics,
            parameters=self.parameters,
            group_data=self.group_data,
            group_order=self.group_order,
            fundamentals=self.fundamentals,
            _skip_validation=self._skip_validation,
            _version=self._version + 1,
        )

    def with_metrics(self, metrics: pd.DataFrame) -> "InstrumentData":
        return InstrumentData(
            prices=self.prices,
            strategies=self.strategies,
            metrics=metrics,
            parameters=self.parameters,
            group_data=self.group_data,
            group_order=self.group_order,
            fundamentals=self.fundamentals,
            _skip_validation=self._skip_validation,
            _version=self._version + 1,
        )

    def with_parameters(self, parameters: pd.DataFrame) -> "InstrumentData":
        return InstrumentData(
            prices=self.prices,
            strategies=self.strategies,
            metrics=self.metrics,
            parameters=parameters,
            group_data=self.group_data,
            group_order=self.group_order,
            fundamentals=self.fundamentals,
            _skip_validation=self._skip_validation,
            _version=self._version + 1,
        )

    def with_strategies(self, strategies: pd.DataFrame) -> "InstrumentData":
        return InstrumentData(
            prices=self.prices,
            strategies=strategies,
            metrics=self.metrics,
            parameters=self.parameters,
            group_data=self.group_data,
            group_order=self.group_order,
            fundamentals=self.fundamentals,
            _skip_validation=self._skip_validation,
            _version=self._version + 1,
        )

    def with_fundamentals(self, fundamentals: pd.DataFrame | None) -> "InstrumentData":
        return InstrumentData(
            prices=self.prices,
            strategies=self.strategies,
            metrics=self.metrics,
            parameters=self.parameters,
            group_data=self.group_data,
            group_order=self.group_order,
            fundamentals=fundamentals,
            _skip_validation=self._skip_validation,
            _version=self._version + 1,
        )

    def canonicalize_strategies(self, orientation: str = "instrument_first") -> pd.DataFrame:
        """Return an oriented copy of strategy data without changing storage."""
        if orientation not in {"instrument_first", "strategy_first"}:
            raise ValueError("orientation must be 'instrument_first' or 'strategy_first'")
        canonical = self._canonical_strategy_frame(self.strategies)
        if canonical.empty and not isinstance(canonical.columns, pd.MultiIndex):
            return canonical
        if orientation == "instrument_first":
            canonical = canonical.reorder_levels(["instrument", "strategy", "field"], axis=1)
        return canonical.sort_index(axis=1).copy()

    def Xadd_strategy_data(self, strategy_name: str, data_type: str, data: pd.DataFrame) -> None:
        """
        Add data to the specified strategy in the strategies DataFrame.

        By detault we add only data for trading days.
        """
        # Ensure correct MultiIndex structure in data
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([[strategy_name], [data_type], data.columns])
        else:
            data.columns.set_levels([strategy_name, data_type], level=[0, 1], inplace=True)

        # Handle the case where self.strategies is empty
        if self.strategies.empty:
            self.strategies = data
        else:
            # Align indices and concatenate
            common_index = self.strategies.index.intersection(data.index)
            self.strategies = pd.concat(
                [self.strategies.reindex(common_index), data.reindex(common_index)],
                axis=1,
            )

    def get_strategy_data(self, strategy_name: str, data_type: str) -> pd.DataFrame:
        """
        Get data for a specific strategy and data type (e.g., signals, positions).
        """
        try:
            return self.strategies.xs(
                (strategy_name, data_type),
                level=["strategy", "field"],
                axis=1,
            ).copy()
        except KeyError:
            return pd.DataFrame()

    def get_all_data_for_strategy(self, strategy_name: str) -> pd.DataFrame:
        """
        Get all data associated with a specific strategy.
        """
        return self.strategies.xs(
            strategy_name,
            level="strategy",
            axis=1,
            drop_level=False,
        ).copy()

    @error_logger("Error getting feature")
    def get_feature(
        self,
        feature_name: str = None,
        strategy_name: str = None,
        level: str = None,
        date_range: slice | tuple[str, str] = None,
        include_non_trading_days: bool = False,
    ) -> pd.DataFrame:
        """
        Return the data for the given feature.

        Examples:
                - get_feature('strategies') returns all strategies data.
                - get_feature('prices') returns all price data.
                - get_feature('metrics') returns all metric data.
                - get_feature('parameters') returns all operational data.
                - get_feature('fundamentals') returns all fundamental data.
                - get_feature('signals')
                - get_feature('positions')
                - get_feature('weights')

                - get_feature(feature_name='close')
                - get_feature('close')
                - get_feature('prices', level='close') returns all closing prices.

                - get_feature('strategies', level='signals') returns signals
                  for all strategies.
                - get_feature('parameters', level='eligibility') returns all
                  eligibility data.
                - get_feature('fundamentals', level='Valuation_TrailingPE')
                  returns trailing P/E ratios.

                - get_feature('strategies', 'SMA_Crossover') returns all data
                  for the 'SMA_Crossover' strategy.
                - get_feature('strategies', 'SMA_Crossover', 'signals')
                  returns all of that strategy's signals.

        Args:
                feature_name: Feature name: 'strategies', 'prices', 'metrics',
                    or 'parameters'.
                strategy_name: The name of the strategy if applicable.
                level: The level to extract (e.g., 'signals', 'positions', 'weights').
                date_range: The date range to slice the data.

        Returns:
                pd.DataFrame: The requested data slice.
        """
        if feature_name == "strategies":
            if strategy_name and level:
                data = self.get_strategy_data(strategy_name, level)
                if data.empty:
                    raise KeyError(f"Strategy {strategy_name!r} or field {level!r} not found")
            elif strategy_name:
                data = self.get_all_data_for_strategy(strategy_name)
            elif level:
                data = self._strategy_field_view(level)
            else:
                data = self.strategies
        elif feature_name in [
            "prices",
            "metrics",
            "parameters",
            "fundamentals",
        ]:
            data = getattr(self, feature_name)
            # If a specific sub-level is requested and columns are MultiIndex, extract it
            if level is not None and isinstance(data.columns, pd.MultiIndex):
                if level in data.columns.get_level_values(-1):
                    data = data.xs(level, axis=1, level=-1)
        elif feature_name in ["signals", "positions", "weights"]:
            data = self._strategy_field_view(feature_name)
        else:
            # Check in all data types
            for attr in ["prices", "metrics", "parameters", "fundamentals"]:
                if hasattr(self, attr):
                    df = getattr(self, attr)
                    if df is None or not hasattr(df, "columns"):
                        continue
                    if feature_name in df.columns.get_level_values(-1):
                        data = df.xs(feature_name, axis=1, level=-1)
                        break
            else:
                raise AttributeError(f"'InstrumentData' object has no feature '{feature_name}'")

        # Apply date range if specified
        if date_range:
            data = data.loc[date_range]

        return data

    def _strategy_field_view(self, field: str) -> pd.DataFrame:
        """Return one field for all strategies as (instrument, strategy)."""
        if self.strategies.empty:
            return self.strategies.copy()
        data = self.strategies.xs(field, axis=1, level="field")
        if isinstance(data.columns, pd.MultiIndex):
            data = data.reorder_levels(["instrument", "strategy"], axis=1).sort_index(axis=1)
        return data.copy()

    @error_logger("Error adding feature")
    def add_feature(self, feature_name: str, data: pd.DataFrame, feature_type: str):
        """
        Add a new feature to the appropriate DataFrame

        Args:
                feature_name: The name of the new feature
                data: The data for the new feature
                feature_type: The type of feature ('price', 'metric', 'signal', or 'operational')
        """
        if feature_type == "strategies":
            raise ValueError(
                "add_feature cannot infer strategy and field levels; use "
                "add_strategy_data(strategy_name, data_type, data)"
            )
        if feature_type in [
            "prices",
            "metrics",
            "parameters",
            "fundamentals",
        ]:
            getattr(self, feature_type)[feature_name] = data
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        self.on_data_change()

    @error_logger("Error getting fundamental feature")
    def _get_fundamental_feature(self, *keys: str) -> pd.DataFrame:
        """
        Get the fundamental data for the given feature(s)
        """
        if len(keys) == 1:
            return self.fundamentals[keys[0]]
        else:
            return self.fundamentals[list(keys)]

    @error_logger("Error getting instruments")
    def get_instruments(self) -> list[str]:
        return self.group_data.index.tolist()

    @error_logger("Error getting dates")
    def get_dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    @error_logger("Error getting fundamental metrics")
    def get_fundamental_metrics(self) -> list[str]:
        return self.fundamentals.columns.tolist() if self.fundamentals is not None else []

    @error_logger("Error getting fundamental data")
    def get_fundamental_data_for_date(self, date: pd.Timestamp) -> pd.DataFrame:
        return self.fundamentals.loc[date] if self.fundamentals is not None else pd.DataFrame()

    @error_logger("Error getting fundamental history data")
    def get_fundamental_history(
        self,
        metric: str,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if self.fundamentals is None:
            return pd.DataFrame()
        if start_date is None:
            start_date = self.get_dates()[0]
        if end_date is None:
            end_date = self.get_dates()[-1]
        return self.fundamentals.loc[start_date:end_date, metric]

    @error_logger("Error updatting fundamental data")
    def update_fundamental_data(
        self,
        new_data: pd.DataFrame,
        *,
        conflict_policy: str = "overwrite",
    ) -> None:
        """Merge new fundamental data with an explicit conflict policy.

        conflict_policy:
            overwrite: new non-null values replace existing values.
            preserve_existing: existing values win; new data fills gaps only.
            raise: fail if both frames contain different non-null values.
        """
        if self.fundamentals is None:
            self.fundamentals = new_data
        else:
            policy = conflict_policy.lower().strip()
            if policy == "preserve_existing":
                self.fundamentals = self.fundamentals.combine_first(new_data)
            elif policy == "overwrite":
                self.fundamentals = new_data.combine_first(self.fundamentals)
            elif policy == "raise":
                old, new = self.fundamentals.align(new_data, join="inner", axis=None)
                conflict = old.notna() & new.notna() & old.ne(new)
                if conflict.any().any():
                    raise ValueError(
                        "Fundamental data conflict detected; use "
                        "conflict_policy='overwrite' or 'preserve_existing'."
                    )
                self.fundamentals = self.fundamentals.combine_first(new_data)
            else:
                raise ValueError(
                    "conflict_policy must be one of: overwrite, preserve_existing, raise"
                )
        self.on_data_change()

    @error_logger("Error in __repr__")
    def __repr__(self) -> str:
        return (
            f"InstrumentData(instruments={len(self.get_instruments())}, "
            f"dates={len(self.get_dates())}, "
            f"price_features={self.prices.columns.tolist()}, "
            f"metric_features={self.metrics.columns.tolist()}, "
            f"parameter_features={self.parameters.columns.tolist()}, "
            f"fundamental_features={self.get_fundamental_metrics()})"
        )

    # For CrossValidation & Walk-Forward -------------------------------------------

    def slice_data(self, date_range: tuple, *, end_inclusive: bool = True) -> "InstrumentData":
        """
        Slice the data based on the given date range

        Args:
                date_range: A tuple of the start and end dates

        Returns:
                A new InstrumentData object with the sliced data
        """
        start, end = date_range

        def _coerce_boundary(value, index: pd.Index) -> pd.Timestamp:
            ts = pd.Timestamp(value)
            if not isinstance(index, pd.DatetimeIndex):
                return ts
            if index.tz is None:
                return ts.tz_localize(None) if ts.tzinfo is not None else ts
            return ts.tz_convert(index.tz) if ts.tzinfo is not None else ts.tz_localize(index.tz)

        def _slice_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
            if frame is None:
                return None
            if frame.empty:
                return frame.copy()
            if isinstance(frame.index, pd.DatetimeIndex):
                start_ts = _coerce_boundary(start, frame.index)
                end_ts = _coerce_boundary(end, frame.index)
                if end_inclusive:
                    mask = (frame.index >= start_ts) & (frame.index <= end_ts)
                else:
                    mask = (frame.index >= start_ts) & (frame.index < end_ts)
                return frame.loc[mask]
            return frame.loc[start:end]

        sliced_prices = _slice_frame(self.prices)
        sliced_strategies = _slice_frame(self.strategies)
        sliced_metrics = _slice_frame(self.metrics)
        sliced_parameters = _slice_frame(self.parameters)
        sliced_fundamentals = _slice_frame(self.fundamentals)

        return InstrumentData(
            prices=sliced_prices,
            strategies=sliced_strategies,
            metrics=sliced_metrics,
            parameters=sliced_parameters,
            group_data=self.group_data,
            group_order=self.group_order,
            fundamentals=sliced_fundamentals,
            _skip_validation=self._skip_validation,
        )

    # Methods --------------------------------------------------------------------

    @error_logger("Error filling NA values")
    def fill_na_values(
        self,
        *,
        ffill_strategies: bool = False,
        price_ffill_limit: int | None = 1,
    ):
        """
        Fill NA values using backtest-safe defaults.

        Prices may be forward-filled for valuation/reporting. Metrics are
        handled by explicit field policy so returns/P&L are never silently
        forward-filled. Strategy signals/weights are not forward-filled by
        default: missing strategy decisions become zero, while stateful
        positions can persist.
        """
        self.prepare_prices_for_valuation(limit=price_ffill_limit)
        self.prepare_metrics_for_reporting()
        self.prepare_signals_for_execution(ffill_all=ffill_strategies)
        self.prepare_positions_for_accounting()

        # Special handling for parameters (by leaf level)
        level1 = self.parameters.columns.get_level_values(1)
        # Boolean flags
        mask_bool = level1.isin(["eligibility", "active", "is_trading_day"])
        if mask_bool.any():
            self.parameters.loc[:, mask_bool] = self.parameters.loc[:, mask_bool].fillna(False)
        # Numeric units
        mask_units = level1 == "units"
        if mask_units.any():
            self.parameters.loc[:, mask_units] = self.parameters.loc[:, mask_units].fillna(0)
        # Other parameters: forward-fill
        mask_other = ~(mask_bool | mask_units)
        if mask_other.any():
            self.parameters.loc[:, mask_other] = self.parameters.loc[:, mask_other].ffill()

    def prepare_prices_for_valuation(self, *, limit: int | None = 1) -> pd.DataFrame:
        """Forward-fill short valuation gaps without carrying stale marks forever."""
        self.prices = self.prices.ffill(limit=limit)
        return self.prices

    def prepare_metrics_for_reporting(self) -> pd.DataFrame:
        """Prepare metrics using field-specific fill policies.

        Missing returns/P&L-like metrics are left missing. Forward-filling them
        can create artificial performance. Explicit cost metrics are zero-filled
        because a missing fee/slippage observation means no recorded cost by
        convention. Slower moving diagnostic metrics may be forward-filled for
        reporting continuity.
        """
        if self.metrics is None or self.metrics.empty:
            return self.metrics
        if not isinstance(self.metrics.columns, pd.MultiIndex):
            return self.metrics

        metric_level = (
            self.metrics.columns.names.index("metric")
            if "metric" in list(self.metrics.columns.names or [])
            else self.metrics.columns.nlevels - 1
        )
        metric_names = self.metrics.columns.get_level_values(metric_level).astype(str).str.lower()

        no_fill = metric_names.isin(
            [
                "returns",
                "return",
                "daily_return",
                "daily_net_return",
                "net_return",
                "pnl",
                "daily_pnl",
                "alpha_signal",
                "signal",
            ]
        )
        zero_fill = metric_names.isin(
            [
                "transaction_costs",
                "transaction_cost",
                "fees",
                "fee",
                "commission",
                "commissions",
                "slippage",
                "slippage_costs",
                "borrow_costs",
                "financing_costs",
            ]
        )
        ffill_allowed = ~(no_fill | zero_fill)

        if zero_fill.any():
            self.metrics.loc[:, zero_fill] = self.metrics.loc[:, zero_fill].fillna(0.0)
        if ffill_allowed.any():
            self.metrics.loc[:, ffill_allowed] = self.metrics.loc[:, ffill_allowed].ffill()
        return self.metrics

    def prepare_signals_for_execution(self, *, ffill_all: bool = False) -> pd.DataFrame:
        """Prepare strategy signals/weights without silently extending signals."""
        if self.strategies.empty:
            return self.strategies
        if ffill_all:
            self.strategies = self.strategies.ffill()
            return self.strategies

        if not isinstance(self.strategies.columns, pd.MultiIndex):
            self.strategies = self.strategies.fillna(0.0)
            return self.strategies

        field_level = (
            self.strategies.columns.names.index("field")
            if "field" in list(self.strategies.columns.names or [])
            else min(1, self.strategies.columns.nlevels - 1)
        )
        fields = self.strategies.columns.get_level_values(field_level)
        stateful = fields.isin(["positions", "position", "holdings"])
        stateless = ~stateful

        if stateless.any():
            self.strategies.loc[:, stateless] = self.strategies.loc[:, stateless].fillna(0.0)
        if stateful.any():
            self.strategies.loc[:, stateful] = self.strategies.loc[:, stateful].ffill().fillna(0.0)
        return self.strategies

    def prepare_positions_for_accounting(self) -> pd.DataFrame:
        """Forward-fill stateful positions, if present in strategies."""
        if self.strategies.empty or not isinstance(self.strategies.columns, pd.MultiIndex):
            return self.strategies
        field_level = (
            self.strategies.columns.names.index("field")
            if "field" in list(self.strategies.columns.names or [])
            else min(1, self.strategies.columns.nlevels - 1)
        )
        fields = self.strategies.columns.get_level_values(field_level)
        stateful = fields.isin(["positions", "position", "holdings"])
        if stateful.any():
            self.strategies.loc[:, stateful] = self.strategies.loc[:, stateful].ffill().fillna(0.0)
        return self.strategies

    @error_logger("Error updating forecasts")
    def update_forecasts(self, new_forecasts: dict[str, Any]) -> None:
        """
        Update forecasts on InstrumentData parameters.
        """
        for ticker, forecast in new_forecasts.items():
            col = (ticker, "forecasts")
            if isinstance(forecast, pd.Series):
                self.parameters[col] = forecast.reindex(self.parameters.index)
            else:
                self.parameters[col] = pd.Series(forecast, index=self.parameters.index)
        self.on_data_change()

    @error_logger("Error adding an instrument")
    def add_instrument(self, ticker: str, data: dict[str, pd.Series]):
        """
        Add an instrument to all DataFrames

        Args:
                ticker: The instrument ticker
                data: A dictionary of the instrument data
        """
        strategy_fields = {"signals", "weights", "positions", "position", "holdings"}
        if not self.strategies.empty:
            strategy_fields.update(map(str, self.strategies.columns.get_level_values("field")))
        ambiguous_fields = sorted(set(map(str, data)) & strategy_fields)
        if ambiguous_fields:
            raise ValueError(
                "add_instrument cannot infer strategy names for strategy fields "
                f"{ambiguous_fields}; add market data first, then use "
                "add_strategy_data"
            )

        for attr, series in data.items():
            if attr in self.prices.columns.get_level_values(1):
                self.prices[(ticker, attr)] = series
            elif attr in self.metrics.columns.get_level_values(1):
                self.metrics[(ticker, attr)] = series
            elif attr in self.parameters.columns.get_level_values(1):
                self.parameters[(ticker, attr)] = series
        self._validate_data()
        self.on_data_change()

    @error_logger("Error removing an instrument")
    def remove_instrument(self, ticker: str):
        """
        Remove an instrument from all DataFrames

        Args:
                ticker: The instrument ticker
        """
        if ticker in self.prices.columns.get_level_values(0):
            self.prices = self.prices.drop(ticker, axis=1, level=0)

        if not self.strategies.empty and ticker in self.strategies.columns.get_level_values(
            "instrument"
        ):
            self.strategies = self.strategies.drop(ticker, axis=1, level="instrument")

        if ticker in self.metrics.columns.get_level_values(0):
            self.metrics = self.metrics.drop(ticker, axis=1, level=0)

        if ticker in self.parameters.columns.get_level_values(0):
            self.parameters = self.parameters.drop(ticker, axis=1, level=0)
        self.on_data_change()

    # Cache invalidation -------------------------------------------------------
    def on_data_change(self) -> None:
        """Increment version to signal downstream caches to invalidate."""
        self._version += 1

    @property
    def version(self) -> int:
        return self._version

    @error_logger("Error getting instrument data")
    def get_instrument_data(self, ticker: str) -> dict[str, pd.Series]:
        """
        Return the data for the given instrument

        Args:
                ticker: The instrument ticker
        Returns:
                A dictionary of the instrument data
        """
        result: dict[str, pd.DataFrame] = {}
        # Prices
        if ticker in self.prices.columns.get_level_values(0):
            result["prices"] = self.prices.xs(ticker, axis=1, level=0)
        else:
            result["prices"] = pd.DataFrame()
        # Strategies
        if not self.strategies.empty and ticker in self.strategies.columns.get_level_values(
            "instrument"
        ):
            result["strategies"] = self.strategies.xs(ticker, axis=1, level="instrument")
        else:
            result["strategies"] = pd.DataFrame()
        # Metrics
        if ticker in self.metrics.columns.get_level_values(0):
            result["metrics"] = self.metrics.xs(ticker, axis=1, level=0)
        else:
            result["metrics"] = pd.DataFrame()
        # Parameters (a.k.a. operational)
        if ticker in self.parameters.columns.get_level_values(0):
            params_df = self.parameters.xs(ticker, axis=1, level=0)
        else:
            params_df = pd.DataFrame()
        result["parameters"] = params_df
        # Backward-compatible key
        result["operational"] = params_df
        return result


# Builders ---------------------------------------------------------------------
class InstrumentDataBuilder:
    @staticmethod
    def from_frames(
        *,
        prices: pd.DataFrame,
        metrics: pd.DataFrame,
        parameters: pd.DataFrame,
        strategies: pd.DataFrame | None = None,
        group_data: pd.Series | None = None,
        group_order: list[str] | None = None,
        fundamentals: pd.DataFrame | None = None,
    ) -> InstrumentData:
        if strategies is None:
            strategies = pd.DataFrame(index=prices.index)
        if group_data is None:
            instruments = list(dict.fromkeys(prices.columns.get_level_values(0)))
            group_data = pd.Series(["GROUP"] * len(instruments), index=instruments)
        if group_order is None:
            group_order = list(dict.fromkeys(group_data.values.tolist()))
        return InstrumentData(
            prices=prices,
            strategies=strategies,
            metrics=metrics,
            parameters=parameters,
            group_data=group_data,
            group_order=group_order,
            fundamentals=fundamentals,
        )

    @staticmethod
    @error_logger("Error updating forecasts")
    def with_forecasts(data: InstrumentData, new_forecasts: dict[str, Any]) -> InstrumentData:
        """Return InstrumentData after applying forecast updates."""
        data.update_forecasts(new_forecasts)
        return data


"""
Note: In-file UnitTests have been removed. Tests now live under
quantjourney/portfolio/_test.
"""
