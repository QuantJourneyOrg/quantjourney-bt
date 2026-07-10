"""
Lightweight schema validators for portfolio data structures.

If Pandera is installed, these can be extended to use pa.DataFrameSchema,
but current implementation uses pragmatic runtime checks to avoid
dependency constraints in tight environments.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd

try:
    import pandera as pa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pa = None  # sentinel

STRICT_PANDERA = os.getenv("QJ_USE_PANDERA", "").lower() in ("1", "true", "yes")


REQUIRED_PRICE_FIELDS = {"open", "high", "low", "close", "adj_close", "volume"}
REQUIRED_METRIC_FIELDS = {
    "returns",
    "volatility",
    "daily_pnl",
    "transaction_costs",
    "net_asset_value",
    "gross_asset_value",
    "daily_net_return",
    "drawdown",
}
REQUIRED_PARAMETER_FIELDS = {
    "exchange",
    "units",
    "eligibility",
    "active",
    "forecasts",
    "is_trading_day",
    "day_type",
}


def _ensure_dt_index(idx: pd.Index, name: str) -> None:
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be a DatetimeIndex (got {type(idx)})")
    # tz awareness is handled upstream; allow either here
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{name}: index must be monotonic increasing")
    if idx.has_duplicates:
        raise ValueError(f"{name}: index contains duplicates")


def _require_multiindex(df: pd.DataFrame, name: str, nlevels: int) -> None:
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != nlevels:
        raise ValueError(f"{name}: expected a {nlevels}-level MultiIndex for columns")


def _require_fields(df: pd.DataFrame, level: int, required: Iterable[str], name: str) -> None:
    have = set(map(str, df.columns.get_level_values(level)))
    missing = set(required) - have
    if missing:
        raise ValueError(f"{name}: missing required fields at level {level}: {sorted(missing)}")


def validate_prices_frame(prices: pd.DataFrame) -> None:
    _ensure_dt_index(prices.index, name="prices")
    _require_multiindex(prices, name="prices", nlevels=2)
    # Level names are advisory; presence of required fields is enforced
    _require_fields(prices, level=1, required=REQUIRED_PRICE_FIELDS, name="prices")
    # Additional validation: volume must be non-negative
    vol = prices.xs("volume", axis=1, level=1, drop_level=False)
    if (vol < 0).any().any():
        raise ValueError("prices: 'volume' contains negative values")
    # Optional Pandera strict check: numeric dtypes and MultiIndex columns validation
    if STRICT_PANDERA and pa is not None:
        try:
            # Build a dynamic schema mapping existing columns to float
            cols = {tuple(col): pa.Column(float, required=True) for col in prices.columns}
            schema = pa.DataFrameSchema(columns=cols, coerce=True, strict=False)
            schema.validate(prices)
        except Exception as e:
            raise ValueError(f"prices: Pandera validation failed: {e}") from e


def validate_metrics_frame(metrics: pd.DataFrame) -> None:
    _ensure_dt_index(metrics.index, name="metrics")
    _require_multiindex(metrics, name="metrics", nlevels=2)
    _require_fields(metrics, level=1, required=REQUIRED_METRIC_FIELDS, name="metrics")
    if STRICT_PANDERA and pa is not None:
        try:
            cols = {tuple(col): pa.Column(float, required=True) for col in metrics.columns}
            schema = pa.DataFrameSchema(columns=cols, coerce=True, strict=False)
            schema.validate(metrics)
        except Exception as e:
            raise ValueError(f"metrics: Pandera validation failed: {e}") from e


def validate_parameters_frame(parameters: pd.DataFrame) -> None:
    _ensure_dt_index(parameters.index, name="parameters")
    _require_multiindex(parameters, name="parameters", nlevels=2)
    _require_fields(parameters, level=1, required=REQUIRED_PARAMETER_FIELDS, name="parameters")
    lvl = parameters.columns.get_level_values(1)
    # Basic dtype sanity checks
    # Units numeric
    if parameters.loc[:, lvl == "units"].apply(pd.to_numeric, errors="coerce").isna().any().any():
        raise ValueError("parameters: 'units' must be numeric or castable to numeric")
    # Booleans not-null
    for flag in ("eligibility", "active", "is_trading_day"):
        if flag in lvl:
            if parameters.loc[:, lvl == flag].isna().any().any():
                raise ValueError(f"parameters: '{flag}' contains NaN; must be boolean")
    if STRICT_PANDERA and pa is not None:
        try:
            cols = {tuple(col): pa.Column(object, required=True) for col in parameters.columns}
            schema = pa.DataFrameSchema(columns=cols, coerce=True, strict=False)
            schema.validate(parameters)
        except Exception as e:
            raise ValueError(f"parameters: Pandera validation failed: {e}") from e


def validate_strategies_frame(strategies: pd.DataFrame | None) -> None:
    if strategies is None or strategies.empty:
        return
    if not isinstance(strategies.columns, pd.MultiIndex) or strategies.columns.nlevels != 3:
        raise ValueError("strategies: expected a 3-level MultiIndex for columns")
    if STRICT_PANDERA and pa is not None:
        try:
            cols = {tuple(col): pa.Column(float, required=False) for col in strategies.columns}
            schema = pa.DataFrameSchema(columns=cols, coerce=True, strict=False)
            schema.validate(strategies)
        except Exception as e:
            raise ValueError(f"strategies: Pandera validation failed: {e}") from e


def validate_nav_series(nav: pd.Series) -> None:
    _ensure_dt_index(nav.index, name="nav")
    if not pd.api.types.is_numeric_dtype(nav.dtype):
        raise ValueError("nav: must be numeric dtype")
    if STRICT_PANDERA and pa is not None:
        try:
            pa.SeriesSchema(pa.Float, coerce=True).validate(nav)
        except Exception as e:
            raise ValueError(f"nav: Pandera validation failed: {e}") from e


def validate_weights_frame(weights: pd.DataFrame, instruments: Iterable[str]) -> None:
    _ensure_dt_index(weights.index, name="weights")
    if not set(weights.columns) >= set(instruments):
        missing = set(instruments) - set(weights.columns)
        raise ValueError(f"weights: missing columns for instruments: {sorted(missing)}")
    if not all(
        pd.api.types.is_float_dtype(weights[c]) or pd.api.types.is_numeric_dtype(weights[c])
        for c in weights.columns
    ):
        raise ValueError("weights: all columns must be numeric")
    if STRICT_PANDERA and pa is not None:
        try:
            schema = pa.DataFrameSchema(
                {str(col): pa.Column(float, nullable=False) for col in weights.columns}, coerce=True
            )
            schema.validate(weights)
        except Exception as e:
            raise ValueError(f"weights: Pandera validation failed: {e}") from e
