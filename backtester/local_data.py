# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Validated local daily equity data for the existing engine data objects."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRICE_FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]


def prepare_local_data(
    observations: pd.DataFrame,
    *,
    instruments: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Validate observations and initialize observed, derived and pre-trade state."""
    frame = observations.copy(deep=True)
    if isinstance(frame.columns, pd.MultiIndex) or not frame.columns.is_unique:
        raise ValueError("Local data requires unique, flat date, symbol, and OHLCV columns")
    required = ["date", "symbol", *PRICE_FIELDS]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Local CSV is missing columns: {missing}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Local CSV is empty or contains missing values")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if not frame["symbol"].str.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,31}").all():
        raise ValueError("Local CSV contains an invalid equity symbol")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("Local CSV requires daily session dates at midnight UTC")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Local CSV contains duplicate symbol/date observations")
    for field in PRICE_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    if not np.isfinite(frame[PRICE_FIELDS].to_numpy(dtype=float)).all():
        raise ValueError("Local CSV contains nonfinite numeric values")
    prices = frame[PRICE_FIELDS[:-1]]
    if prices.le(0).any().any() or frame["volume"].lt(0).any():
        raise ValueError("Local CSV requires positive prices and nonnegative volume")
    price_upper = frame[["open", "close"]].max(axis=1)
    price_lower = frame[["open", "close"]].min(axis=1)
    rtol = 8 * np.finfo(float).eps
    invalid = (
        ((price_upper > frame["high"]) & ~np.isclose(price_upper, frame["high"], rtol=rtol, atol=0))
        | ((frame["low"] > price_lower) & ~np.isclose(frame["low"], price_lower, rtol=rtol, atol=0))
        | (
            (frame["low"] > frame["high"])
            & ~np.isclose(frame["low"], frame["high"], rtol=rtol, atol=0)
        )
    )
    if invalid.any():
        row = frame.loc[invalid].iloc[0]
        raise ValueError(f"Local CSV has incoherent OHLC for {row['symbol']} at {row['date']}")
    if not np.isclose(frame["close"], frame["adj_close"], rtol=rtol, atol=0).all():
        raise ValueError("Local CSV requires all OHLC prices on the adjusted-close basis")

    available = sorted(frame["symbol"].unique().tolist())
    selected = list(
        dict.fromkeys(
            str(s).strip().upper() for s in (available if instruments is None else instruments)
        )
    )
    if not selected:
        raise ValueError("Specify instruments or use the explicit CSV configuration option")
    unavailable = sorted(set(selected) - set(available))
    if unavailable:
        raise ValueError(f"Local CSV is missing requested instruments: {unavailable}")
    frame = frame[frame["symbol"].isin(selected)]
    first, last = frame["date"].min(), frame["date"].max()
    lower = pd.Timestamp(start, tz="UTC") if start else first
    upper = pd.Timestamp(end, tz="UTC") if end else last
    if lower > upper or lower < first or upper > last:
        raise ValueError(
            "Requested period is outside local CSV coverage; use --csv-config explicitly"
        )
    frame = frame[frame["date"].between(lower, upper)]
    if frame.empty:
        raise ValueError("Local CSV has no observations in the requested period")
    wide = frame.pivot(index="date", columns="symbol", values=PRICE_FIELDS).swaplevel(axis=1)
    columns = pd.MultiIndex.from_product([selected, PRICE_FIELDS], names=["instrument", "field"])
    wide = wide.reindex(columns=columns).sort_index()
    if wide.isna().any().any():
        raise ValueError("Local CSV has missing bars in the selected symbol/date panel")

    metrics_parts, parameter_parts = {}, {}
    for symbol in selected:
        close = wide[(symbol, "adj_close")]
        returns = close.pct_change(fill_method=None)
        nav = close / close.iloc[0]
        derived = {
            "returns": returns,
            "volatility": returns.rolling(20, min_periods=2).std(),
            "daily_pnl": nav.diff(),
            "net_asset_value": nav,
            "gross_asset_value": nav,
            "daily_net_return": returns,
            "drawdown": nav / nav.cummax() - 1,
            "transaction_costs": pd.Series(0.0, index=wide.index),
        }
        parameters = {
            "exchange": pd.Series(np.nan, index=wide.index),
            "day_type": pd.Series(np.nan, index=wide.index),
            "forecasts": pd.Series(np.nan, index=wide.index),
            "units": pd.Series(0.0, index=wide.index),
            "eligibility": pd.Series(True, index=wide.index),
            "active": pd.Series(True, index=wide.index),
            "is_trading_day": pd.Series(True, index=wide.index),
        }
        metrics_parts[symbol] = pd.DataFrame(derived)
        parameter_parts[symbol] = pd.DataFrame(parameters)
    return {
        "prices": wide,
        "metrics": pd.concat(metrics_parts, axis=1),
        "parameters": pd.concat(parameter_parts, axis=1),
        # Initial portfolio state is all cash; execution replaces this NAV.
        "nav": pd.Series(1.0, index=wide.index, name="nav"),
        "instrument_names": selected,
        "summary": {
            "source": "local_csv",
            "instruments": len(selected),
            "dates": len(wide),
            "start": wide.index[0].date().isoformat(),
            "end": wide.index[-1].date().isoformat(),
        },
    }


prepare_csv_data = prepare_local_data


def load_local_data(
    data: str | os.PathLike[str] | pd.DataFrame, **selection: Any
) -> dict[str, Any]:
    """Read a local file or snapshot without fetching market data."""
    return _load_local_data(data, file_format=None, **selection)


def load_csv_data(path: str, **selection: Any) -> dict[str, Any]:
    """Preserve CSV-only parsing, including files without a suffix."""
    return _load_local_data(path, file_format="csv", **selection)


def _load_local_data(
    data: str | os.PathLike[str] | pd.DataFrame,
    *,
    file_format: str | None,
    **selection: Any,
) -> dict[str, Any]:
    file_name: str | None = None
    if isinstance(data, pd.DataFrame):
        observations = data.copy(deep=True)
        content = b""
        input_format = "dataframe"
        hash_basis = "dataframe-selected-prices-csv-v1"
    else:
        file = Path(data)
        input_format = file_format or file.suffix.lower().lstrip(".")
        if input_format not in {"csv", "parquet"}:
            raise ValueError("Local data requires a .csv or .parquet file, or a pandas DataFrame")
        content = file.read_bytes()
        file_name = file.name
        hash_basis = "file_bytes"
        if input_format == "csv":
            observations = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
        else:
            try:
                observations = pd.read_parquet(io.BytesIO(content), engine="pyarrow")
            except ImportError as error:
                raise ImportError(
                    "Parquet input requires the optional reader: "
                    'pip install "quantjourney-bt[parquet]"'
                ) from error
    result = prepare_local_data(observations, **selection)
    if input_format == "dataframe":
        content = (
            result["prices"]
            .sort_index(axis=1)
            .to_csv(
                float_format="%.17g",
                date_format="%Y-%m-%dT%H:%M:%S.%f%z",
            )
            .encode("utf-8")
        )
    result["summary"]["source"] = f"local_{input_format}"
    result["local_data"] = {
        "file_name": file_name,
        "input_format": input_format,
        "sha256": hashlib.sha256(content).hexdigest(),
        "sha256_basis": hash_basis,
        "price_basis": "all OHLC adjusted; volume as supplied",
        "requested_start": selection.get("start"),
        "requested_end": selection.get("end"),
        "actual_start": result["summary"]["start"],
        "actual_end": result["summary"]["end"],
        "symbols": result["instrument_names"],
        "rows": len(result["prices"]),
        "state_initialization": "observed prices; derived input returns; neutral pre-trade state",
    }
    return result
