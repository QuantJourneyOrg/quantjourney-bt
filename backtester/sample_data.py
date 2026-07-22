# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Deterministic sample market-data payloads for credential-free demos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _frame_payload(df: pd.DataFrame) -> dict[str, Any]:
    columns = [
        {"instrument": str(instrument), "field": str(field)}
        for instrument, field in df.columns.to_list()
    ]
    safe = df.astype(object).where(pd.notna(df), None)
    return {
        "columns": columns,
        "index": [idx.isoformat() for idx in df.index],
        "data": safe.values.tolist(),
    }


def _series_payload(series: pd.Series) -> dict[str, Any]:
    safe = series.astype(object).where(pd.notna(series), None)
    return {
        "index": [idx.isoformat() for idx in series.index],
        "data": safe.tolist(),
    }


def build_sample_bt_payload(
    *,
    instruments: list[str],
    start: str,
    end: str,
    initial_nav: float = 100.0,
) -> dict[str, Any]:
    """Build a /bt/prepare-compatible payload from deterministic OHLCV data."""
    symbols = [str(symbol).strip().upper() for symbol in instruments if str(symbol).strip()]
    if not symbols:
        symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    dates = pd.bdate_range(start=start, end=end, tz="UTC")
    if len(dates) < 260:
        dates = pd.bdate_range(end=pd.Timestamp(end, tz="UTC"), periods=260)

    t = np.arange(len(dates), dtype=float)
    price_frames: list[pd.DataFrame] = []
    return_frames: list[pd.Series] = []

    for idx, symbol in enumerate(symbols):
        drift = 0.00038 + idx * 0.00004
        cycle = 0.00065 * np.sin(t / (23.0 + idx) + idx * 0.6)
        shorter_cycle = 0.00035 * np.cos(t / (9.0 + idx * 0.5))
        stress = -0.0030 * np.exp(-0.5 * ((t - (1150.0 + idx * 5.0)) / 16.0) ** 2)
        rebound = 0.0020 * np.exp(-0.5 * ((t - (1210.0 + idx * 5.0)) / 24.0) ** 2)
        daily_return = drift + cycle + shorter_cycle + stress + rebound
        close = (80.0 + idx * 12.0) * np.exp(np.cumsum(daily_return))
        open_ = close * (1.0 + 0.0015 * np.sin(t / 5.0 + idx))
        high = np.maximum(open_, close) * (1.0 + 0.004 + 0.001 * np.sin(t / 11.0))
        low = np.minimum(open_, close) * (1.0 - 0.004 - 0.001 * np.cos(t / 13.0))
        volume = 1_000_000 + (idx + 1) * 75_000 + (50_000 * (1.0 + np.sin(t / 9.0)))

        prices = pd.DataFrame(
            {
                (symbol, "open"): open_,
                (symbol, "high"): high,
                (symbol, "low"): low,
                (symbol, "close"): close,
                (symbol, "adj_close"): close,
                (symbol, "volume"): volume,
            },
            index=dates,
        )
        price_frames.append(prices)

        returns = pd.Series(close, index=dates).pct_change(fill_method=None)
        returns.iloc[0] = 0.0
        return_frames.append(returns.rename(symbol))

    prices_df = pd.concat(price_frames, axis=1)
    prices_df.columns = pd.MultiIndex.from_tuples(prices_df.columns, names=["instrument", "field"])

    returns_df = pd.concat(return_frames, axis=1).fillna(0.0)
    portfolio_nav = initial_nav * (1.0 + returns_df.mean(axis=1)).cumprod()

    metric_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    for symbol in symbols:
        ret = returns_df[symbol]
        synthetic_nav = (1.0 + ret).cumprod()
        drawdown = synthetic_nav / synthetic_nav.cummax() - 1.0
        metrics = pd.DataFrame(
            {
                (symbol, "returns"): ret,
                (symbol, "volatility"): ret.rolling(20, min_periods=2).std().fillna(0.0),
                (symbol, "daily_pnl"): ret,
                (symbol, "transaction_costs"): 0.0,
                (symbol, "net_asset_value"): synthetic_nav,
                (symbol, "gross_asset_value"): synthetic_nav,
                (symbol, "daily_net_return"): ret,
                (symbol, "drawdown"): drawdown,
            },
            index=dates,
        )
        metric_frames.append(metrics)

        parameters = pd.DataFrame(
            {
                (symbol, "exchange"): 0.0,
                (symbol, "units"): 0.0,
                (symbol, "eligibility"): 1.0,
                (symbol, "active"): 1.0,
                (symbol, "forecasts"): 0.0,
                (symbol, "is_trading_day"): 1.0,
                (symbol, "day_type"): 1.0,
            },
            index=dates,
        )
        parameter_frames.append(parameters)

    metrics_df = pd.concat(metric_frames, axis=1)
    metrics_df.columns = pd.MultiIndex.from_tuples(
        metrics_df.columns, names=["instrument", "field"]
    )
    parameters_df = pd.concat(parameter_frames, axis=1)
    parameters_df.columns = pd.MultiIndex.from_tuples(
        parameters_df.columns, names=["instrument", "field"]
    )

    return {
        "session_id": "sample-session",
        "dataset_id": "sample-dataset",
        "instrument_names": symbols,
        "prices": _frame_payload(prices_df),
        "metrics": _frame_payload(metrics_df),
        "parameters": _frame_payload(parameters_df),
        "nav": _series_payload(portfolio_nav),
        "summary": {
            "source": "sample",
            "instruments": len(symbols),
            "dates": len(dates),
            "start": dates[0].date().isoformat(),
            "end": dates[-1].date().isoformat(),
        },
    }
