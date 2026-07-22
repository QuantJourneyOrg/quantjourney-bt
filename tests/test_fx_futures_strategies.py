# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STRATEGIES = Path(__file__).resolve().parents[1] / "strategies"


def _load(filename: str):
    path = STRATEGIES / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trend_prices(columns: list[str], periods: int = 320) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=periods, tz="UTC")
    values = {
        column: 100.0 * np.exp(np.linspace(-0.20 + i * 0.12, 0.35 - i * 0.10, periods))
        for i, column in enumerate(columns)
    }
    return pd.DataFrame(values, index=index)


def test_fx_time_series_momentum_is_unit_gross_after_warmup() -> None:
    module = _load("example_weights_23_fx_time_series_momentum.py")
    close = _trend_prices(["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"])

    signals, weights = module.build_fx_time_series_momentum(close)

    assert set(signals.iloc[-1].unique()).issubset({-1.0, 0.0, 1.0})
    assert weights.iloc[-1].abs().sum() == pytest.approx(1.0)
    assert np.isfinite(weights.to_numpy()).all()


def test_fx_cross_sectional_momentum_is_dollar_neutral() -> None:
    module = _load("example_weights_24_fx_cross_sectional_momentum.py")
    close = _trend_prices(["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"])

    signals, weights = module.build_fx_cross_sectional_momentum(close)
    last = weights.iloc[-1]

    assert (signals.iloc[-1] > 0.0).sum() == 1
    assert (signals.iloc[-1] < 0.0).sum() == 1
    assert last.sum() == pytest.approx(0.0)
    assert last.abs().sum() == pytest.approx(1.0)


def test_continuous_futures_trend_is_finite_and_unit_gross() -> None:
    module = _load("example_weights_25_continuous_futures_trend.py")
    close = _trend_prices(["MES=F", "MNQ=F", "ZN=F", "CL=F", "GC=F", "ZC=F"])

    signals, weights = module.build_continuous_futures_trend(close)

    assert set(signals.iloc[-1].unique()).issubset({-1.0, 0.0, 1.0})
    assert weights.iloc[-1].abs().sum() == pytest.approx(1.0)
    assert np.isfinite(weights.to_numpy()).all()


def test_fx_standard_lot_sizing_respects_notional_cap() -> None:
    module = _load("example_orders_19_fx_momentum_lots.py")

    lots = module.size_standard_lots(
        nav=1_000_000.0,
        price=1.10,
        atr=0.01,
        multiplier=1.0,
        lot_size=100_000.0,
    )

    assert lots == 1
    assert (
        module.size_standard_lots(
            nav=1_000_000.0,
            price=1.10,
            atr=float("nan"),
            multiplier=1.0,
            lot_size=100_000.0,
        )
        == 0
    )


def test_futures_contract_sizing_respects_risk_notional_and_margin_caps() -> None:
    module = _load("example_orders_20_futures_donchian_contracts.py")

    contracts = module.size_futures_contracts(
        nav=2_000_000.0,
        price=5_000.0,
        atr=50.0,
        multiplier=50.0,
        margin=15_840.0,
    )

    assert contracts == 2
