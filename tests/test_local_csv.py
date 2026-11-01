# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Focused local CSV trust-boundary and offline-routing checks."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from backtester.engines import benchmark
from backtester.local_data import load_csv_data, load_local_data, prepare_csv_data


@pytest.fixture
def observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "adj_close": close,
                "volume": 1000.0,
            }
            for date, offset in [("2020-01-02", 0), ("2020-01-03", 1), ("2020-01-06", 2)]
            for symbol, base in [("AAPL", 100.5), ("MSFT", 200.5)]
            for close in [base + offset]
        ]
    )


def test_csv_and_memory_share_data_without_mutating_input(tmp_path, observations) -> None:
    original = observations.copy(deep=True)
    path = tmp_path / "prices.csv"
    observations.to_csv(path, index=False)
    disk = load_csv_data(str(path))
    memory = prepare_csv_data(observations)
    for key in ["prices", "metrics", "parameters"]:
        pd.testing.assert_frame_equal(disk[key], memory[key])
    pd.testing.assert_frame_equal(observations, original)
    assert disk["local_data"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert disk["parameters"].xs("forecasts", axis=1, level=1).isna().all().all()


@pytest.mark.parametrize(
    "defect",
    [
        "column",
        "duplicate",
        "missing_bar",
        "nonfinite",
        "negative",
        "volume",
        "high",
        "low",
        "basis",
        "intraday",
        "symbol",
    ],
)
def test_invalid_observations_fail_closed(observations, defect) -> None:
    frame = observations.copy()
    if defect == "column":
        frame = frame.drop(columns="open")
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "missing_bar":
        frame = frame.iloc[1:]
    else:
        field, value = {
            "nonfinite": ("open", np.inf),
            "negative": ("close", -1.0),
            "volume": ("volume", -1.0),
            "high": ("high", 1.0),
            "low": ("low", 1000.0),
            "basis": ("adj_close", 50.0),
            "intraday": ("date", "2020-01-02T12:00:00Z"),
            "symbol": ("symbol", "../AAPL"),
        }[defect]
        frame.loc[0, field] = value
    with pytest.raises(ValueError):
        prepare_csv_data(frame)


def test_selection_never_silently_shortens_or_drops_instruments(observations) -> None:
    with pytest.raises(ValueError, match="coverage"):
        prepare_csv_data(observations, start="2000-01-01")
    with pytest.raises(ValueError, match="missing requested"):
        prepare_csv_data(observations, instruments=["SPY"])
    with pytest.raises(ValueError, match="Specify instruments"):
        prepare_csv_data(observations, instruments=[])


class LocalStrategy(Backtester):
    supports_csv_config = True

    def _compute_signals(self) -> pd.DataFrame:
        return self.data.get_feature("adj_close").notna().astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        return self.signals.div(len(self.instruments))


@pytest.mark.parametrize("input_kind", ["csv", "parquet", "dataframe"])
def test_generic_local_input_keeps_strategy_selection(
    tmp_path, observations, monkeypatch, input_kind
) -> None:
    async def forbidden(*args, **kwargs):
        raise AssertionError("Local input attempted authenticated data access")

    monkeypatch.setattr(Backtester, "_get_sdk_client", forbidden)
    monkeypatch.delenv("QJ_SAMPLE_DATA", raising=False)
    original = observations.copy(deep=True)
    data = observations
    if input_kind != "dataframe":
        data = tmp_path / f"prices.{input_kind}"
        if input_kind == "csv":
            observations.to_csv(data, index=False)
        else:
            observations.to_parquet(data, index=False, engine="pyarrow")
    strategy = LocalStrategy(
        source="local",
        local_data=data,
        instruments=["MSFT"],
        backtest_period={"start": "2020-01-03", "end": "2020-01-06"},
        skip_analysis=True,
        reports_directory=str(tmp_path / "reports"),
    )
    asyncio.run(strategy.run_strategy())
    assert strategy._sdk_client is None
    assert strategy.instruments == ["MSFT"]
    assert list(strategy.signals.columns) == ["MSFT"]
    assert len(strategy.signals) == 2
    assert strategy.backtest_period.start == "2020-01-03"
    assert strategy._build_run_metadata()["local_data"]["input_format"] == input_kind
    pd.testing.assert_frame_equal(observations, original)


def test_local_input_readers_match_and_label_snapshot_identity(tmp_path, observations) -> None:
    csv_path = tmp_path / "prices.csv"
    parquet_path = tmp_path / "prices.parquet"
    observations.to_csv(csv_path, index=False)
    observations.to_parquet(parquet_path, index=False, engine="pyarrow")
    inputs = [load_local_data(item) for item in (csv_path, parquet_path, observations)]
    for result in inputs[1:]:
        for key in ("prices", "metrics", "parameters", "nav"):
            if key == "nav":
                pd.testing.assert_series_equal(inputs[0][key], result[key])
            else:
                pd.testing.assert_frame_equal(inputs[0][key], result[key])
    assert (
        inputs[1]["local_data"]["sha256"] == hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    )
    snapshot = inputs[2]["local_data"]
    assert snapshot["file_name"] is None
    assert snapshot["sha256_basis"] == "dataframe-selected-prices-csv-v1"
    reordered = load_local_data(observations.iloc[::-1])
    assert reordered["local_data"]["sha256"] == snapshot["sha256"]


def test_local_input_is_detached_and_explicit_input_precedes_environment(
    tmp_path, observations, monkeypatch
) -> None:
    monkeypatch.delenv("QJ_SAMPLE_DATA", raising=False)
    monkeypatch.setenv("QJ_DATA_CSV", "unused.csv")
    monkeypatch.setenv("QJ_DATA_FILE", "unused.parquet")
    settings = {
        "instruments": ["AAPL"],
        "backtest_period": {"start": "2020-01-02", "end": "2020-01-06"},
        "skip_analysis": True,
        "reports_directory": str(tmp_path / "reports"),
    }
    with pytest.raises(ValueError, match="QJ_DATA_FILE and QJ_DATA_CSV"):
        LocalStrategy(**settings)
    with pytest.raises(ValueError, match="local_data and data_csv"):
        LocalStrategy(local_data=observations, data_csv="prices.csv", **settings)
    strategy = LocalStrategy(source="local", local_data=observations, **settings)
    observations.loc[:, "close"] = 0.0
    asyncio.run(strategy.run_strategy())
    assert strategy.data.get_feature("close").iloc[0, 0] == 100.5


@pytest.mark.parametrize("input_kind", ["csv", "parquet", "dataframe"])
def test_local_input_invalid_prices_fail_for_each_reader(
    tmp_path, observations, input_kind
) -> None:
    observations.loc[0, "high"] = 1.0
    data = observations
    if input_kind != "dataframe":
        data = tmp_path / f"invalid.{input_kind}"
        if input_kind == "csv":
            observations.to_csv(data, index=False)
        else:
            observations.to_parquet(data, index=False, engine="pyarrow")
    with pytest.raises(ValueError, match="incoherent OHLC"):
        load_local_data(data)


def test_local_input_missing_reader_and_legacy_suffix_rules(
    tmp_path, observations, monkeypatch
) -> None:
    legacy_path = tmp_path / "prices"
    observations.to_csv(legacy_path, index=False)
    assert load_csv_data(str(legacy_path))["local_data"]["input_format"] == "csv"
    with pytest.raises(ValueError, match=".csv or .parquet"):
        load_local_data(legacy_path)

    def missing_reader(*args, **kwargs):
        raise ImportError("reader missing")

    parquet_path = tmp_path / "prices.parquet"
    parquet_path.write_bytes(b"test-only unread parquet")
    monkeypatch.setattr(pd, "read_parquet", missing_reader)
    with pytest.raises(ImportError, match=r"quantjourney-bt\[parquet\]"):
        load_local_data(parquet_path)


@pytest.mark.parametrize("friendly", ["0", "1"])
def test_cloud_missing_credentials_fail_even_with_relaxed_fetch(
    tmp_path, monkeypatch, friendly
) -> None:
    monkeypatch.setenv("QJ_FRIENDLY_ERRORS", friendly)
    monkeypatch.setenv("QJ_REQUIRE_CREDENTIALS", "1")
    for key in (
        "QJ_DATA_CSV",
        "QJ_DATA_FILE",
        "QJ_SAMPLE_DATA",
        "QJ_API_KEY",
        "QJ_EMAIL",
        "QJ_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    strategy = LocalStrategy(
        instruments=["AAPL"],
        backtest_period={"start": "2020-01-02", "end": "2020-01-06"},
        strict_data_fetch=False,
        skip_analysis=True,
        reports_directory=str(tmp_path / "reports"),
    )
    with pytest.raises(ValueError, match="No credentials set"):
        asyncio.run(strategy.run_strategy())


def _strategy(path: Path, **kwargs) -> LocalStrategy:
    return LocalStrategy(
        data_csv=path,
        csv_config=True,
        instruments=["IGNORED"],
        backtest_period={"start": "2000-01-01", "end": "2030-01-01"},
        skip_analysis=True,
        **kwargs,
    )


def test_local_execution_never_initializes_network_clients(
    tmp_path, observations, monkeypatch
) -> None:
    path = tmp_path / "prices.csv"
    observations.to_csv(path, index=False)

    async def forbidden(*args, **kwargs):
        raise AssertionError("A local run attempted network data access")

    monkeypatch.setattr(Backtester, "_get_sdk_client", forbidden)
    monkeypatch.delenv("QJ_SAMPLE_DATA", raising=False)
    strategy = _strategy(path, reports_directory=str(tmp_path / "reports"))
    asyncio.run(strategy.run_strategy())
    assert strategy._sdk_client is None
    assert strategy.instruments == ["AAPL", "MSFT"]
    assert strategy.backtest_period.start == "2020-01-02"
    assert strategy._source == "local_csv"
    assert strategy._build_run_metadata()["local_data"]["csv_config"] is True
    assert strategy.portfolio_data is not None
    assert not strategy.portfolio_data.net_asset_value.empty


@pytest.mark.parametrize("sample_env", [False, True])
def test_sample_conflicts_and_override_opt_in(tmp_path, monkeypatch, sample_env) -> None:
    monkeypatch.delenv("QJ_SAMPLE_DATA", raising=False)
    if sample_env:
        monkeypatch.setenv("QJ_SAMPLE_DATA", "1")
    with pytest.raises(ValueError, match="cannot be combined"):
        _strategy(tmp_path / "prices.csv", source="yfinance" if sample_env else "sample")
    monkeypatch.delenv("QJ_SAMPLE_DATA", raising=False)
    with pytest.raises(ValueError, match="does not opt in"):
        Backtester(data_csv="prices.csv", csv_config=True)


def test_invalid_local_data_raises_even_with_relaxed_cloud_options(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("QJ_SAMPLE_DATA", raising=False)
    strategy = _strategy(tmp_path / "absent.csv", strict_data_fetch=False, allow_partial_data=True)
    with pytest.raises(FileNotFoundError):
        asyncio.run(strategy.run_strategy())


def test_local_benchmark_does_not_fall_back_to_providers(observations, monkeypatch) -> None:
    async def forbidden(*args, **kwargs):
        raise AssertionError("Local benchmark attempted a provider call")

    monkeypatch.setattr(benchmark, "_fetch_benchmark_via_sdk", forbidden)
    monkeypatch.setattr(benchmark, "_get_returns_via_yfinance", forbidden)
    if hasattr(benchmark, "_get_returns_via_connector"):
        monkeypatch.setattr(benchmark, "_get_returns_via_connector", forbidden)
    prices = prepare_csv_data(observations)["prices"]
    args = (
        [None, prices.index]
        if "data_connector" in inspect.signature(benchmark.get_benchmark_returns).parameters
        else [prices.index]
    )
    missing = asyncio.run(benchmark.get_benchmark_returns(*args, symbol="SPY", local_prices=prices))
    present = asyncio.run(
        benchmark.get_benchmark_returns(*args, symbol="AAPL", local_prices=prices)
    )
    assert missing is None
    assert present is not None and present.notna().sum() == 2


@pytest.mark.parametrize("symbol", ["SPY", "AAPL"])
def test_public_plot_uses_only_the_configured_local_benchmark(
    observations, monkeypatch, symbol
) -> None:
    from types import SimpleNamespace

    from backtester.engines.performance import (
        StrategyPerformanceAnalysis,
        StrategyPerformanceConfig,
    )
    from backtester.portfolio import PortfolioPlots

    analysis = object.__new__(StrategyPerformanceAnalysis)
    prices = prepare_csv_data(observations)["prices"]
    analysis.config = StrategyPerformanceConfig(
        local_data=True, benchmark={"symbol": symbol, "name": symbol}
    )
    analysis.instruments_data = SimpleNamespace(prices=prices)
    analysis.portfolio_data = SimpleNamespace(returns=pd.Series(0.0, index=prices.index))
    analysis.portfolio_calc = object()
    analysis.instrument_calc = object()
    captured = []
    monkeypatch.setattr(
        PortfolioPlots, "plot_cumulative_returns", lambda **kwargs: captured.append(kwargs)
    )
    dict(analysis._public_plot_definitions())["cumulative_returns"]()
    assert captured[0]["instrument_calc"] is None
    assert (captured[0]["benchmark_returns"] is None) == (symbol == "SPY")


def test_direct_cloud_relaxed_fetch_keeps_legacy_skip(tmp_path, monkeypatch) -> None:
    for key in (
        "QJ_DATA_CSV",
        "QJ_DATA_FILE",
        "QJ_SAMPLE_DATA",
        "QJ_FRIENDLY_ERRORS",
        "QJ_REQUIRE_CREDENTIALS",
        "QJ_API_KEY",
        "QJ_EMAIL",
        "QJ_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    strategy = LocalStrategy(
        instruments=["AAPL"],
        backtest_period={"start": "2020-01-02", "end": "2020-01-06"},
        strict_data_fetch=False,
        skip_analysis=True,
        reports_directory=str(tmp_path / "reports"),
    )
    assert asyncio.run(strategy.run_strategy()) is None
    assert strategy.portfolio_data is None
