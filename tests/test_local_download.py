# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Offline checks for the repository's Yahoo snapshot downloader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf

from backtester.local_data import load_local_data

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "local_yahoo_downloader", ROOT / "_local" / "download_yfinance.py"
)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


@pytest.fixture
def yahoo_bars() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [1000, 1100, 1200],
        },
        index=dates,
    )


def test_download_uses_adjusted_daily_prices_and_keeps_fixed_symbols(monkeypatch, yahoo_bars):
    calls = []

    def download(symbols, **kwargs):
        calls.append((symbols, kwargs))
        return pd.concat({symbol: yahoo_bars for symbol in symbols}, axis=1)

    monkeypatch.setattr(yf, "download", download)
    result = downloader.download_observations(["MSFT", "AAPL"], "2020-01-02", "2020-01-07")
    assert set(result["symbol"]) == {"MSFT", "AAPL"}
    assert result["close"].equals(result["adj_close"])
    assert calls[0][0] == ["MSFT", "AAPL"]
    assert calls[0][1]["auto_adjust"] is True
    assert calls[0][1]["interval"] == "1d"
    assert calls[0][1]["end"] == "2020-01-07"


def test_download_rejects_missing_symbol_sessions(monkeypatch, yahoo_bars):
    raw = pd.concat({"AAPL": yahoo_bars, "MSFT": yahoo_bars}, axis=1)
    raw.loc[raw.index[1], "MSFT"] = float("nan")
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="missing bars"):
        downloader.download_observations(["AAPL", "MSFT"], "2020-01-02", "2020-01-07")


def test_single_symbol_flat_response_is_supported(monkeypatch, yahoo_bars):
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: yahoo_bars)
    result = downloader.download_observations(["AAPL"], "2020-01-02", "2020-01-07")
    assert result["symbol"].tolist() == ["AAPL"] * 3
    assert result["date"].tolist() == ["2020-01-02", "2020-01-03", "2020-01-06"]


def test_snapshot_formats_match_and_metadata_is_separate(monkeypatch, yahoo_bars, tmp_path):
    monkeypatch.setattr(
        yf,
        "download",
        lambda symbols, **kwargs: pd.concat(dict.fromkeys(symbols, yahoo_bars), axis=1),
    )
    symbols = ["MSFT", "AAPL"]
    observations = downloader.download_observations(symbols, "2020-01-02", "2020-01-07")
    selection = {
        "start": "2016-10-31",
        "end_exclusive": "2026-10-31",
        "companies": [{"symbol": symbol, "company": symbol} for symbol in symbols],
        "selection": "Synthetic test selection",
        "constituent_source": "https://example.invalid/selection",
        "limitations": ["Test fixture"],
    }
    original = json.dumps(selection, sort_keys=True)
    output = tmp_path / "snapshot"
    metadata = downloader.write_snapshot(
        observations, selection, output, "2020-01-02", "2020-01-07", "test-version"
    )
    csv_file = output / metadata["files"]["csv"]["file"]
    parquet_file = output / metadata["files"]["parquet"]["file"]
    pd.testing.assert_frame_equal(
        load_local_data(csv_file)["prices"],
        load_local_data(parquet_file)["prices"],
        check_dtype=False,
        check_exact=True,
    )
    assert metadata["files"]["csv"]["sha256"] == hashlib.sha256(csv_file.read_bytes()).hexdigest()
    assert metadata["provider_version"] == "test-version"
    assert metadata["row_count"] == 6
    assert json.dumps(selection, sort_keys=True) == original
    assert len(list(output.glob("*.json"))) == 1


def test_existing_snapshot_is_preserved_before_network_access(monkeypatch, tmp_path):
    output = tmp_path / "snapshot"
    output.mkdir()
    sentinel = output / "existing.csv"
    sentinel.write_text("original\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing output must fail before network access")

    monkeypatch.setattr(downloader, "download_observations", forbidden)
    with pytest.raises(SystemExit) as error:
        downloader.main(["--output-dir", str(output)])
    assert error.value.code == 2
    assert sentinel.read_text(encoding="utf-8") == "original\n"


def test_committed_selection_contains_no_price_download_metadata():
    selection = downloader.read_selection()
    assert len(selection["companies"]) == 50
    assert not {"created_at", "provider_version", "files", "sha256"}.intersection(selection)
    assert all("rows" not in company for company in selection["companies"])
    assert Path(downloader.SELECTION_FILE).name == "universe_50.json"
