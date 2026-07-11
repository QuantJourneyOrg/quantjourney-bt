# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

from backtester.cli.qj_data import main
from backtester.cli.qj_data_api import _build_headers, build_qj_data_snapshot


def test_qj_data_snapshot_normalizes_public_metadata() -> None:
    snapshot = build_qj_data_snapshot(
        base_url="https://api.quantjourney.cloud",
        help_doc={
            "title": "QuantJourney Backtester Help",
            "snapshot_date": "2026-07-09",
        },
        catalog_doc={
            "asset_classes": ["equity", "etf"],
            "datasets": [{"id": "prepared_prices", "label": "Prepared OHLCV price frames"}],
            "example_universes": [{"id": "us_mega_cap_tech", "symbols": ["AAPL", "MSFT"]}],
            "sources": [{"id": "yfinance", "label": "Yahoo Finance"}],
        },
        granularities_doc={
            "granularities": [{"id": "1d", "category": "eod", "label": "Daily bars"}],
        },
        sources_doc=None,
    )

    assert snapshot.base_url == "https://api.quantjourney.cloud"
    assert snapshot.asset_classes == ["equity", "etf"]
    assert snapshot.sources[0]["id"] == "yfinance"
    assert snapshot.granularities[0]["id"] == "1d"
    assert snapshot.datasets[0]["id"] == "prepared_prices"
    assert snapshot.example_universes[0]["id"] == "us_mega_cap_tech"
    assert snapshot.available_symbols[0]["symbol"] == "AAPL"
    assert snapshot.available_symbols[1]["symbol"] == "MSFT"


def test_qj_data_main_is_callable() -> None:
    assert callable(main)


def test_qj_data_build_headers_adds_bearer_token() -> None:
    headers = _build_headers(api_key="test-key")

    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer test-key"
