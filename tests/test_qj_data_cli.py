# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import Mock, call, patch

import pytest

from backtester.cli import qj_data
from backtester.cli.qj_data_api import (
    QJDataSnapshot,
    build_qj_data_snapshot,
    fetch_qj_data_snapshot,
)
from backtester.sdk.client import APIError


def _metadata_documents() -> tuple[dict, dict, dict, dict]:
    help_doc = {
        "title": "QuantJourney Backtester Help",
        "snapshot_date": "2026-07-11",
    }
    catalog_doc = {
        "asset_classes": ["equity", "etf"],
        "datasets": [{"id": "prepared_prices", "label": "Prepared OHLCV price frames"}],
        "example_universes": [
            {
                "id": "us_mega_cap_tech",
                "label": "US mega-cap tech basket",
                "symbols": ["msft", "AAPL", "AAPL"],
            }
        ],
        "sources": [{"id": "catalog-source", "label": "Catalog fallback"}],
    }
    granularities_doc = {
        "granularities": [{"id": "1d", "category": "eod", "label": "Daily bars"}],
    }
    sources_doc = {"sources": [{"id": "yfinance", "label": "Yahoo Finance"}]}
    return help_doc, catalog_doc, granularities_doc, sources_doc


def _snapshot() -> QJDataSnapshot:
    help_doc, catalog_doc, granularities_doc, sources_doc = _metadata_documents()
    return build_qj_data_snapshot(
        base_url="https://api.quantjourney.cloud",
        help_doc=help_doc,
        catalog_doc=catalog_doc,
        granularities_doc=granularities_doc,
        sources_doc=sources_doc,
    )


def test_qj_data_snapshot_normalizes_public_example_metadata() -> None:
    snapshot = _snapshot()

    assert snapshot.base_url == "https://api.quantjourney.cloud"
    assert snapshot.asset_classes == ["equity", "etf"]
    assert snapshot.sources[0]["id"] == "yfinance"
    assert snapshot.granularities[0]["id"] == "1d"
    assert snapshot.datasets[0]["id"] == "prepared_prices"
    assert snapshot.example_universes[0]["id"] == "us_mega_cap_tech"
    assert [item["symbol"] for item in snapshot.example_symbols] == ["AAPL", "MSFT"]
    assert snapshot.example_symbols[0]["universe_count"] == 1


def test_fetch_qj_data_uses_unauthenticated_public_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QJ_API_KEY", "must-not-be-read-or-sent")
    help_doc, catalog_doc, granularities_doc, sources_doc = _metadata_documents()
    client = Mock()
    client.get.side_effect = [help_doc, catalog_doc, granularities_doc, sources_doc]

    with patch("backtester.cli.qj_data_api.APIClient", return_value=client) as client_class:
        snapshot = fetch_qj_data_snapshot(
            base_url="https://metadata.example.test",
            timeout=7,
        )

    client_class.assert_called_once_with(
        base_url="https://metadata.example.test",
        api_key=None,
        timeout=7,
        enable_cache=False,
    )
    assert client.get.call_args_list == [
        call("/bt/meta/help"),
        call("/bt/meta/catalog"),
        call("/bt/meta/granularities"),
        call("/bt/meta/sources"),
    ]
    assert snapshot.sources_doc == sources_doc


def test_fetch_qj_data_falls_back_to_catalog_sources() -> None:
    help_doc, catalog_doc, granularities_doc, _sources_doc = _metadata_documents()
    client = Mock()
    client.get.side_effect = [
        help_doc,
        catalog_doc,
        granularities_doc,
        APIError("optional sources endpoint unavailable"),
    ]

    with patch("backtester.cli.qj_data_api.APIClient", return_value=client):
        snapshot = fetch_qj_data_snapshot()

    assert snapshot.sources_doc is None
    assert snapshot.sources == [{"id": "catalog-source", "label": "Catalog fallback"}]


def test_fetch_qj_data_rejects_non_object_core_payload() -> None:
    client = Mock()
    client.get.return_value = []

    with (
        patch("backtester.cli.qj_data_api.APIClient", return_value=client),
        pytest.raises(APIError, match="Expected a JSON object"),
    ):
        fetch_qj_data_snapshot()


def test_qj_bt_root_help_does_not_contact_the_api(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("backtester.cli.qj_data.fetch_qj_data_snapshot") as fetch,
        pytest.raises(SystemExit) as exc_info,
    ):
        qj_data.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "qj-bt" in output
    assert "data" in output
    fetch.assert_not_called()


def test_qj_bt_data_help_lists_sections_and_transport_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        qj_data.main(["data", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "example-symbols" in output
    assert "--json" in output
    assert "--base-url" in output


def test_qj_bt_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("backtester.cli.qj_data.fetch_qj_data_snapshot") as fetch:
        assert qj_data.main([]) == 0

    assert "qj-bt" in capsys.readouterr().out
    fetch.assert_not_called()


def test_qj_bt_data_can_exit_from_forced_interactive_browser() -> None:
    with (
        patch("backtester.cli.qj_data.fetch_qj_data_snapshot", return_value=_snapshot()),
        patch("backtester.cli.qj_data.console.status", return_value=nullcontext()),
        patch("backtester.cli.qj_data.clear_screen"),
        patch("backtester.cli.qj_data.show_home_banner"),
        patch("backtester.cli.qj_data._select", return_value="exit"),
    ):
        assert qj_data.main(["data", "--interactive"]) == 0


def test_qj_bt_data_without_tty_renders_overview_instead_of_prompting() -> None:
    with (
        patch("backtester.cli.qj_data.fetch_qj_data_snapshot", return_value=_snapshot()),
        patch("backtester.cli.qj_data.sys.stdin.isatty", return_value=False),
        patch("backtester.cli.qj_data.sys.stdout.isatty", return_value=False),
        patch("backtester.cli.qj_data.show_overview") as show_overview,
        patch("backtester.cli.qj_data._run_interactive") as interactive,
    ):
        assert qj_data.main(["data"]) == 0

    show_overview.assert_called_once()
    interactive.assert_not_called()


def test_qj_bt_data_section_renders_deterministic_table() -> None:
    with (
        patch("backtester.cli.qj_data.fetch_qj_data_snapshot", return_value=_snapshot()),
        patch("backtester.cli.qj_data.show_sources") as show_sources,
    ):
        assert qj_data.main(["data", "sources"]) == 0

    show_sources.assert_called_once()


def test_qj_bt_data_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("backtester.cli.qj_data.fetch_qj_data_snapshot", return_value=_snapshot()):
        assert qj_data.main(["data", "sources", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["section"] == "sources"
    assert payload["snapshot_date"] == "2026-07-11"
    assert payload["data"][0]["id"] == "yfinance"


def test_qj_data_main_reports_metadata_failure() -> None:
    with (
        patch(
            "backtester.cli.qj_data.fetch_qj_data_snapshot",
            side_effect=APIError("metadata unavailable"),
        ),
        patch("backtester.cli.qj_data.show_error") as show_error,
    ):
        assert qj_data.main(["data", "overview"]) == 1

    show_error.assert_called_once_with("Failed to load metadata: metadata unavailable")


def test_qj_bt_json_failure_is_valid_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "backtester.cli.qj_data.fetch_qj_data_snapshot",
        side_effect=APIError("metadata unavailable"),
    ):
        assert qj_data.main(["data", "sources", "--json"]) == 1

    payload = json.loads(capsys.readouterr().err)
    assert payload == {"error": "metadata unavailable"}
