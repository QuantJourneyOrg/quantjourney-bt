"""
qj_data_api — public metadata fetch + normalization for qj-data
---------------------------------------------------------------

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests

from backtester.sdk.client import APIClient, APIError

DEFAULT_API_BASE_URL = "https://api.quantjourney.cloud"


@dataclass(slots=True)
class QJDataSnapshot:
    base_url: str
    help_doc: dict[str, Any]
    catalog_doc: dict[str, Any]
    granularities_doc: dict[str, Any]
    sources_doc: dict[str, Any] | None
    sources: list[dict[str, Any]]
    granularities: list[dict[str, Any]]
    asset_classes: list[Any]
    datasets: list[dict[str, Any]]
    example_universes: list[dict[str, Any]]
    available_symbols: list[dict[str, Any]]
    source_label: str = "live-api"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _build_available_symbols(example_universes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    symbol_map: dict[str, dict[str, Any]] = {}

    for universe in example_universes:
        universe_id = str(universe.get("id", "-"))
        universe_label = str(universe.get("label", universe_id))
        for raw_symbol in _as_list(universe.get("symbols")):
            symbol = str(raw_symbol).strip().upper()
            if not symbol:
                continue
            if symbol not in symbol_map:
                symbol_map[symbol] = {
                    "id": symbol,
                    "label": symbol,
                    "symbol": symbol,
                    "universes": [],
                    "granularity_status": "Not exposed by public asset metadata",
                    "period_status": "Not exposed by public asset metadata",
                    "date_range_status": "Not exposed by public asset metadata",
                }
            symbol_map[symbol]["universes"].append(
                {
                    "id": universe_id,
                    "label": universe_label,
                }
            )

    available_symbols = sorted(symbol_map.values(), key=lambda item: item["symbol"])
    for item in available_symbols:
        item["universe_count"] = len(item["universes"])
        item["universe_labels"] = [universe["label"] for universe in item["universes"]]
        item["label"] = f"{item['symbol']} ({item['universe_count']} universe(s))"

    return available_symbols


def build_qj_data_snapshot(
    *,
    base_url: str,
    help_doc: dict[str, Any],
    catalog_doc: dict[str, Any],
    granularities_doc: dict[str, Any],
    sources_doc: dict[str, Any] | None = None,
    source_label: str = "live-api",
) -> QJDataSnapshot:
    sources = _as_list((sources_doc or {}).get("sources")) or _as_list(catalog_doc.get("sources"))
    granularities = _as_list(granularities_doc.get("granularities")) or _as_list(
        catalog_doc.get("granularities")
    )
    example_universes = _as_list(catalog_doc.get("example_universes"))

    return QJDataSnapshot(
        base_url=base_url,
        help_doc=help_doc,
        catalog_doc=catalog_doc,
        granularities_doc=granularities_doc,
        sources_doc=sources_doc,
        sources=sources,
        granularities=granularities,
        asset_classes=_as_list(catalog_doc.get("asset_classes")),
        datasets=_as_list(catalog_doc.get("datasets")),
        example_universes=example_universes,
        available_symbols=_build_available_symbols(example_universes),
        source_label=source_label,
    )


def fetch_qj_data_snapshot(
    base_url: str = DEFAULT_API_BASE_URL,
    timeout: int = 20,
    api_key: str | None = None,
) -> QJDataSnapshot:
    api_key = api_key or os.getenv("QJ_API_KEY")
    client = APIClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        enable_cache=False,
    )
    headers = _build_headers(api_key=api_key)

    help_doc = _get_json_metadata(client, "/bt/meta/help", headers=headers)
    catalog_doc = _get_json_metadata(client, "/bt/meta/catalog", headers=headers)
    granularities_doc = _get_json_metadata(client, "/bt/meta/granularities", headers=headers)

    try:
        sources_doc_any = _get_json_metadata(client, "/bt/meta/sources", headers=headers)
        sources_doc = sources_doc_any if isinstance(sources_doc_any, dict) else None
    except APIError:
        sources_doc = None

    if not isinstance(help_doc, dict) or not isinstance(catalog_doc, dict) or not isinstance(
        granularities_doc, dict
    ):
        raise APIError("Expected dictionary payloads from the core metadata endpoints.")

    return build_qj_data_snapshot(
        base_url=base_url,
        help_doc=help_doc,
        catalog_doc=catalog_doc,
        granularities_doc=granularities_doc,
        sources_doc=sources_doc,
        source_label="live-api",
    )


def _build_headers(*, api_key: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "quantjourney-bt/qj-data",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_json_metadata(
    client: APIClient,
    endpoint: str,
    *,
    headers: dict[str, str],
) -> Any:
    url = f"{client.base_url}{endpoint}"
    try:
        response = client.session.get(url, headers=headers, timeout=client.timeout)
    except requests.RequestException as exc:
        raise APIError(f"HTTP request failed: {exc}") from exc

    client._last_request_id = response.headers.get("X-Request-ID")
    content_type = response.headers.get("Content-Type", "")
    final_url = response.url

    if not response.ok:
        raise APIError(f"HTTP {response.status_code}: {response.text[:200]}")

    if "application/json" not in content_type.lower():
        raise APIError(
            f"Expected JSON from {endpoint}, got content-type '{content_type or 'unknown'}' "
            f"at {final_url}. The public metadata endpoint may be redirecting or unavailable."
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise APIError(
            f"Invalid JSON from {endpoint} at {final_url}. "
            "The public metadata endpoint may be redirecting or unavailable."
        ) from exc
