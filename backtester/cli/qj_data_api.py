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

from dataclasses import dataclass
from typing import Any

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
    asset_classes: list[str]
    datasets: list[dict[str, Any]]
    example_universes: list[dict[str, Any]]
    example_symbols: list[dict[str, Any]]
    source_label: str = "live-api"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _as_string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _build_example_symbols(
    example_universes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build an index of symbols referenced by the documented example universes.

    This is deliberately not called an availability catalog: the public metadata
    contract exposes examples, not exhaustive per-symbol coverage.
    """
    symbol_map: dict[str, dict[str, Any]] = {}

    for universe in example_universes:
        universe_id = str(universe.get("id", "-"))
        universe_label = str(universe.get("label", universe_id))
        seen_in_universe: set[str] = set()
        for raw_symbol in _as_list(universe.get("symbols")):
            symbol = str(raw_symbol).strip().upper()
            if not symbol or symbol in seen_in_universe:
                continue
            seen_in_universe.add(symbol)
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

    example_symbols = sorted(symbol_map.values(), key=lambda item: item["symbol"])
    for item in example_symbols:
        item["universe_count"] = len(item["universes"])
        item["universe_labels"] = [universe["label"] for universe in item["universes"]]
        item["label"] = f"{item['symbol']} ({item['universe_count']} universe(s))"

    return example_symbols


def build_qj_data_snapshot(
    *,
    base_url: str,
    help_doc: dict[str, Any],
    catalog_doc: dict[str, Any],
    granularities_doc: dict[str, Any],
    sources_doc: dict[str, Any] | None = None,
    source_label: str = "live-api",
) -> QJDataSnapshot:
    sources = _as_dict_list((sources_doc or {}).get("sources")) or _as_dict_list(
        catalog_doc.get("sources")
    )
    granularities = _as_dict_list(granularities_doc.get("granularities")) or _as_dict_list(
        catalog_doc.get("granularities")
    )
    example_universes = _as_dict_list(catalog_doc.get("example_universes"))

    return QJDataSnapshot(
        base_url=base_url,
        help_doc=help_doc,
        catalog_doc=catalog_doc,
        granularities_doc=granularities_doc,
        sources_doc=sources_doc,
        sources=sources,
        granularities=granularities,
        asset_classes=_as_string_list(catalog_doc.get("asset_classes")),
        datasets=_as_dict_list(catalog_doc.get("datasets")),
        example_universes=example_universes,
        example_symbols=_build_example_symbols(example_universes),
        source_label=source_label,
    )


def _get_json_metadata(client: APIClient, endpoint: str) -> dict[str, Any]:
    payload = client.get(endpoint)
    if not isinstance(payload, dict):
        raise APIError(f"Expected a JSON object from the public metadata endpoint {endpoint}.")
    return payload


def fetch_qj_data_snapshot(
    base_url: str = DEFAULT_API_BASE_URL,
    timeout: int = 20,
) -> QJDataSnapshot:
    """Fetch the unauthenticated public metadata snapshot.

    The metadata contract is public, so this helper intentionally never reads or
    transmits ``QJ_API_KEY``. Authenticated market-data access remains separate.
    """
    client = APIClient(
        base_url=base_url,
        api_key=None,
        timeout=timeout,
        enable_cache=False,
    )

    help_doc = _get_json_metadata(client, "/bt/meta/help")
    catalog_doc = _get_json_metadata(client, "/bt/meta/catalog")
    granularities_doc = _get_json_metadata(client, "/bt/meta/granularities")

    try:
        sources_doc = _get_json_metadata(client, "/bt/meta/sources")
    except APIError:
        sources_doc = None

    return build_qj_data_snapshot(
        base_url=base_url,
        help_doc=help_doc,
        catalog_doc=catalog_doc,
        granularities_doc=granularities_doc,
        sources_doc=sources_doc,
        source_label="live-api",
    )
