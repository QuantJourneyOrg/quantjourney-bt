# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Regression tests for tenant- and principal-scoped SDK ETag caching."""

from __future__ import annotations

import asyncio

from backtester.sdk.client import APIClient, AsyncAPIClient


def _response(data, etag='"etag-response"'):
    class Response:
        status_code = 200
        headers = {"ETag": etag}

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"success": True, "data": data}

    return Response()


def test_cache_key_is_fully_opaque_and_canonical():
    client = APIClient(
        "https://api.example.test",
        api_key="secret-token-a",
        tenant_id="tenant-a",
    )
    endpoint = "/private/positions"
    params = {"date": "2026-01-02", "filters": {"z": 2, "a": 1}}

    key = client._cache_key(endpoint, params)
    reordered_key = client._cache_key(
        endpoint,
        {"filters": {"a": 1, "z": 2}, "date": "2026-01-02"},
    )

    assert key == reordered_key
    assert len(key) == 64
    int(key, 16)
    for plaintext in (endpoint, "2026-01-02", "secret-token-a", "tenant-a"):
        assert plaintext not in key


def test_sync_token_change_scopes_and_invalidates_cache():
    client = APIClient(
        "https://api.example.test",
        api_key="secret-token-a",
        tenant_id="tenant-a",
    )
    key_a = client._cache_key("/positions", {"date": "2026-01-02"})
    assert "secret-token-a" not in key_a
    assert "tenant-a" not in key_a
    assert client.cache is not None
    client.cache.set(key_a, '"etag-a"', {"owner": "principal-a"})

    client.set_bearer_tokens("secret-token-b")
    key_b = client._cache_key("/positions", {"date": "2026-01-02"})

    assert key_b != key_a
    assert client.cache.get(key_a) is None
    assert "secret-token-b" not in key_b


def test_sync_tenant_change_scopes_and_invalidates_cache():
    client = APIClient(
        "https://api.example.test",
        api_key="shared-token",
        tenant_id="tenant-a",
    )
    key_a = client._cache_key("/positions")
    assert client.cache is not None
    client.cache.set(key_a, '"etag-a"', {"tenant": "a"})

    client.set_tenant("tenant-b")
    key_b = client._cache_key("/positions")

    assert key_b != key_a
    assert client.cache.get(key_a) is None


def test_async_token_and_tenant_changes_scope_and_invalidate_cache():
    client = AsyncAPIClient(
        "https://api.example.test",
        api_key="secret-token-a",
        tenant_id="tenant-a",
    )
    try:
        key_a = client._cache_key("/positions", {"date": "2026-01-02"})
        assert client.cache is not None
        client.cache.set(key_a, '"etag-a"', {"owner": "principal-a"})

        client.set_bearer_tokens("secret-token-b")
        key_b = client._cache_key("/positions", {"date": "2026-01-02"})
        assert key_b != key_a
        assert client.cache.get(key_a) is None

        client.cache.set(key_b, '"etag-b"', {"tenant": "a"})
        client.set_tenant("tenant-b")
        key_c = client._cache_key("/positions", {"date": "2026-01-02"})
        assert key_c != key_b
        assert client.cache.get(key_b) is None
        assert "secret-token-b" not in key_c
        assert "tenant-b" not in key_c
    finally:
        asyncio.run(client.close())


def test_sync_request_keeps_one_context_snapshot_during_transport(monkeypatch):
    client = APIClient(
        "https://api.example.test",
        api_key="secret-token-a",
        tenant_id="tenant-a",
    )
    params = {"date": "2026-01-02"}
    old_key = client._cache_key("/positions", params)
    assert client.cache is not None
    client.cache.set(old_key, '"etag-before-race"', {"owner": "cached-a"})
    captured_headers = {}

    def request_transport(request, **kwargs):
        client.set_tenant("tenant-b")
        client.set_bearer_tokens("secret-token-b")
        captured_headers.update(request.headers)
        return _response({"owner": "response-a"})

    monkeypatch.setattr(client.session, "send", request_transport)

    result = client.get("/positions", params=params)

    assert result == {"owner": "response-a"}
    assert captured_headers["Authorization"] == "Bearer secret-token-a"
    assert captured_headers["X-Tenant-Id"] == "tenant-a"
    assert captured_headers["If-None-Match"] == '"etag-before-race"'
    assert client.cache.get(old_key)["data"] == {"owner": "response-a"}
    assert client.cache.get(client._cache_key("/positions", params)) is None


def test_async_request_keeps_one_context_snapshot_during_transport(monkeypatch):
    client = AsyncAPIClient(
        "https://api.example.test",
        api_key="secret-token-a",
        tenant_id="tenant-a",
    )
    params = {"date": "2026-01-02"}
    old_key = client._cache_key("/positions", params)
    assert client.cache is not None
    client.cache.set(old_key, '"etag-before-race"', {"owner": "cached-a"})
    captured_headers = {}

    async def request_transport(request, **kwargs):
        client.set_tenant("tenant-b")
        client.set_bearer_tokens("secret-token-b")
        captured_headers.update(
            {
                "Authorization": request.headers.get("Authorization"),
                "X-Tenant-Id": request.headers.get("X-Tenant-Id"),
                "If-None-Match": request.headers.get("If-None-Match"),
            }
        )
        return _response({"owner": "response-a"})

    monkeypatch.setattr(client.client, "send", request_transport)
    try:
        result = asyncio.run(client.get("/positions", params=params))

        assert result == {"owner": "response-a"}
        assert captured_headers["Authorization"] == "Bearer secret-token-a"
        assert captured_headers["X-Tenant-Id"] == "tenant-a"
        assert captured_headers["If-None-Match"] == '"etag-before-race"'
        assert client.cache.get(old_key)["data"] == {"owner": "response-a"}
        assert client.cache.get(client._cache_key("/positions", params)) is None
    finally:
        asyncio.run(client.close())
