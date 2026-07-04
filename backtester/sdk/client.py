"""
QuantJourney SDK - HTTP Client Module
=====================================

This module provides low-level HTTP clients for the QuantJourney API.

Classes:
    APIClient: Synchronous HTTP client using requests with connection pooling.
    AsyncAPIClient: Asynchronous HTTP client using httpx (optional dependency).
    ETagCache: Simple in-memory cache for ETag-based conditional requests.
    APIError: Exception raised for API errors with request tracing.

Features:
    - Configurable connection pooling (pool_connections, pool_maxsize)
    - ETag caching for GET requests
    - Automatic token refresh on 401 responses
    - X-Request-ID for request tracing
    - Configurable timeouts (connect, read, write)

Usage (sync):
    >>> from backtester.sdk.client import APIClient
    >>> client = APIClient(
    ...     base_url="https://api.quantjourney.cloud",
    ...     api_key="YOUR_API_KEY",
    ...     pool_connections=20,
    ...     pool_maxsize=50,
    ... )
    >>> data = client.get("/health")

Usage (async):
    >>> from backtester.sdk.client import AsyncAPIClient
    >>> async with AsyncAPIClient(
    ...     base_url="https://api.quantjourney.cloud",
    ...     api_key="YOUR_API_KEY",
    ...     max_connections=100,
    ... ) as client:
    ...     data = await client.get("/health")

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

import json
import requests
from requests.adapters import HTTPAdapter
from dataclasses import dataclass, field
import os
import uuid
import hashlib
from typing import Any, Dict, Optional

# Optional async support
try:  # pragma: no cover - optional dependency
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None

# Default pool settings for requests.Session
POOL_CONNECTIONS = 20
POOL_MAXSIZE = 50


# =============================================================================
# ETag Cache for SDK
# =============================================================================

class ETagCache:
    """
    Simple in-memory cache for ETag-based conditional requests.
    
    Stores response data keyed by URL, with ETag for validation.
    """
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached entry for URL."""
        return self._cache.get(url)
    
    def set(self, url: str, etag: str, data: Any) -> None:
        """Store response with ETag."""
        # Simple LRU-ish: remove oldest if at capacity
        if len(self._cache) >= self._max_size:
            # Remove first item (oldest)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[url] = {"etag": etag, "data": data}
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    def remove(self, url: str) -> bool:
        """Remove specific URL from cache. Returns True if removed, False if not found."""
        if url in self._cache:
            del self._cache[url]
            return True
        return False


class APIError(Exception):
    """Exception raised for API errors."""
    
    def __init__(self, message: str, request_id: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.request_id = request_id
        self.error_code = error_code
    
    def __str__(self):
        parts = [super().__str__()]
        if self.request_id:
            parts.append(f"[request_id: {self.request_id}]")
        if self.error_code:
            parts.append(f"[code: {self.error_code}]")
        return " ".join(parts)


@dataclass
class APIResponse:
    """Wrapper for API responses."""
    success: bool
    data: Any
    error: Optional[str] = None
    status_code: int = 200
    request_id: Optional[str] = None  # X-Request-ID from response

    def __bool__(self):
        return self.success


class DomainResponse:
    """Wrapper for domain API responses that preserves meta information.
    
    This class provides backward compatibility by behaving like the data itself,
    while also exposing the meta field with provider information.
    
    Examples:
        >>> result = qj.equities.get_pricing(symbol="AAPL")
        >>> # Works as before - result behaves like data
        >>> prices = result  # or result["close"] if it's a dict
        >>> # New: access meta information
        >>> provider = result.meta["provider"]  # "fmp"
        >>> method = result.meta["provider_method"]  # "get_historical_prices"
    """
    
    def __init__(self, data: Any, meta: Dict[str, Any]):
        self._data = data
        self.meta = meta
    
    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying data."""
        return getattr(self._data, name)
    
    def __getitem__(self, key: Any):
        """Delegate item access to the underlying data."""
        return self._data[key]
    
    def __setitem__(self, key: Any, value: Any):
        """Delegate item assignment to the underlying data."""
        self._data[key] = value
    
    def __iter__(self):
        """Delegate iteration to the underlying data."""
        return iter(self._data)
    
    def __len__(self):
        """Delegate length to the underlying data."""
        return len(self._data)
    
    def __repr__(self):
        """Return representation of the underlying data."""
        return repr(self._data)
    
    def __str__(self):
        """Return string representation of the underlying data."""
        return str(self._data)
    
    def __bool__(self):
        """Return truthiness of the underlying data."""
        return bool(self._data)
    
    def __eq__(self, other: Any):
        """Compare with underlying data."""
        if isinstance(other, DomainResponse):
            return self._data == other._data
        return self._data == other
    
    def __ne__(self, other: Any):
        """Compare with underlying data."""
        return not self.__eq__(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation with data and meta."""
        if isinstance(self._data, dict):
            return {**self._data, "meta": self.meta}
        return {"data": self._data, "meta": self.meta}


class ConnectorEndpoint:
    """Base class for connector endpoints."""

    def __init__(self, api_client: 'APIClient', connector_name: str):
        self.api_client = api_client
        self.connector_name = connector_name

    def _call(self, method: str, **params) -> Any:
        # Friendly param normalization for batch-friendly methods
        try:
            # For selected connectors, map symbol->symbols when server expects plural
            if self.connector_name in {"yf", "eod", "fmp"} and method in {
                "get_historical_prices", "get_intraday_prices", "get_real_time_prices", "get_forex_intraday_prices"
            }:
                if "symbols" not in params and "symbol" in params:
                    sym = params.pop("symbol")
                    params["symbols"] = sym
        except Exception:
            pass

        return self.api_client._request(
            endpoint=f"/{self.connector_name}/{method}",
            payload=params,
        )

    def __getattr__(self, name: str):
        # Dynamic fallback: allow calling any server-exposed method without a stub.
        if name.startswith("get_") or name.startswith("search_"):
            def _dyn_call(**params):
                return self._call(name, **params)
            return _dyn_call
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")


class APIClient:
    """Low-level HTTP client for API requests.
    
    Features:
    - X-Request-ID: Automatically generates unique request IDs for tracing
    - ETag caching: Caches GET responses and uses conditional requests
    - Auto token refresh: Refreshes expired JWT tokens automatically
    """

    def __init__(
        self, 
        base_url: str, 
        api_key: Optional[str] = None, 
        auth_url: Optional[str] = None,
        timeout: int = 30, 
        tenant_id: Optional[str] = None,
        enable_cache: bool = True,
        cache_max_size: int = 1000,
        pool_connections: int = POOL_CONNECTIONS,
        pool_maxsize: int = POOL_MAXSIZE,
    ):
        self.base_url = base_url.rstrip('/')
        self.auth_url = (auth_url or os.getenv("QJ_AUTH_URL") or "https://auth.quantjourney.cloud").rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._refresh_token: Optional[str] = None
        self._last_request_id: Optional[str] = None
        
        # ETag cache
        self._cache_enabled = enable_cache
        self._cache = ETagCache(max_size=cache_max_size) if enable_cache else None
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        if tenant_id:
            self.session.headers.update({"X-Tenant-Id": tenant_id})
    
    @property
    def last_request_id(self) -> Optional[str]:
        """Get the X-Request-ID from the last request."""
        return self._last_request_id
    
    @property
    def cache(self) -> Optional[ETagCache]:
        """Access the ETag cache."""
        return self._cache
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        if self._cache:
            self._cache.clear()
    
    def invalidate_cache(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Invalidate a specific cached response.
        
        Args:
            endpoint: The API endpoint (e.g., "/v1/market/historical")
            params: Optional query parameters
            
        Returns:
            True if cache entry was removed, False if not found
        """
        if not self._cache:
            return False
        cache_key = f"{endpoint}?{json.dumps(params, sort_keys=True)}" if params else endpoint
        return self._cache.remove(cache_key)

    def set_bearer_tokens(self, access_token: str, refresh_token: Optional[str] = None) -> None:
        """Set Authorization bearer tokens and update headers."""
        self.api_key = access_token
        self._refresh_token = refresh_token
        if access_token:
            self.session.headers.update({"Authorization": f"Bearer {access_token}"})
        else:
            self.session.headers.pop("Authorization", None)

    def set_tenant(self, tenant_id: Optional[str]) -> None:
        if tenant_id:
            self.session.headers.update({"X-Tenant-Id": tenant_id})
        else:
            self.session.headers.pop("X-Tenant-Id", None)

    def _maybe_log(self, direction: str, url: str, obj: Any, method: str = "POST") -> None:
        debug = str(os.getenv("QJ_SDK_DEBUG", "")).lower() in {"1", "true", "yes"}
        if not debug:
            return
        if direction == "->":
            print(f"[QJ SDK] {method.upper()} {url} payload={obj}")
        else:
            snippet = str(obj)
            if isinstance(snippet, str) and len(snippet) > 1000:
                snippet = snippet[:1000] + "..."
            print(f"[QJ SDK] <- {direction} {snippet}")

    def _handle_json(self, resp: requests.Response, request_id: Optional[str] = None) -> Any:
        data = resp.json()
        # If API uses { success, data } wrapper, unwrap; else return raw dict
        if isinstance(data, dict) and "success" in data:
            if data.get("success"):
                response_data = data.get("data")
                # If meta exists (from domain API), wrap response to preserve it
                if "meta" in data and isinstance(data["meta"], dict):
                    return DomainResponse(response_data, data["meta"])
                return response_data
            # Extract RFC 7807 error details if available
            error_code = data.get("error_code") or data.get("type")
            msg = data.get("error") or data.get("detail") or f"HTTP {resp.status_code}"
            error = APIError(f"API error: {msg}")
            error.request_id = request_id or data.get("request_id")
            error.error_code = error_code
            error.status_code = resp.status_code
            error.response_body = data
            raise error
        # Check for RFC 7807 Problem Details format
        if isinstance(data, dict) and "type" in data and "detail" in data:
            error = APIError(f"API error: {data.get('detail')}")
            error.request_id = request_id or data.get("request_id")
            error.error_code = data.get("error_code") or data.get("code") or data.get("type")
            error.status_code = resp.status_code
            error.response_body = data
            raise error
        return data

    def _refresh_and_retry(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Any:
        # Attempt refresh if we have a refresh token
        if not self._refresh_token:
            raise APIError("Unauthorized and no refresh token available")
        try:
            headers = {"X-Request-ID": request_id} if request_id else {}
            r = self.session.post(
                f"{self.auth_url}/auth/refresh",
                json={"refresh_token": self._refresh_token}, 
                headers=headers,
                timeout=self.timeout
            )
            r.raise_for_status()
            doc = r.json()
            access = doc.get("access_token")
            refresh = doc.get("refresh_token") or self._refresh_token
            if not access:
                raise APIError("Refresh did not return access_token")
            self.set_bearer_tokens(access, refresh)
        except Exception as e:
            raise APIError(f"Token refresh failed: {e}")
        # Retry once with same request_id
        return self._request_with_method(method, endpoint, payload=payload, params=params, _retry=False, _request_id=request_id)

    def _request_with_method(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        _retry: bool = True,
        _request_id: Optional[str] = None,
    ) -> Any:
        url = f"{self.base_url}{endpoint}"
        body = payload if method.upper() not in ("GET", "DELETE") else params
        
        # Generate or use provided X-Request-ID
        request_id = _request_id or self._generate_request_id()
        headers = {"X-Request-ID": request_id}
        
        # Check ETag cache for GET requests
        cache_key = None
        if self._cache_enabled and self._cache and method.upper() == "GET":
            cache_key = f"{endpoint}?{json.dumps(params, sort_keys=True)}" if params else endpoint
            cached = self._cache.get(cache_key)
            if cached:
                headers["If-None-Match"] = cached["etag"]
        
        try:
            self._maybe_log("->", url, body, method=method)
            method_upper = method.upper()
            if method_upper == "GET":
                response = self.session.get(url, params=params or {}, headers=headers, timeout=self.timeout)
            elif method_upper == "DELETE":
                response = self.session.delete(url, params=params or {}, headers=headers, timeout=self.timeout)
            elif method_upper == "PATCH":
                response = self.session.patch(url, json=payload or {}, headers=headers, timeout=self.timeout)
            elif method_upper == "PUT":
                response = self.session.put(url, json=payload or {}, headers=headers, timeout=self.timeout)
            else:
                response = self.session.post(url, json=payload or {}, headers=headers, timeout=self.timeout)
            
            # Store X-Request-ID from response
            self._last_request_id = response.headers.get("X-Request-ID", request_id)
            
            # Handle 304 Not Modified - return cached data
            if response.status_code == 304 and cache_key and self._cache:
                cached = self._cache.get(cache_key)
                if cached:
                    self._maybe_log(str(response.status_code), url, "(from cache)", method=method)
                    return cached["data"]
            
            # Don't try refresh for auth endpoints - they should fail with proper error
            is_auth_endpoint = endpoint.startswith("/auth/")
            if response.status_code == 401 and _retry and not is_auth_endpoint:
                return self._refresh_and_retry(method, endpoint, payload=payload, params=params, request_id=request_id)
            response.raise_for_status()
            data = self._handle_json(response, request_id=self._last_request_id)
            self._maybe_log(str(response.status_code), url, data, method=method)
            
            # Store in ETag cache for GET requests with ETag header
            etag = response.headers.get("ETag")
            if method_upper == "GET" and etag and cache_key and self._cache:
                self._cache.set(cache_key, etag, data)
            
            return data
        except requests.exceptions.HTTPError as e:
            # Extract JSON error body from the response if available
            error_body = None
            if e.response is not None:
                try:
                    error_body = e.response.json()
                except (ValueError, AttributeError):
                    pass
            
            if error_body:
                # Create APIError with structured error info
                error_msg = error_body.get('detail', error_body.get('error', str(e)))
                error = APIError(
                    message=f"HTTP {e.response.status_code}: {error_msg}",
                    request_id=error_body.get('request_id', self._last_request_id or request_id),
                    error_code=error_body.get('error_code') or error_body.get('code') or error_body.get('type')
                )
                error.status_code = e.response.status_code
                error.response_body = error_body  # Store full error body for inspection
            else:
                error = APIError(f"HTTP request failed: {str(e)}")
                error.request_id = self._last_request_id or request_id
            raise error
        except requests.exceptions.RequestException as e:
            error = APIError(f"HTTP request failed: {str(e)}")
            error.request_id = self._last_request_id or request_id
            raise error
        except ValueError as e:
            error = APIError(f"Invalid JSON response: {e}")
            error.request_id = self._last_request_id or request_id
            raise error
        except APIError:
            raise  # Re-raise APIError as-is (already has request_id)
        except Exception as e:
            error = APIError(f"Unexpected error: {str(e)}")
            error.request_id = self._last_request_id or request_id
            raise error

    def _request(self, endpoint: str, payload: Dict[str, Any], _retry: bool = True) -> Any:
        # Normalize symbols to list (API server iterates - string would be split into chars)
        if payload and "symbols" in payload:
            sym = payload["symbols"]
            if isinstance(sym, str):
                payload["symbols"] = [s.strip() for s in sym.split(",")] if "," in sym else [sym]
        
        return self._request_with_method("POST", endpoint, payload=payload, params=None, _retry=_retry)

    def _request_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, _retry: bool = True) -> Any:
        return self._request_with_method("GET", endpoint, payload=None, params=params, _retry=_retry)

    def health_check(self) -> Dict[str, Any]:
        try:
            return self._request_get("/health")
        except APIError as e:
            return {"status": "error", "error": str(e)}

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, _retry: bool = True) -> Any:
        return self._request_get(endpoint, params=params, _retry=_retry)

    def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None, _retry: bool = True) -> Any:
        return self._request_with_method("POST", endpoint, payload=json or {}, _retry=_retry)

    # ------------------------------------------------------------------ #
    # Connector health helpers
    def get_connectors_status(self) -> Dict[str, Any]:
        """Return aggregated connector health snapshot."""
        return self._request_get("/connectors/status")

    def get_connector_health(self, connector_name: str) -> Dict[str, Any]:
        """Return health payload for a specific connector."""
        return self._request_get(f"/connectors/{connector_name}/health")


class AsyncAPIClient:
    """Async HTTP client for API requests (httpx.AsyncClient).

    Notes:
    - Optional dependency: requires httpx
    - Intended for asyncio/FastAPI usage
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        auth_url: Optional[str] = None,
        tenant_id: Optional[str] = None,
        enable_cache: bool = True,
        cache_max_size: int = 1000,
        max_connections: int = 100,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        write_timeout: float = 10.0,
    ):
        if httpx is None:
            raise ImportError("httpx is required for AsyncAPIClient")

        self.base_url = base_url.rstrip('/')
        self.auth_url = (auth_url or os.getenv("QJ_AUTH_URL") or "https://auth.quantjourney.cloud").rstrip('/')
        self.api_key = api_key
        self._refresh_token: Optional[str] = None
        self._last_request_id: Optional[str] = None

        # ETag cache
        self._cache_enabled = enable_cache
        self._cache = ETagCache(max_size=cache_max_size) if enable_cache else None

        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if tenant_id:
            headers["X-Tenant-Id"] = tenant_id

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=read_timeout,
        )

        self.client = httpx.AsyncClient(
            headers=headers,
            limits=limits,
            timeout=timeout,
            follow_redirects=True,
        )

    @property
    def last_request_id(self) -> Optional[str]:
        return self._last_request_id

    @property
    def cache(self) -> Optional[ETagCache]:
        return self._cache

    def _generate_request_id(self) -> str:
        return str(uuid.uuid4())

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()

    def invalidate_cache(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> bool:
        if not self._cache:
            return False
        cache_key = f"{endpoint}?{json.dumps(params, sort_keys=True)}" if params else endpoint
        return self._cache.remove(cache_key)

    def set_bearer_tokens(self, access_token: str, refresh_token: Optional[str] = None) -> None:
        self.api_key = access_token
        self._refresh_token = refresh_token
        if access_token:
            self.client.headers["Authorization"] = f"Bearer {access_token}"
        else:
            self.client.headers.pop("Authorization", None)

    def set_tenant(self, tenant_id: Optional[str]) -> None:
        if tenant_id:
            self.client.headers["X-Tenant-Id"] = tenant_id
        else:
            self.client.headers.pop("X-Tenant-Id", None)

    def _maybe_log(self, direction: str, url: str, obj: Any, method: str = "POST") -> None:
        debug = str(os.getenv("QJ_SDK_DEBUG", "")).lower() in {"1", "true", "yes"}
        if not debug:
            return
        if direction == "->":
            print(f"[QJ SDK] {method.upper()} {url} payload={obj}")
        else:
            snippet = str(obj)
            if isinstance(snippet, str) and len(snippet) > 1000:
                snippet = snippet[:1000] + "..."
            print(f"[QJ SDK] <- {direction} {snippet}")

    def _handle_json(self, resp: Any, request_id: Optional[str] = None) -> Any:
        data = resp.json()
        if isinstance(data, dict) and "success" in data:
            if data.get("success"):
                response_data = data.get("data")
                if "meta" in data and isinstance(data["meta"], dict):
                    return DomainResponse(response_data, data["meta"])
                return response_data
            error_code = data.get("error_code") or data.get("type")
            msg = data.get("error") or data.get("detail") or f"HTTP {resp.status_code}"
            error = APIError(f"API error: {msg}")
            error.request_id = request_id or data.get("request_id")
            error.error_code = error_code
            error.status_code = resp.status_code
            error.response_body = data
            raise error
        if isinstance(data, dict) and "type" in data and "detail" in data:
            error = APIError(f"API error: {data.get('detail')}")
            error.request_id = request_id or data.get("request_id")
            error.error_code = data.get("error_code") or data.get("code") or data.get("type")
            error.status_code = resp.status_code
            error.response_body = data
            raise error
        return data

    async def _refresh_and_retry(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Any:
        if not self._refresh_token:
            raise APIError("Unauthorized and no refresh token available")
        try:
            headers = {"X-Request-ID": request_id} if request_id else {}
            r = await self.client.post(
                f"{self.auth_url}/auth/refresh",
                json={"refresh_token": self._refresh_token},
                headers=headers,
            )
            r.raise_for_status()
            doc = r.json()
            access = doc.get("access_token")
            refresh = doc.get("refresh_token") or self._refresh_token
            if not access:
                raise APIError("Refresh did not return access_token")
            self.set_bearer_tokens(access, refresh)
        except Exception as e:
            raise APIError(f"Token refresh failed: {e}")
        return await self._request_with_method(method, endpoint, payload=payload, params=params, _retry=False, _request_id=request_id)

    async def _request_with_method(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        _retry: bool = True,
        _request_id: Optional[str] = None,
    ) -> Any:
        url = f"{self.base_url}{endpoint}"
        body = payload if method.upper() not in ("GET", "DELETE") else params

        request_id = _request_id or self._generate_request_id()
        headers = {"X-Request-ID": request_id}

        cache_key = None
        if self._cache_enabled and self._cache and method.upper() == "GET":
            cache_key = f"{endpoint}?{json.dumps(params, sort_keys=True)}" if params else endpoint
            cached = self._cache.get(cache_key)
            if cached:
                headers["If-None-Match"] = cached["etag"]

        try:
            self._maybe_log("->", url, body, method=method)
            method_upper = method.upper()
            response = await self.client.request(
                method_upper,
                url,
                params=params if method_upper in {"GET", "DELETE"} else None,
                json=payload if method_upper not in {"GET", "DELETE"} else None,
                headers=headers,
            )

            self._last_request_id = response.headers.get("X-Request-ID", request_id)

            if response.status_code == 304 and cache_key and self._cache:
                cached = self._cache.get(cache_key)
                if cached:
                    self._maybe_log(str(response.status_code), url, "(from cache)", method=method)
                    return cached["data"]

            is_auth_endpoint = endpoint.startswith("/auth/")
            if response.status_code == 401 and _retry and not is_auth_endpoint:
                return await self._refresh_and_retry(method, endpoint, payload=payload, params=params, request_id=request_id)

            response.raise_for_status()
            data = self._handle_json(response, request_id=self._last_request_id)
            self._maybe_log(str(response.status_code), url, data, method=method)

            etag = response.headers.get("ETag")
            if method_upper == "GET" and etag and cache_key and self._cache:
                self._cache.set(cache_key, etag, data)

            return data
        except Exception as e:
            response = getattr(e, "response", None)
            error_body = None
            if response is not None:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = None
            if isinstance(error_body, dict):
                status_code = getattr(response, "status_code", None)
                error_msg = error_body.get("detail") or error_body.get("error") or str(e)
                error = APIError(
                    message=f"HTTP {status_code}: {error_msg}" if status_code else f"HTTP request failed: {error_msg}",
                    request_id=error_body.get("request_id", self._last_request_id or request_id),
                    error_code=error_body.get("error_code") or error_body.get("code") or error_body.get("type"),
                )
                error.status_code = status_code
                error.response_body = error_body
            else:
                error = APIError(f"HTTP request failed: {str(e)}")
                error.request_id = self._last_request_id or request_id
                if response is not None:
                    error.status_code = getattr(response, "status_code", None)
            raise error

    async def _request(self, endpoint: str, payload: Dict[str, Any], _retry: bool = True) -> Any:
        if payload and "symbols" in payload:
            sym = payload["symbols"]
            if isinstance(sym, str):
                payload["symbols"] = [s.strip() for s in sym.split(",")] if "," in sym else [sym]
        return await self._request_with_method("POST", endpoint, payload=payload, params=None, _retry=_retry)

    async def _request_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, _retry: bool = True) -> Any:
        return await self._request_with_method("GET", endpoint, payload=None, params=params, _retry=_retry)

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, _retry: bool = True) -> Any:
        return await self._request_get(endpoint, params=params, _retry=_retry)

    async def health_check(self) -> Dict[str, Any]:
        try:
            return await self._request_get("/health")
        except APIError as e:
            return {"status": "error", "error": str(e)}

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncAPIClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = ["APIClient", "AsyncAPIClient", "APIError", "APIResponse", "ConnectorEndpoint", "DomainResponse"]
