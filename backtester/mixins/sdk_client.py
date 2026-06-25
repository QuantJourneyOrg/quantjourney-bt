"""
SDKClientMixin — API authentication, data fetching, server-side calcs
=====================================================================

Extracted from core.py to keep the Backtester class focused on the
strategy pipeline (data → signals → weights → positions → performance).

All methods expect the host class to have the attributes set by
Backtester.__init__ (api_url, _email, _password, _api_key, _sdk_client,
_source, _granularity, backtest_period, instruments, etc.).

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

import logging
import os
from typing import Dict, Any, List, Optional

try:
    from backtester.utils.logger import logger
except Exception:
    logger = logging.getLogger("backtester")


class SDKClientMixin:
    """API authentication, market-data fetching, and server-side calculations."""

    @staticmethod
    def _replace_existing_session_enabled() -> bool:
        value = os.getenv("QJ_REPLACE_EXISTING_SESSION", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    @staticmethod
    def _response_detail(resp: Any) -> Any:
        try:
            body = resp.json()
        except Exception:
            return getattr(resp, "text", "")
        if isinstance(body, dict):
            return body.get("detail") or body
        return body

    @classmethod
    def _is_active_session_conflict(cls, resp: Any) -> bool:
        if getattr(resp, "status_code", None) != 409:
            return False
        detail = cls._response_detail(resp)
        return isinstance(detail, dict) and detail.get("code") == "active_session_exists"

    # ─────────────────────────────────────────────────────────────────
    # SDK Client — uses quantjourney.sdk.client.AsyncAPIClient
    # ─────────────────────────────────────────────────────────────────

    async def _get_sdk_client(self):
        """Lazy-initialize and return the SDK async client."""
        if self._sdk_client is not None:
            return self._sdk_client

        from backtester.sdk.client import AsyncAPIClient

        if self._api_key:
            # API key auth — handled by SDK (Bearer header)
            self._sdk_client = AsyncAPIClient(
                base_url=self.api_url,
                api_key=self._api_key,
                auth_url=os.getenv("QJ_AUTH_URL") or "https://auth.quantjourney.cloud",
                read_timeout=120.0,
            )
            logger.info("[Backtester] Using API key auth via SDK")
        elif self._email and self._password:
            auth_url = (os.getenv("QJ_AUTH_URL") or "https://auth.quantjourney.cloud").rstrip("/")
            # Email/password — login and set tokens
            self._sdk_client = AsyncAPIClient(
                base_url=self.api_url,
                auth_url=auth_url,
                read_timeout=120.0,
            )
            # Login to get JWT
            login_payload = {
                "email": self._email,
                "password": self._password,
                "service": os.getenv("QJ_AUTH_SERVICE", "backtester"),
            }
            resp = await self._sdk_client.client.post(
                f"{auth_url}/auth/login",
                json=login_payload,
            )
            if self._is_active_session_conflict(resp):
                if not self._replace_existing_session_enabled():
                    raise ValueError(
                        f"Authentication blocked by an active QuantJourney session at {auth_url}/auth/login\n"
                        f"  Email: {self._email}\n"
                        f"  Set QJ_REPLACE_EXISTING_SESSION=1 to let the CLI replace the existing session,\n"
                        f"  or use QJ_API_KEY to avoid browser-session conflicts."
                    )
                logger.info("[Backtester] Active auth session exists; replacing it for this headless backtester run")
                resp = await self._sdk_client.client.post(
                    f"{auth_url}/auth/login",
                    json={**login_payload, "replace_existing_session": True},
                )
            if resp.status_code == 401:
                raise ValueError(
                    f"Authentication failed (401 Unauthorized) at {auth_url}/auth/login\n"
                    f"  Email: {self._email}\n"
                    f"  Please check your QJ_EMAIL / QJ_PASSWORD environment variables,\n"
                    f"  or set QJ_API_KEY for API key authentication.\n"
                    f"  Hint: export QJ_API_KEY='your-key-here' or add it to .env"
                )
            if resp.status_code == 409:
                raise ValueError(
                    f"Authentication failed (409 Conflict) at {auth_url}/auth/login\n"
                    f"  Email: {self._email}\n"
                    f"  Detail: {self._response_detail(resp)}\n"
                    f"  Try QJ_REPLACE_EXISTING_SESSION=1 or use QJ_API_KEY."
                )
            if resp.status_code == 403:
                raise ValueError(
                    f"Authentication failed (403 Forbidden) at {auth_url}/auth/login\n"
                    f"  Email: {self._email}\n"
                    f"  Detail: {self._response_detail(resp)}"
                )
            resp.raise_for_status()
            data = resp.json()
            access = data["access_token"]
            refresh = data.get("refresh_token")
            self._sdk_client.set_bearer_tokens(access, refresh)
            expires = data.get("expires_in", "?")
            logger.info(f"[Backtester] Logged in as {self._email} (expires in {expires}s)")
        else:
            raise ValueError(
                "Backtester requires either (email + password) or api_key"
            )

        return self._sdk_client

    # ─────────────────────────────────────────────────────────────────
    # Data Fetching — POST /bt/prepare (via SDK)
    # ─────────────────────────────────────────────────────────────────

    async def _fetch_market_data(self) -> None:
        """
        Fetch market data from /bt/prepare API via the SDK client.
        Auto token refresh on 401 is handled by the SDK.
        """
        client = await self._get_sdk_client()

        payload = {
            "provider": {
                "source": self._source,
                "granularity": self._granularity,
            },
            "backtest_period": {
                "start": self.backtest_period.start,
                "end": self.backtest_period.end,
            },
            "instruments": self.instruments,
            "persist": self._persist,
            "dedupe": self._dedupe,
            "force_refresh": self._force_refresh,
        }

        logger.info(
            f"[Backtester] POST /bt/prepare: {len(self.instruments)} instruments, "
            f"{self.backtest_period.start}..{self.backtest_period.end}, source={self._source}"
        )

        self._api_response = await client._request("/bt/prepare", payload)

        self.session_id = self._api_response.get("session_id")
        self.dataset_id = self._api_response.get("dataset_id")
        summary = self._api_response.get("summary", {})

        logger.info(
            f"[Backtester] Data received: "
            f"session={self.session_id}, dataset={self.dataset_id}, "
            f"instruments={summary.get('instruments')}, dates={summary.get('dates')}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Server-Side Calculations (optional convenience)
    # ─────────────────────────────────────────────────────────────────

    async def calc_portfolio_server(
        self,
        calc_ids: List[str],
        params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run calculations on the server via POST /bt/calc/portfolio.
        Returns the raw results dict.
        """
        if not self.session_id:
            raise ValueError("No session_id — run prepare first")

        client = await self._get_sdk_client()

        result = await client._request("/bt/calc/portfolio", {
            "session_id": self.session_id,
            "calc_ids": calc_ids,
            "params": params or {},
        })
        return result.get("results", {}) if isinstance(result, dict) else result
