# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Optional Reddit Stocks v1 archive adapter, adapted from public PR #6."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from backtester.observations import ObservationData

_LIMIT = 1_048_576
_COLUMNS = ["observed_at", "data_date", "found", "mentions", "sentiment_score"]


def _parse(payload: Any) -> tuple[str, bool, pd.DataFrame]:
    if not isinstance(payload, dict) or payload.get("source") != "reddit_stocks":
        raise ValueError("Expected a Reddit Stocks observation archive")
    ticker = payload.get("ticker")
    if not isinstance(ticker, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,19}", ticker):
        raise ValueError("Expected one stock ticker")
    if type(payload.get("synthetic")) is not bool or not isinstance(
        payload.get("observations"), list
    ):
        raise ValueError("Archive requires a synthetic flag and observation list")
    rows = []
    for row in payload["observations"]:
        try:
            received = datetime.fromisoformat(row["observed_at"])
            day = date.fromisoformat(row["data_date"])
            if received.utcoffset() is None or day.isoformat() != row["data_date"]:
                raise ValueError("Use a time-zone-aware receipt and an ISO data date")
            if day >= received.astimezone(UTC).date():
                raise ValueError("Adanos observations must describe a completed UTC day")
            score, mentions, found = row["sentiment_score"], row["mentions"], row["found"]
            if type(found) is not bool or (
                mentions is not None and (type(mentions) is not int or mentions < 0)
            ):
                raise ValueError("Invalid observation activity")
            if score is not None and (
                type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1
            ):
                raise ValueError("Sentiment must be null or finite and between -1 and 1")
        except (KeyError, TypeError, AttributeError):
            raise ValueError("Malformed Adanos observation") from None
        rows.append({column: row[column] for column in _COLUMNS})
    return ticker.upper(), payload["synthetic"], pd.DataFrame(rows, columns=_COLUMNS)


def load_adanos_observations(path: str | os.PathLike[str]) -> ObservationData:
    """Read a bounded local archive; never contact a provider during replay."""
    file = Path(path)
    with file.open("rb") as stream:
        raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT:
        raise ValueError("Use an Adanos archive at most 1 MiB")
    ticker, synthetic, frame = _parse(json.loads(raw))
    return ObservationData(
        frame,
        instrument=ticker,
        source="reddit_stocks",
        synthetic=synthetic,
        provenance_columns=("found", "mentions", "sentiment_score"),
        archive_bytes=raw,
        file_name=file.name,
    )


def capture_adanos_snapshot(*, ticker: str, day: date) -> dict[str, Any]:
    """Explicitly capture one completed day; record receipt after receiving the body."""
    if not isinstance(ticker, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,19}", ticker):
        raise ValueError("Expected one stock ticker")
    ticker = ticker.upper()
    if type(day) is not date or day >= datetime.now(UTC).date():
        raise ValueError("Choose a completed UTC day")
    key = os.getenv("ADANOS_API_KEY", "")
    if not key or any(not 33 <= ord(c) <= 126 for c in key):
        raise ValueError("Configure a valid server-side ADANOS_API_KEY")
    try:
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as client:
            with client.stream(
                "GET",
                f"https://api.adanos.com/reddit/stocks/v1/stock/{ticker}",
                params={"from": day.isoformat(), "to": day.isoformat()},
                headers={"X-API-Key": key},
            ) as response:
                if not response.is_success:
                    raise RuntimeError(f"Adanos request failed (HTTP {response.status_code})")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > _LIMIT:
                        raise RuntimeError("Adanos response exceeds 1 MiB")
                received = datetime.now(UTC).isoformat()
    except httpx.HTTPError:
        raise RuntimeError("Adanos transport error or timeout") from None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("ticker") != ticker:
            raise ValueError("Unexpected ticker")
        archive = {
            "source": "reddit_stocks",
            "ticker": ticker,
            "synthetic": False,
            "observations": [
                {
                    "observed_at": received,
                    "data_date": day.isoformat(),
                    "found": data["found"],
                    "mentions": data.get("mentions"),
                    "sentiment_score": data.get("sentiment_score"),
                }
            ],
        }
        _parse(archive)
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("Adanos returned an invalid observation") from None
    return archive
