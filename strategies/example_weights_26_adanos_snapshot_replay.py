# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Example Weights 26 - Adanos own-observation replay.

Mode: weights, long/cash. Universe: one US equity, AAPL in the synthetic demo.
Idea: illustrate source-labelled sentiment observed before a decision, without
projecting a history downloaded today backwards. Positive observations choose a
long target; missing, stale, zero and negative observations choose cash. This is
an illustrative rule, not a calibrated model or evidence of predictive value.
Native bar-shift and close-rebalance semantics remain unchanged.

Usage: ./strategy.sh example_weights_26_adanos_snapshot_replay
Both bundled inputs are synthetic. Own snapshots do not establish provider-wide
historical point-in-time coverage or point-in-time correctness of price data.
"""

import asyncio
import hashlib
import json
import math
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy


def _parse(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict) or payload.get("source") != "reddit_stocks":
        raise ValueError("Expected a Reddit Stocks observation archive")
    if not isinstance(payload.get("ticker"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.-]{0,19}", payload["ticker"]
    ):
        raise ValueError("Expected one stock ticker")
    if type(payload.get("synthetic")) is not bool or not isinstance(
        payload.get("observations"), list
    ):
        raise ValueError("Archive needs a synthetic flag and observation list")
    rows = []
    for row in payload["observations"]:
        try:
            observed_at = datetime.fromisoformat(row["observed_at"])
            data_date = date.fromisoformat(row["data_date"])
            if observed_at.utcoffset() is None:
                raise ValueError("Observation timestamps must include a timezone")
            observed_at = observed_at.astimezone(UTC)
            if data_date.isoformat() != row["data_date"] or data_date >= observed_at.date():
                raise ValueError("Data must refer to a completed UTC day before observation")
            score, mentions, found = row["sentiment_score"], row["mentions"], row["found"]
            if type(found) is not bool or (
                mentions is not None and (type(mentions) is not int or mentions < 0)
            ):
                raise ValueError("Invalid observation activity")
            if score is not None and (
                type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1
            ):
                raise ValueError("Sentiment must be null or finite and between -1 and 1")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Malformed observation") from error
        rows.append(
            {
                "observed_at": observed_at,
                "data_date": data_date,
                "sentiment_score": score,
                "mentions": mentions,
                "found": found,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "observed_at",
            "data_date",
            "sentiment_score",
            "mentions",
            "found",
        ],
    ).set_index("observed_at")
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates:
        raise ValueError("Observation timestamps must be unique")
    return frame.sort_index()


def capture_snapshot(*, ticker: str, day: date) -> dict[str, Any]:
    """Explicitly acquire one closed UTC day outside any replay; never persist a key."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,19}", ticker):
        raise ValueError("Expected one stock ticker")
    if type(day) is not date or day >= datetime.now(UTC).date():
        raise ValueError("Choose a completed UTC day")
    api_key = os.getenv("ADANOS_API_KEY", "")
    if not api_key or any(not 33 <= ord(character) <= 126 for character in api_key):
        raise ValueError("Configure a valid server-side ADANOS_API_KEY")
    try:
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as client:
            response = client.get(
                f"https://api.adanos.com/reddit/stocks/v1/stock/{ticker}",
                params={"from": day.isoformat(), "to": day.isoformat()},
                headers={"X-API-Key": api_key},
            )
    except httpx.HTTPError:
        raise RuntimeError("Adanos transport error or timeout") from None
    if not response.is_success:
        raise RuntimeError(f"Adanos request failed (HTTP {response.status_code})")
    try:
        data = response.json()
        if not isinstance(data, dict) or data.get("ticker") != ticker:
            raise ValueError("Unexpected ticker")
        snapshot = {
            "source": "reddit_stocks",
            "ticker": ticker,
            "synthetic": False,
            "observations": [
                {
                    "observed_at": datetime.now(UTC).isoformat(),
                    "data_date": day.isoformat(),
                    "found": data["found"],
                    "mentions": data.get("mentions"),
                    "sentiment_score": data.get("sentiment_score"),
                }
            ],
        }
        _parse(snapshot)
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("Adanos returned an invalid observation") from None
    return snapshot


class AdanosSnapshotReplay(Backtester):
    supports_csv_config = True

    def __init__(
        self,
        *,
        snapshot_file: Path,
        local_data: str | os.PathLike[str] | pd.DataFrame,
        reports_directory: str = "./reports",
    ) -> None:
        raw = Path(snapshot_file).read_bytes()
        if len(raw) > 1_048_576:
            raise ValueError("Use a small research observation archive, at most 1 MiB")
        self._archive = json.loads(raw)
        self._observations = _parse(self._archive)
        self._snapshot_hash = hashlib.sha256(raw).hexdigest()
        self._decisions: list[dict[str, Any]] = []
        super().__init__(
            source="local",
            local_data=local_data,
            csv_config=True,
            # Native CSV configuration replaces this required constructor seed.
            backtest_period={"start": "2026-10-19", "end": "2026-11-03"},
            instruments=[self._archive["ticker"]],
            strategy_name="ExampleWeights26_AdanosOwnObservationResearch",
            strategy_type="Synthetic research"
            if self._archive["synthetic"]
            else "Own-observation research",
            benchmark_symbol=self._archive["ticker"],
            benchmark_name="Local price reference",
            indicators_config=[],
            rebalance_policy=RebalancePolicy(frequency="D"),
            strict_reporting=True,
            show_text_reports=False,
            save_text_reports=True,
            show_portfolio_plots=False,
            save_portfolio_plots=False,
            reports_directory=reports_directory,
        )

    def _compute_signals(self) -> pd.DataFrame:
        close = self.data.get_feature("adj_close")
        if list(close.columns) != [self._archive["ticker"]]:
            raise ValueError("Local prices must contain exactly the archive's stock")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        self._decisions = []
        for day in close.index:
            cutoff = pd.Timestamp(day)
            cutoff = (
                cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            )
            eligible = self._observations.loc[self._observations.index < cutoff]
            score, observed_at, data_date = None, None, None
            if not eligible.empty:
                row = eligible.iloc[-1]
                observed_at = eligible.index[-1].isoformat()
                data_date = row["data_date"].isoformat()
                if (
                    (cutoff.date() - row["data_date"]).days <= 3
                    and row["found"]
                    and pd.notna(row["mentions"])
                    and row["mentions"] > 0
                    and pd.notna(row["sentiment_score"])
                ):
                    score = float(row["sentiment_score"])
                    signals.loc[day, close.columns[0]] = float(score > 0)
            self._decisions.append(
                {
                    "date": cutoff.date().isoformat(),
                    "observed_at": observed_at,
                    "data_date": data_date,
                    "sentiment_score": score,
                }
            )
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        return self.signals.copy()

    def _build_run_metadata(self) -> dict[str, Any]:
        metadata = super()._build_run_metadata()
        metadata["adanos_research"] = {
            "source": "reddit_stocks",
            "snapshot_sha256": self._snapshot_hash,
            "synthetic": self._archive["synthetic"],
            "limitation": "Own observations only, not provider-wide historical point-in-time data",
            "decisions": self._decisions,
        }
        return metadata


def _synthetic_inputs(snapshot_file: Path) -> pd.DataFrame:
    snapshot_file.write_text(
        json.dumps(
            {
                "source": "reddit_stocks",
                "ticker": "AAPL",
                "synthetic": True,
                "observations": [
                    {
                        "observed_at": "2026-10-20T06:00:00Z",
                        "data_date": "2026-10-19",
                        "found": True,
                        "mentions": 10,
                        "sentiment_score": 0.6,
                    },
                    {
                        "observed_at": "2026-10-21T18:00:00Z",
                        "data_date": "2026-10-19",
                        "found": True,
                        "mentions": 12,
                        "sentiment_score": -0.6,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    values = [100, 101, 99, 102, 103, 101, 104, 102, 105, 104, 103, 106]
    return pd.DataFrame(
        [
            {
                "date": day.isoformat(),
                "symbol": "AAPL",
                "open": value,
                "high": value + 1,
                "low": value - 1,
                "close": value,
                "adj_close": value,
                "volume": 10000,
            }
            for day, value in zip(pd.bdate_range("2026-10-19", periods=12), values, strict=True)
        ]
    )


async def main() -> None:
    price_file, legacy_file = os.getenv("QJ_DATA_FILE"), os.getenv("QJ_DATA_CSV")
    snapshot_file = os.getenv("ADANOS_SNAPSHOT_FILE")
    if price_file and legacy_file:
        raise ValueError("Choose only one local price file")
    price_file = price_file or legacy_file
    if bool(price_file) != bool(snapshot_file):
        raise ValueError(
            "Supply both local prices and ADANOS_SNAPSHOT_FILE, or use both demo inputs"
        )
    with TemporaryDirectory(prefix="adanos-synthetic-demo-") as directory:
        snapshot_path = Path(snapshot_file) if snapshot_file else Path(directory) / "snapshot.json"
        local_data = price_file if price_file else _synthetic_inputs(snapshot_path)
        sample_setting = os.environ.pop("QJ_SAMPLE_DATA", None) if not price_file else None
        try:
            strategy = AdanosSnapshotReplay(snapshot_file=snapshot_path, local_data=local_data)
        finally:
            if sample_setting is not None:
                os.environ["QJ_SAMPLE_DATA"] = sample_setting
        await strategy.run_strategy()
        strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
