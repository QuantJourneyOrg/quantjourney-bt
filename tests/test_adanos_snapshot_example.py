# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from backtester import Backtester

SPEC = importlib.util.spec_from_file_location(
    "adanos_example",
    Path(__file__).parents[1] / "strategies/example_weights_26_adanos_snapshot_replay.py",
)
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Research replay must not access a network or SDK")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(Backtester, "_get_sdk_client", forbidden)
    for key in ("QJ_DATA_FILE", "QJ_DATA_CSV", "QJ_SAMPLE_DATA", "QJ_OUTPUT_DIR", "QJ_NO_REPORTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ADANOS_API_KEY", raising=False)


def archive(score=0.6):
    return {
        "source": "reddit_stocks",
        "ticker": "AAPL",
        "synthetic": True,
        "observations": [
            {
                "observed_at": "2026-10-20T06:00:00Z",
                "data_date": "2026-10-19",
                "found": True,
                "mentions": 10,
                "sentiment_score": score,
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


def prices():
    values = [100, 101, 99, 102, 103, 101, 104, 102, 105, 104, 103, 106]
    return pd.DataFrame(
        [
            {
                "date": day.strftime("%Y-%m-%d"),
                "symbol": "AAPL",
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "adj_close": close,
                "volume": 10000,
            }
            for day, close in zip(
                pd.bdate_range("2026-10-19", periods=len(values)), values, strict=True
            )
        ]
    )


def test_native_replay_records_availability_and_never_backdates_revision(tmp_path):
    snapshot = tmp_path / "observations.json"
    snapshot.write_text(json.dumps(archive()))
    strategy = example.AdanosSnapshotReplay(
        snapshot_file=snapshot,
        local_data=prices(),
        reports_directory=str(tmp_path / "reports"),
    )
    asyncio.run(strategy.run_strategy())
    signals = strategy.signals["AAPL"]
    assert signals.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert strategy._sdk_client is None
    nav = strategy.portfolio_data.net_asset_value
    assert len(nav) == 12 and nav.notna().all() and (nav > 0).all()
    assert nav.iloc[:3].nunique() == 1
    assert 99_980 < nav.iloc[3] < 100_000
    assert nav.iloc[4] > nav.iloc[3]
    metadata_paths = list((tmp_path / "reports").rglob("run_metadata.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text())["adanos_research"]
    assert metadata["snapshot_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert metadata["synthetic"] is True
    assert metadata["decisions"][1]["sentiment_score"] is None
    assert metadata["decisions"][2]["sentiment_score"] == 0.6
    assert metadata["decisions"][3]["sentiment_score"] == -0.6
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "command": "uv run --no-sync pytest -q tests/test_adanos_snapshot_example.py",
                "snapshot_sha256": metadata["snapshot_sha256"],
                "inputs": {"snapshot": archive(), "prices": prices().to_dict("records")},
                "signals": signals.tolist(),
                "nav": nav.tolist(),
                "result": "PASS",
            },
            indent=2,
        )
    )


def test_distributed_example_runs_without_checkout_only_fixtures(tmp_path, monkeypatch):
    copied = tmp_path / "standalone_example.py"
    shutil.copyfile(example.__file__, copied)
    spec = importlib.util.spec_from_file_location("standalone_adanos_example", copied)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("ADANOS_SNAPSHOT_FILE", raising=False)
    monkeypatch.setenv("QJ_OUTPUT_DIR", str(tmp_path / "reports"))
    asyncio.run(module.main())
    outputs = list((tmp_path / "reports").rglob("run_metadata.json"))
    assert len(outputs) == 1
    assert json.loads(outputs[0].read_text())["adanos_research"]["synthetic"] is True


def test_native_sample_launcher_runs_paired_synthetic_inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("ADANOS_SNAPSHOT_FILE", raising=False)
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('Sample launcher must stay offline')\n"
        "socket.socket.connect = forbidden\n"
        "socket.socket.connect_ex = forbidden\n"
    )
    reports = tmp_path / "reports"
    command = [
        sys.executable,
        "strategy.py",
        "example_weights_26_adanos_snapshot_replay",
        "--sample-data",
        "--output",
        str(reports),
    ]
    environment = {**os.environ, "PYTHONPATH": str(guard), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = list(reports.rglob("run_metadata.json"))
    assert len(outputs) == 1
    metadata = json.loads(outputs[0].read_text())["adanos_research"]
    assert metadata["synthetic"] is True
    assert metadata["decisions"][2]["sentiment_score"] == 0.6
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {"PYTHONPATH": str(guard)},
                "inputs": "Paired synthetic inputs generated by the standalone example",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "metadata": metadata,
                "result": "PASS",
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("score,expected", [(0, 0), (None, None)])
def test_observed_zero_is_not_missing(tmp_path, score, expected):
    payload = archive(score)
    payload["observations"] = payload["observations"][:1]
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(payload))
    strategy = example.AdanosSnapshotReplay(
        snapshot_file=snapshot, local_data=prices(), reports_directory=str(tmp_path / "reports")
    )
    asyncio.run(strategy.run_strategy())
    decisions = strategy._build_run_metadata()["adanos_research"]["decisions"]
    assert decisions[2]["sentiment_score"] == expected
    assert strategy.signals.loc["2026-10-21", "AAPL"] == 0


@pytest.mark.parametrize("defect", ["naive", "future_day", "duplicate", "bool", "range"])
def test_invalid_snapshot_is_rejected_before_execution(tmp_path, defect):
    payload = archive()
    row = payload["observations"][0]
    if defect == "naive":
        row["observed_at"] = "2026-10-20T06:00:00"
    elif defect == "future_day":
        row["data_date"] = "2026-10-20"
    elif defect == "duplicate":
        payload["observations"].append(dict(row))
    elif defect == "bool":
        row["sentiment_score"] = True
    else:
        row["sentiment_score"] = 2
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        example.AdanosSnapshotReplay(snapshot_file=snapshot, local_data=prices())


def test_explicit_capture_uses_one_closed_utc_day_and_keeps_key_private(monkeypatch):
    calls = []
    original_client = httpx.Client

    def respond(request):
        calls.append(request)
        assert request.headers["X-API-Key"] == "fixture-only-private-key"
        return httpx.Response(
            200,
            json={
                "ticker": "AAPL",
                "found": True,
                "mentions": 8,
                "sentiment_score": 0,
            },
        )

    monkeypatch.setenv("ADANOS_API_KEY", "fixture-only-private-key")
    monkeypatch.setattr(
        example.httpx,
        "Client",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    result = example.capture_snapshot(ticker="AAPL", day=date(2026, 10, 19))
    assert len(calls) == 1
    assert str(calls[0].url) == (
        "https://api.adanos.com/reddit/stocks/v1/stock/AAPL?from=2026-10-19&to=2026-10-19"
    )
    assert result["synthetic"] is False
    assert result["observations"][0]["sentiment_score"] == 0
    assert "fixture-only-private-key" not in json.dumps(result)


@pytest.mark.parametrize("found", [True, False])
@pytest.mark.parametrize("include_count", [True, False])
def test_capture_preserves_missing_activity_and_replays_cash(
    tmp_path, monkeypatch, found, include_count
):
    original_client = httpx.Client
    payload = {"ticker": "AAPL", "found": found, "sentiment_score": None}
    if include_count:
        payload["mentions"] = None
    monkeypatch.setenv("ADANOS_API_KEY", "fixture-only-private-key")
    monkeypatch.setattr(
        example.httpx,
        "Client",
        lambda **kwargs: original_client(
            **kwargs,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        ),
    )
    snapshot = example.capture_snapshot(ticker="AAPL", day=date(2026, 10, 19))
    assert snapshot["observations"][0]["mentions"] is None
    snapshot["synthetic"] = True
    snapshot["observations"][0]["observed_at"] = "2026-10-20T06:00:00Z"
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot))
    strategy = example.AdanosSnapshotReplay(
        snapshot_file=path, local_data=prices(), reports_directory=str(tmp_path / "reports")
    )
    asyncio.run(strategy.run_strategy())
    assert strategy.signals["AAPL"].tolist() == [0] * 12
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "command": "pytest -q tests/test_adanos_snapshot_example.py -k missing_activity",
                "inputs": {
                    "response": payload,
                    "snapshot": snapshot,
                    "prices": prices().to_dict("records"),
                },
                "signals": strategy.signals["AAPL"].tolist(),
                "result": "PASS",
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("status", [302, 401, 403, 404, 422, 429, 503])
def test_capture_failures_are_safe_and_not_retried(monkeypatch, status):
    calls = []
    original_client = httpx.Client

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status,
            text="fixture-key-do-not-expose",
            headers={
                "Location": "https://not-a-provider.invalid",
                "Retry-After": "1",
            },
        )

    monkeypatch.setenv("ADANOS_API_KEY", "fixture-key-do-not-expose")
    monkeypatch.setattr(
        example.httpx,
        "Client",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        example.capture_snapshot(ticker="AAPL", day=date(2026, 10, 19))
    assert "fixture-key-do-not-expose" not in str(error.value)
    assert len(calls) == 1
