# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Availability and provenance boundaries for the optional observation extension."""

import asyncio
import hashlib
import importlib.util
import json
import socket
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from backtester.observations import ObservationData
from backtester.observations.adanos import capture_adanos_snapshot, load_adanos_observations

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/observations"
PERIOD = {"start": "2024-11-04", "end": "2026-11-03"}

_spec = importlib.util.spec_from_file_location(
    "observation_example", ROOT / "strategies/example_weights_26_adanos_observations.py"
)
assert _spec is not None and _spec.loader is not None
_strategy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_strategy)
AdanosSentiment = _strategy.AdanosSentiment


def frame():
    return pd.DataFrame(
        [
            {"observed_at": "2026-10-20T06:00:00Z", "data_date": "2026-10-19", "value": 0.0},
            {"observed_at": "2026-10-21T18:00:00Z", "data_date": "2026-10-19", "value": None},
        ]
    )


def data(table=None, **kwargs):
    return ObservationData(
        frame() if table is None else table,
        instrument="AAPL",
        source="test",
        synthetic=True,
        **kwargs,
    )


def test_alignment_preserves_zero_missing_revision_and_index():
    index = pd.date_range("2026-10-20", periods=7, tz="UTC", name="research_date")
    aligned = data().align(index, max_data_age_days=3)
    pd.testing.assert_index_equal(aligned.index, index)
    assert not aligned.iloc[0].usable
    assert aligned.iloc[1].usable and aligned.iloc[1]["value"] == 0
    assert aligned.iloc[2].usable and pd.isna(aligned.iloc[2]["value"])
    assert not aligned.iloc[3].usable
    assert aligned.iloc[3].observed_at == pd.Timestamp("2026-10-21T18:00:00Z")
    exact = data().align(pd.DatetimeIndex(["2026-10-20T06:00:00Z"]))
    assert not exact.iloc[0].usable
    shifted = data().align(pd.DatetimeIndex(["2026-10-21T02:00:00+02:00"], name="local"))
    assert shifted.iloc[0].usable
    assert str(shifted.index.tz) == "UTC+02:00"


@pytest.mark.parametrize("timestamp", ["2026-10-20T06:00:00", 1760000000, None, pd.NaT])
def test_invalid_availability_fails_closed(timestamp):
    table = frame()
    table.loc[0, "observed_at"] = timestamp
    with pytest.raises(ValueError):
        data(table)


def test_duplicates_decision_order_and_foreign_instruments_fail_closed():
    table = frame()
    table.loc[1, "observed_at"] = "2026-10-20T08:00:00+02:00"
    with pytest.raises(ValueError, match="unique"):
        data(table)
    for dates in (
        ["2026-10-21", "2026-10-20"],
        ["2026-10-21", "2026-10-21"],
        ["2026-10-21T06:00:00"],
    ):
        with pytest.raises(ValueError):
            data().align(pd.DatetimeIndex(dates))
    table = frame().assign(Ticker=["AAPL", "MSFT"])
    with pytest.raises(ValueError, match="identifier"):
        data(table)
    for limit in [-1, True, 3.0]:
        with pytest.raises(ValueError):
            data().align(pd.date_range("2026-10-20", periods=2), max_data_age_days=limit)


def test_empty_optional_date_and_late_backfill_contract():
    dates = pd.date_range("2026-10-21", periods=3)
    assert not data(frame().iloc[:0]).align(dates, max_data_age_days=3).usable.any()
    no_date = data(frame().drop(columns="data_date"))
    assert no_date.align(dates).usable.all()
    with pytest.raises(ValueError, match="requires data_date"):
        no_date.align(dates, max_data_age_days=3)
    late = pd.DataFrame(
        [
            {"observed_at": "2026-10-21T06:00:00Z", "data_date": "2026-10-20", "value": 1.0},
            {"observed_at": "2026-10-21T18:00:00Z", "data_date": "2026-10-01", "value": -1.0},
        ]
    )
    selected = data(late).align(pd.DatetimeIndex(["2026-10-22"]), max_data_age_days=3)
    assert not selected.iloc[0].usable and selected.iloc[0]["value"] == -1


def test_immutable_snapshot_hash_and_minimal_provenance():
    table = frame().assign(private_note="test-only text")
    original = data(table)
    digest = original.metadata["sha256"]
    table.loc[0, "value"] = 12.0
    selected = original.align(pd.date_range("2026-10-21", periods=1))
    selected.loc[:, "value"] = 99.0
    assert original.align(selected.index).iloc[0]["value"] == 0
    assert original.metadata["sha256"] == digest
    assert "private_note" not in json.dumps(original.provenance(selected))
    changed = ObservationData(frame(), instrument="AAPL", source="other", synthetic=True)
    assert changed.metadata["sha256"] != data().metadata["sha256"]
    numeric = data(
        frame().assign(flag=[np.bool_(True), np.bool_(False)]), provenance_columns=("flag", "value")
    )
    records = numeric.provenance(numeric.align(pd.date_range("2026-10-21", periods=2)))
    assert records[0]["flag"] is True and records[0]["value"] == 0
    assert records[1]["value"] is None


@pytest.fixture
def offline(monkeypatch):
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("Observation replay attempted network access")

    async def forbidden_sdk(*args, **kwargs):
        attempts.append("SDK")
        raise AssertionError("Observation replay attempted SDK access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(Backtester, "_get_sdk_client", forbidden_sdk)
    for key in (
        "QJ_SAMPLE_DATA",
        "QJ_CSV_CONFIG",
        "QJ_STRICT_DATA_FETCH",
        "QJ_DATA_FILE",
        "QJ_DATA_CSV",
        "QJ_OUTPUT_DIR",
        "QJ_NO_REPORTS",
        "QJ_SKIP_ANALYSIS",
        "ADANOS_SNAPSHOT_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    return attempts


def strategy(tmp_path, **kwargs):
    return AdanosSentiment(
        observation_data=load_adanos_observations(FIXTURES / "adanos_two_years.json"),
        max_data_age_days=3,
        instruments=["AAPL"],
        backtest_period=PERIOD,
        source="local",
        local_data=FIXTURES / "prices_two_years.csv",
        reports_directory=str(tmp_path),
        show_text_reports=False,
        save_portfolio_plots=False,
        save_text_reports=True,
        **kwargs,
    )


def test_two_year_native_parity_and_run_metadata(tmp_path, offline):
    spec = importlib.util.spec_from_file_location(
        "adanos_pr6_reference", ROOT / "tests/reference/adanos_pr6.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    old = reference.AdanosSnapshotReplay(
        snapshot_file=FIXTURES / "adanos_two_years.json",
        local_data=FIXTURES / "prices_two_years.csv",
        reports_directory=str(tmp_path / "original"),
    )
    new = strategy(tmp_path / "shared")
    asyncio.run(old.run_strategy())
    asyncio.run(new.run_strategy())
    pd.testing.assert_frame_equal(old.signals, new.signals)
    pd.testing.assert_series_equal(
        old.portfolio_data.net_asset_value, new.portfolio_data.net_asset_value
    )
    assert len(new.signals) == 522 and not offline
    files = list((tmp_path / "shared").rglob("run_metadata.json"))
    metadata = json.loads(files[0].read_text())["external_observations"]
    assert metadata["synthetic"] is True
    assert metadata["sample_price_mode"] is False
    assert "synthetic_prices" not in metadata
    assert (
        metadata["archive"]["sha256"]
        == hashlib.sha256((FIXTURES / "adanos_two_years.json").read_bytes()).hexdigest()
    )
    assert len(metadata["calls"]) == 1 and len(metadata["calls"][0]) == 522
    assert metadata["calls"][0][0]["observed_at"] is None
    assert metadata["calls"][0][0]["usable"] is False
    assert "Synthetic" in new._build_strategy_params()["Strategy Type"]
    for name in ("performance_report.txt", "summary.txt"):
        report = (files[0].parent / name).read_text()
        assert report.startswith("SYNTHETIC RESEARCH")
        assert "Synthetic observations: True; sample price mode: False" in report
    exported = new._build_run_metadata()
    exported["external_observations"]["calls"][0][0]["usable"] = True
    assert new._build_run_metadata()["external_observations"]["calls"][0][0]["usable"] is False
    asyncio.run(new.run_strategy())
    assert len(new._observation_calls) == 1


def test_empty_sample_and_other_feature_use_same_shared_module(tmp_path, offline):
    empty = ObservationData(
        pd.DataFrame(columns=["observed_at", "data_date", "found", "mentions", "sentiment_score"]),
        instrument="AAPL",
        source="reddit_stocks",
        synthetic=True,
    )
    sample = AdanosSentiment(
        observation_data=empty,
        max_data_age_days=3,
        instruments=["AAPL"],
        backtest_period={"start": "2026-10-01", "end": "2026-10-30"},
        source="sample",
        reports_directory=str(tmp_path / "empty"),
        skip_analysis=True,
    )
    asyncio.run(sample.run_strategy())
    assert not sample.signals.any().any() and not offline
    earnings = ObservationData(
        pd.DataFrame(
            [
                {
                    "observed_at": "2026-10-20T07:00:00Z",
                    "data_date": "2026-10-20",
                    "eps_surprise": Decimal("0.12"),
                }
            ]
        ),
        instrument="AAPL",
        source="test_earnings",
        synthetic=True,
    )
    aligned = earnings.align(pd.DatetimeIndex(["2026-10-21"]), max_data_age_days=3)
    assert aligned.iloc[0].usable and aligned.iloc[0].eps_surprise == Decimal("0.12")


def test_metadata_failure_is_fatal_even_with_relaxed_environment(tmp_path, monkeypatch, offline):
    monkeypatch.setenv("QJ_STRICT_REPORTING", "0")
    original_open = Path.open

    def fail_metadata(path, *args, **kwargs):
        if path.name == "run_metadata.json":
            raise OSError("Test-only metadata failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_metadata)
    with pytest.raises(OSError, match="metadata failure"):
        asyncio.run(strategy(tmp_path, skip_analysis=True).run_strategy())
    with pytest.raises(ValueError, match="weight mode"):
        strategy(tmp_path, execution_mode="orders")
    with pytest.raises(ValueError, match="strict reporting"):
        strategy(tmp_path, strict_reporting=False)


@pytest.mark.parametrize("status", [302, 401, 429, 503])
def test_capture_errors_are_separate_from_observations(monkeypatch, status):
    from datetime import date

    client = httpx.Client
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            status, text="test response", headers={"Location": "https://invalid.example"}
        )

    monkeypatch.setenv("ADANOS_API_KEY", "fixture")
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(respond))
    )
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        capture_adanos_snapshot(ticker="AAPL", day=date(2026, 10, 19))
    assert len(seen) == 1


def test_capture_nullable_activity_and_bounded_archives(tmp_path, monkeypatch):
    from datetime import date

    client = httpx.Client

    def respond(request):
        assert request.url.params == httpx.QueryParams({"from": "2026-10-19", "to": "2026-10-19"})
        return httpx.Response(200, json={"ticker": "AAPL", "found": True, "sentiment_score": None})

    monkeypatch.setenv("ADANOS_API_KEY", "fixture")
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(respond))
    )
    result = capture_adanos_snapshot(ticker="AAPL", day=date(2026, 10, 19))
    assert result["observations"][0]["mentions"] is None
    assert result["synthetic"] is False and "fixture" not in json.dumps(result)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(result))
    loaded = load_adanos_observations(path)
    assert loaded.metadata["archive"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b" " * (1_048_576 + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        load_adanos_observations(path)


def test_default_main_is_an_honest_empty_sample(tmp_path, monkeypatch, offline):
    monkeypatch.delenv("ADANOS_SNAPSHOT_FILE", raising=False)
    monkeypatch.setenv("QJ_OUTPUT_DIR", str(tmp_path))
    asyncio.run(_strategy.main())
    path = next(tmp_path.rglob("run_metadata.json"))
    metadata = json.loads(path.read_text())
    observation = metadata["external_observations"]
    assert observation["synthetic"] is True
    assert not any(row["usable"] for row in observation["calls"][0])
    assert "Synthetic" in metadata["strategy_type"] and not offline


def test_replaced_csv_universe_is_checked_after_processing(tmp_path, offline):
    prices = pd.read_csv(FIXTURES / "prices_two_years.csv").head(4)
    wider = pd.concat([prices, prices.assign(symbol="MSFT")], ignore_index=True)

    class FileSelection(AdanosSentiment):
        supports_csv_config = True

    run = FileSelection(
        observation_data=load_adanos_observations(FIXTURES / "adanos_two_years.json"),
        instruments=["AAPL"],
        backtest_period=PERIOD,
        source="local",
        local_data=wider,
        csv_config=True,
        strict_data_fetch=False,
        reports_directory=str(tmp_path),
    )
    with pytest.raises(ValueError, match="one matching instrument"):
        asyncio.run(run.run_strategy())


def test_missing_observation_use_cannot_finish_as_success(tmp_path, monkeypatch, offline):
    run = strategy(tmp_path, skip_analysis=True)
    monkeypatch.setattr(
        run, "_compute_signals", lambda: run.data.get_feature("adj_close").notna().astype(float)
    )
    with pytest.raises(ValueError, match="must call observations_at"):
        asyncio.run(run.run_strategy())


def test_capture_bounds_success_response_body(monkeypatch):
    from datetime import date

    from backtester.observations import adanos

    client = httpx.Client
    monkeypatch.setattr(adanos, "_LIMIT", 64)
    monkeypatch.setenv("ADANOS_API_KEY", "fixture")
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(
            **kwargs,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 65)),
        ),
    )
    with pytest.raises(RuntimeError, match="exceeds"):
        capture_adanos_snapshot(ticker="AAPL", day=date(2026, 10, 19))


def test_late_sample_selection_still_labels_synthetic_prices(tmp_path, monkeypatch, offline):
    observations = ObservationData(
        pd.DataFrame(columns=["observed_at", "data_date", "found", "mentions", "sentiment_score"]),
        instrument="AAPL",
        source="test",
        synthetic=False,
    )
    monkeypatch.setenv("QJ_SAMPLE_DATA", "1")
    run = AdanosSentiment(
        observation_data=observations,
        max_data_age_days=3,
        instruments=["AAPL"],
        backtest_period={"start": "2026-10-01", "end": "2026-10-30"},
        reports_directory=str(tmp_path),
        skip_analysis=True,
    )
    asyncio.run(run.run_strategy())
    assert "Synthetic" in run.strategy_type
    assert run._build_strategy_params()["Sample price mode"] == "True"
    assert run._observation_data.synthetic is False and not offline


def test_typed_empty_observations_remain_missing():
    table = pd.DataFrame(
        {
            "observed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "flag": pd.Series(dtype=bool),
            "count": pd.Series(dtype="int64"),
        }
    )
    observations = data(table, provenance_columns=("flag", "count"))
    aligned = observations.align(pd.date_range("2026-10-20", periods=2))
    assert aligned[["flag", "count"]].isna().all().all()
    assert not aligned.usable.any()
    assert all(
        row["flag"] is None and row["count"] is None for row in observations.provenance(aligned)
    )


def test_w26_rejects_universe_replacement_flag(tmp_path, monkeypatch, offline):
    monkeypatch.setenv("QJ_CSV_CONFIG", "1")
    monkeypatch.setenv("QJ_DATA_FILE", str(FIXTURES / "prices_two_years.csv"))
    monkeypatch.setenv("ADANOS_SNAPSHOT_FILE", str(FIXTURES / "adanos_two_years.json"))
    with pytest.raises(ValueError, match="does not opt in"):
        asyncio.run(_strategy.main())


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed_at", "2026-10-21T06:00:00"),
        ("data_date", "2026-10-21"),
        ("sentiment_score", 2.0),
        ("mentions", True),
    ],
)
def test_adanos_rejects_invalid_observations(tmp_path, field, value):
    payload = {
        "source": "reddit_stocks",
        "ticker": "AAPL",
        "synthetic": True,
        "observations": [
            {
                "observed_at": "2026-10-21T06:00:00Z",
                "data_date": "2026-10-20",
                "found": True,
                "mentions": 1,
                "sentiment_score": 0.5,
            }
        ],
    }
    payload["observations"][0][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_adanos_observations(path)


def test_generic_future_data_day_is_rejected():
    table = frame()
    table.loc[0, "data_date"] = "2026-10-22"
    with pytest.raises(ValueError, match="cannot follow"):
        data(table)


def test_capture_rejects_wrong_ticker(monkeypatch):
    from datetime import date

    client = httpx.Client
    monkeypatch.setenv("ADANOS_API_KEY", "fixture")
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(
            **kwargs,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"ticker": "MSFT", "found": True})
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="invalid observation"):
        capture_adanos_snapshot(ticker="AAPL", day=date(2026, 10, 19))


def test_fetch_errors_keep_the_original_cause(tmp_path, monkeypatch, offline):
    monkeypatch.setenv("QJ_STRICT_DATA_FETCH", "0")
    run = strategy(tmp_path, skip_analysis=True)

    async def fail_fetch():
        raise RuntimeError("Test-only fetch failure")

    monkeypatch.setattr(run, "_fetch_market_data", fail_fetch)
    with pytest.raises(RuntimeError, match="Test-only fetch failure"):
        asyncio.run(run.run_strategy())


@pytest.mark.parametrize(
    "attribute,value", [("instrument", "MSFT"), ("source", "changed"), ("synthetic", False)]
)
def test_observation_identity_labels_are_read_only(attribute, value):
    observations = data()
    before = observations.metadata
    with pytest.raises(AttributeError):
        setattr(observations, attribute, value)
    assert observations.metadata == before
    assert observations.instrument == before["instrument"]
    assert observations.source == before["source"]
    assert observations.synthetic is before["synthetic"]
