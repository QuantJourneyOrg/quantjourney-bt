# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Fail-closed tests for external contract metadata and FX settlement."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from backtester import Backtester
from backtester.execution import UnsupportedCurrencyConversionError
from backtester.execution.contract_spec import (
    AssetClass,
    ContractSpec,
    contract_spec_from_mapping,
)
from backtester.portfolio.accounting import PortfolioLedger, build_weight_ledger
from backtester.portfolio.weight_cost import FixedBpsWeightCostModel
from backtester.sample_data import build_sample_bt_payload


def _valid_fx_mapping(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "asset_class": "fx",
        "base_currency": "EUR",
        "quote_currency": "USD",
        "lot_size": 100_000.0,
    }
    values.update(overrides)
    return values


def _backtester(
    *,
    base_currency: str = "USD",
    contract_specs: dict[str, ContractSpec] | None = None,
) -> Backtester:
    return Backtester(
        instruments=[],
        backtest_period={"start": "2024-01-01", "end": "2024-12-31"},
        source="sample",
        base_currency=base_currency,
        contract_specs=contract_specs,
        show_text_reports=False,
        skip_analysis=True,
    )


def test_fx_quote_must_equal_portfolio_settlement_currency() -> None:
    usd_portfolio = _backtester()

    with pytest.raises(
        UnsupportedCurrencyConversionError,
        match=r"USDJPY.*JPY.*settles in USD",
    ):
        usd_portfolio._contract_spec("USDJPY")

    # USD-quoted FX remains supported, and a JPY portfolio may use JPY PnL.
    assert usd_portfolio._contract_spec("EURUSD").quote_currency == "USD"
    assert _backtester(base_currency="JPY")._contract_spec("USDJPY").pnl(
        1.0, 150.0, 151.0
    ) == pytest.approx(100_000.0)


def test_currency_guard_does_not_restrict_equities_or_futures() -> None:
    specs = {
        "DAX": ContractSpec.future("DAX", multiplier=25.0, quote_currency="EUR"),
        "SAP": ContractSpec.equity("SAP", quote_currency="EUR"),
    }
    bt = _backtester(contract_specs=specs)

    assert bt._contract_spec("DAX").asset_class == AssetClass.FUTURE
    assert bt._contract_spec("SAP").asset_class == AssetClass.EQUITY


def test_standalone_ledgers_validate_fx_settlement_currency() -> None:
    spec = ContractSpec.fx("USDJPY", base_currency="USD", quote_currency="JPY", pip_size=0.01)

    def resolver(instrument: str) -> ContractSpec:
        return spec

    with pytest.raises(UnsupportedCurrencyConversionError):
        PortfolioLedger(
            initial_cash=1_000_000.0,
            instruments=["USDJPY"],
            contract_spec_resolver=resolver,
        )
    ledger = PortfolioLedger(
        initial_cash=1_000_000.0,
        instruments=["USDJPY"],
        contract_spec_resolver=resolver,
        settlement_currency="JPY",
    )
    assert ledger.contract_spec("USDJPY") is spec

    dates = pd.date_range("2024-01-01", periods=2)
    weights = pd.DataFrame({"USDJPY": [0.0, 0.5]}, index=dates)
    prices = pd.DataFrame({"USDJPY": [150.0, 151.0]}, index=dates)
    kwargs = {
        "actual_weights": weights,
        "portfolio_returns": pd.Series(0.0, index=dates),
        "prices": prices,
        "initial_capital": 1_000_000.0,
        "rebalance_flags": pd.Series(True, index=dates),
        "cost_model": FixedBpsWeightCostModel(total_bps=0.0),
        "contract_spec_resolver": resolver,
    }
    with pytest.raises(UnsupportedCurrencyConversionError):
        build_weight_ledger(**kwargs)
    result, _, _ = build_weight_ledger(**kwargs, settlement_currency="JPY")
    assert result.positions.iloc[-1, 0] > 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("multiplier", "not-a-number"),
        ("tick_size", float("nan")),
        ("margin", -1.0),
        ("lot_size", 0.0),
        ("quantity_step", 0.0),
        ("min_quantity", -0.01),
        ("pip_size", None),
    ],
)
def test_mapping_rejects_invalid_numeric_values(field: str, value: object) -> None:
    values = _valid_fx_mapping(**{field: value})

    with pytest.raises((TypeError, ValueError), match=field):
        contract_spec_from_mapping("EURUSD", values)


def test_mapping_rejects_unknown_asset_class() -> None:
    with pytest.raises(ValueError, match=r"unsupported asset_class.*swap"):
        contract_spec_from_mapping("IRS", {"asset_class": "swap"})


def test_mapping_parses_false_strings_without_truthiness_bug() -> None:
    spec = contract_spec_from_mapping(
        "EURUSD",
        _valid_fx_mapping(continuous="false", inverse="false"),
    )

    assert spec.continuous is False
    assert spec.inverse is False

    with pytest.raises(ValueError, match="inverse must be a boolean"):
        contract_spec_from_mapping("EURUSD", _valid_fx_mapping(inverse="sometimes"))


@pytest.mark.parametrize(
    ("values", "missing_field"),
    [
        ({}, "asset_class"),
        ({"asset_class": "fx"}, "base_currency"),
        (
            {
                "asset_class": "fx",
                "base_currency": "USD",
                "lot_size": 100_000.0,
            },
            "quote_currency",
        ),
        (
            {
                "asset_class": "fx",
                "base_currency": "USD",
                "quote_currency": "JPY",
            },
            "lot_size",
        ),
        ({"asset_class": "future"}, "multiplier"),
    ],
)
def test_external_mapping_requires_complete_economic_metadata(
    values: dict[str, object], missing_field: str
) -> None:
    with pytest.raises(ValueError, match=missing_field):
        contract_spec_from_mapping("USDJPY", values)


def test_fx_mapping_validates_explicit_pair_identity() -> None:
    spec = contract_spec_from_mapping(
        "USDJPY=X",
        {
            "asset_class": "fx",
            "base_currency": "USD",
            "quote_currency": "JPY",
            "lot_size": 100_000.0,
        },
    )
    assert spec.base_currency == "USD"
    assert spec.quote_currency == "JPY"

    with pytest.raises(ValueError, match=r"symbol implies USD/JPY.*EUR/USD"):
        contract_spec_from_mapping(
            "USDJPY",
            {
                "asset_class": "fx",
                "base_currency": "EUR",
                "quote_currency": "USD",
                "lot_size": 100_000.0,
            },
        )


def test_invalid_api_contract_spec_is_not_silently_ignored() -> None:
    bt = _backtester()
    payload = build_sample_bt_payload(
        instruments=["EURUSD=X"],
        start="2024-01-01",
        end="2024-12-31",
    )
    payload["instrument_specs"] = {"EURUSD=X": _valid_fx_mapping(lot_size="broken")}
    bt._api_response = payload

    with pytest.raises(ValueError, match=r"EURUSD=X.*lot_size"):
        asyncio.run(bt._process_market_data())


def test_non_object_api_contract_spec_is_not_silently_ignored() -> None:
    bt = _backtester()
    payload = build_sample_bt_payload(
        instruments=["EURUSD=X"],
        start="2024-01-01",
        end="2024-12-31",
    )
    payload["instrument_specs"] = {"EURUSD=X": ["fx", 100_000]}
    bt._api_response = payload

    with pytest.raises(ValueError, match=r"EURUSD=X.*expected an object"):
        asyncio.run(bt._process_market_data())
