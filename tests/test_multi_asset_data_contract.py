# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from backtester import Backtester
from backtester.execution import (
    AssetClass,
    ContractSpec,
    contract_spec_from_mapping,
    get_contract_spec,
)
from backtester.sample_data import build_sample_bt_payload


def _instrument_specs() -> dict[str, dict[str, object]]:
    return {
        "EURUSD=X": {
            "provider_symbol": "EURUSD=X",
            "asset_class": "fx",
            "instrument_type": "spot_fx",
            "base_currency": "EUR",
            "quote_currency": "USD",
            "exchange": "OTC",
            "calendar": "FX_24_5",
            "multiplier": 1.0,
            "tick_size": 0.0001,
            "pip_size": 0.0001,
            "lot_size": 100_000.0,
            "margin": 0.0,
        },
        "ES=F": {
            "provider_symbol": "ES=F",
            "asset_class": "future",
            "instrument_type": "future_continuous",
            "quote_currency": "USD",
            "exchange": "CME",
            "calendar": "US_FUTURES_23_5",
            "multiplier": 50.0,
            "tick_size": 0.25,
            "lot_size": 1.0,
            "margin": 15_840.0,
            "continuous": True,
        },
    }


def _payload() -> dict[str, object]:
    payload = build_sample_bt_payload(
        instruments=["EURUSD=X", "ES=F"],
        start="2024-01-01",
        end="2024-12-31",
    )
    payload["instrument_specs"] = _instrument_specs()
    return payload


def _backtester(**kwargs) -> Backtester:
    return Backtester(
        instruments=["EURUSD=X", "ES=F"],
        backtest_period={"start": "2024-01-01", "end": "2024-12-31"},
        source="sample",
        execution_mode="orders",
        show_text_reports=False,
        skip_analysis=True,
        **kwargs,
    )


def test_contract_spec_mapping_and_yahoo_registry_aliases() -> None:
    spec = contract_spec_from_mapping("EURUSD=X", _instrument_specs()["EURUSD=X"])

    assert spec.asset_class == AssetClass.FX
    assert spec.symbol == "EURUSD=X"
    assert spec.lot_size == 100_000.0
    assert spec.tick_size == pytest.approx(0.0001)
    assert spec.base_currency == "EUR"
    assert spec.quote_currency == "USD"
    assert get_contract_spec("ES=F").multiplier == 50.0
    assert get_contract_spec("EURUSD=X").asset_class == AssetClass.FX
    assert get_contract_spec("NZDUSD=X").asset_class == AssetClass.FX
    assert get_contract_spec("ZC=F").asset_class == AssetClass.FUTURE


def test_api_specs_drive_groups_and_contract_aware_sizing() -> None:
    bt = _backtester()
    bt._api_response = _payload()

    asyncio.run(bt._process_market_data())

    assert bt.instruments_data.group_data.to_dict() == {
        "EURUSD=X": "fx",
        "ES=F": "future",
    }
    assert bt.instruments_data.group_order == ["fx", "future"]
    assert bt._contract_spec("EURUSD=X").lot_size == 100_000.0
    assert bt._contract_spec("ES=F").multiplier == 50.0
    assert bt._quantity_for_notional("EURUSD=X", 110_000.0, 1.10) == pytest.approx(1.0)
    assert bt._quantity_for_notional("ES=F", 250_000.0, 5_000.0) == pytest.approx(1.0)


def test_manual_contract_spec_overrides_api_metadata() -> None:
    manual = ContractSpec.future("ES=F", multiplier=5.0, tick_size=0.25)
    bt = _backtester(contract_specs={"ES=F": manual})
    bt._api_response = _payload()

    asyncio.run(bt._process_market_data())

    assert bt._contract_spec("ES=F") is manual
    assert bt._contract_spec("ES=F").multiplier == 5.0


def test_negative_futures_price_preserves_nav_cash_ledger_identity() -> None:
    class Data:
        def __init__(self, frames):
            self.frames = frames

        def get_feature(self, name):
            return self.frames[name]

    class Portfolio:
        cash_buffer = 0.05

        def update_net_asset_value(self, nav):
            self.net_asset_value = nav

        def update_positions(self, positions):
            self.positions = positions

        def update_weights(self, weights):
            self.weights = weights

        def update_cash(self, cash):
            self.cash = cash

        def assert_accounting_identity(self):
            expected = self.cash + self.position_values.sum(axis=1)
            pd.testing.assert_series_equal(self.net_asset_value, expected, check_names=False)

    class BuyAndHoldFuture(Backtester):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.submitted = False

        def _compute_orders(self, date, bars, current_positions, nav):
            if not self.submitted:
                self.order_market("CL=F", 1.0)
                self.submitted = True

    dates = pd.date_range("2020-04-17", periods=4, freq="B", tz="UTC")
    prices = pd.DataFrame({"CL=F": [10.0, 10.0, -37.0, 20.0]}, index=dates)
    frames = {
        "adj_close": prices,
        "open": prices,
        "high": prices,
        "low": prices,
        "volume": pd.DataFrame({"CL=F": [1_000.0] * 4}, index=dates),
    }
    spec = ContractSpec.future("CL=F", multiplier=1_000.0, tick_size=0.01)
    strategy = BuyAndHoldFuture(
        instruments=["CL=F"],
        backtest_period={"start": "2020-04-17", "end": "2020-04-23"},
        execution_mode="orders",
        contract_specs={"CL=F": spec},
        show_text_reports=False,
        skip_analysis=True,
    )
    strategy.instruments_data = Data(frames)
    strategy.portfolio_data = Portfolio()

    strategy._compute_performance_order_based()

    assert spec.notional(1.0, -37.0) == pytest.approx(37_000.0)
    assert strategy.portfolio_data.net_asset_value.loc[dates[2]] == pytest.approx(53_000.0)
    assert strategy.portfolio_data.net_asset_value.loc[dates[3]] == pytest.approx(110_000.0)
