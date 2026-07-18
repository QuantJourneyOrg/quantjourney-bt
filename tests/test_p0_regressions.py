# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from backtester.engines.blotter import Blotter
from backtester.execution import BarData, ContractSpec
from backtester.reporting_frequency import infer_periods_per_year
from backtester.risk import InverseVolModel, PositionLimitModel, RiskParityModel
from backtester.sample_data import build_sample_bt_payload
from backtester.walkforward.runner import FoldRunner


def _backtester(**kwargs) -> Backtester:
    return Backtester(
        instruments=kwargs.pop("instruments", ["AAPL"]),
        backtest_period={"start": "2024-01-01", "end": "2024-12-31"},
        show_text_reports=False,
        skip_analysis=True,
        **kwargs,
    )


def test_position_limit_does_not_create_inactive_positions():
    dates = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"A": [0.8, 0.8], "B": 0.0, "C": 0.0}, index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=weights.columns)
    adjusted = PositionLimitModel(max_weight=0.4).adjust(weights, returns)
    assert adjusted.iloc[-1].to_dict() == {"A": 0.4, "B": 0.0, "C": 0.0}


def test_risk_parity_preserves_strategy_signs():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=80)
    returns = pd.DataFrame(
        rng.normal(0.0, [0.01, 0.015, 0.02], size=(len(dates), 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    weights = pd.DataFrame({"A": 0.5, "B": -0.5, "C": 0.0}, index=dates)
    adjusted = RiskParityModel(lookback=20, rebalance_freq="D").adjust(weights, returns)
    assert adjusted.iloc[-1]["A"] > 0.0
    assert adjusted.iloc[-1]["B"] < 0.0
    assert adjusted.iloc[-1]["C"] == 0.0


def test_pure_inverse_vol_preserves_strategy_signs():
    dates = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame({"A": 0.5, "B": -0.5, "C": 0.0}, index=dates)
    returns = pd.DataFrame(
        {"A": [0.01, -0.01, 0.0], "B": [0.02, -0.02, 0.0], "C": 0.0},
        index=dates,
    )

    adjusted = InverseVolModel(
        lookback=2,
        ann_factor=1.0,
        blend_alpha=False,
    ).adjust(weights, returns)

    assert adjusted.iloc[-1]["A"] > 0.0
    assert adjusted.iloc[-1]["B"] < 0.0
    assert adjusted.iloc[-1]["C"] == 0.0
    assert adjusted.iloc[-1].abs().sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("symbol", "spec", "price", "nav", "percent", "expected_quantity"),
    [
        ("ES", ContractSpec.future("ES", multiplier=50), 5_000.0, 1_000_000.0, 0.5, 2.0),
        ("EURUSD", ContractSpec.fx("EURUSD", lot_size=100_000), 1.1, 110_000.0, 1.0, 1.0),
    ],
)
def test_percent_sizing_uses_contract_notional(
    symbol, spec, price, nav, percent, expected_quantity
):
    bt = _backtester(instruments=[symbol], execution_mode="orders", contract_specs={symbol: spec})
    date = pd.Timestamp("2024-01-02", tz="UTC")
    bt._order_context = {
        "date": date,
        "bars": {symbol: BarData(date, price, price, price, price, 1_000_000)},
        "positions": {symbol: 0.0},
        "nav": nav,
    }
    bt.order_percent(symbol, percent)
    assert bt.fill_engine.pending_orders[0].quantity == pytest.approx(expected_quantity)


def test_bulk_trade_with_supplied_order_id_still_creates_order_record():
    blotter = Blotter()
    trades = pd.DataFrame(
        {
            "OrderID": ["order-1"],
            "Timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
            "Instrument": ["AAPL"],
            "Side": ["buy"],
            "Quantity": [10.0],
            "Price": [100.0],
            "TradeValue": [1_000.0],
        }
    )
    blotter.record_trades_bulk(trades)
    assert len(blotter.get_orders_dataframe()) == 1


def test_walkforward_reads_canonical_blotter_columns():
    dates = pd.bdate_range("2024-01-02", periods=30, tz="UTC")
    nav = pd.Series(100_000.0 * np.cumprod(1.0 + np.linspace(-0.001, 0.002, 30)), index=dates)
    pdata = SimpleNamespace(net_asset_value=nav, periods_per_year=252)
    blotter = Blotter()
    blotter.record_trade("o1", "AAPL", "buy", 100.0, 100.0, 10_000.0, dates[5])
    blotter.record_trade("o2", "AAPL", "sell", 100.0, 101.0, 10_100.0, dates[10])
    runner = FoldRunner(SimpleNamespace(fold_id=1), pdata, blotter=blotter)
    metrics = runner._compute_metrics_for_window(dates[1], dates[-1])
    assert metrics["n_trades"] == 2
    assert np.isfinite(metrics["turnover_ann"])


def test_data_completeness_is_fail_closed():
    bt = _backtester(instruments=["AAPL", "MSFT"])
    bt._api_response = {"instrument_names": ["AAPL"], "partial": True}
    assert bt._strict_data_fetch is True
    with pytest.raises(ValueError, match="Incomplete market-data response"):
        bt._validate_data_completeness_response()


def test_five_minute_frequency_and_sample_payload():
    sessions = [
        pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=78, freq="5min")
        for day in pd.bdate_range("2024-01-02", periods=3)
    ]
    assert infer_periods_per_year(sessions[0].append(sessions[1:])) == 252 * 78
    payload = build_sample_bt_payload(
        instruments=["AAPL", "MSFT"], start="2024-01-01", end="2024-12-31"
    )
    assert payload["instrument_names"] == ["AAPL", "MSFT"]
