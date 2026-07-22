"""Parity coverage for performance-only engine refactors.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from backtester.execution.contract_spec import ContractSpec, get_contract_spec
from backtester.execution.order_types import Fill, OrderSide
from backtester.portfolio.accounting.ledger import PortfolioLedger
from backtester.risk.inverse_vol import InverseVolModel
from backtester.risk.risk_parity import RiskParityModel


def _legacy_inverse_vol(
    model: InverseVolModel,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    out = weights.copy()
    ann_factor = float(model.ann_factor) if model.ann_factor is not None else np.sqrt(252.0)
    for i in range(model.lookback, len(weights)):
        row_w = weights.iloc[i]
        active = row_w.abs() > 1e-10
        if active.sum() == 0:
            out.iloc[i] = 0.0
            continue
        window = returns.iloc[max(0, i - model.lookback) : i]
        inv_vol = 1.0 / (window.std() * ann_factor).clip(lower=model.min_vol)
        if model.blend_alpha:
            blended = (row_w * inv_vol).where(active, 0.0)
        else:
            blended = (np.sign(row_w) * inv_vol).where(active, 0.0)
        total = blended.abs().sum()
        if total < 1e-10:
            out.iloc[i] = 0.0
            continue
        out.iloc[i] = blended / total * row_w.abs().sum()
    return out


def _legacy_risk_parity(
    model: RiskParityModel,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    out = weights.copy()
    periods = pd.Series(weights.index.to_period("M"), index=weights.index)
    is_rescore = periods != periods.shift(1)
    previous = None
    previous_signature = None
    for i in range(model.lookback, len(weights)):
        row_w = weights.iloc[i]
        active = row_w.abs() > 1e-10
        n_active = int(active.sum())
        signature = tuple(np.sign(row_w.to_numpy(dtype=float)))
        if n_active < 2:
            out.iloc[i] = row_w
            previous = None
            previous_signature = signature
            continue
        if is_rescore.iloc[i] or previous is None or signature != previous_signature:
            window = returns.iloc[max(0, i - model.lookback) : i]
            active_columns = row_w.index[active]
            signs = np.sign(row_w.loc[active_columns].to_numpy(dtype=float))
            covariance = window[active_columns].mul(signs, axis=1).cov().values
            solved = model._solve_erc(covariance, n_active)
            full = pd.Series(0.0, index=row_w.index)
            for position, column in enumerate(active_columns):
                full[column] = signs[position] * solved[position]
            previous = full * row_w.abs().sum()
            previous_signature = signature
        out.iloc[i] = previous
    return out


def test_engines_package_keeps_reporting_modules_lazy() -> None:
    code = """
import sys
import backtester.engines
assert 'backtester.engines.performance' not in sys.modules
assert 'backtester.portfolio.portfolio_plots' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_inverse_vol_vectorization_matches_legacy_loop() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=140)
    columns = ["A", "B", "C", "D"]
    returns = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), 4)), index=dates, columns=columns)
    returns.iloc[30:33, 2] = np.nan
    raw = rng.normal(size=(len(dates), 4))
    raw[20:25, :] = 0.0
    weights = pd.DataFrame(raw, index=dates, columns=columns)
    weights = weights.div(weights.abs().sum(axis=1).replace(0.0, 1.0), axis=0)

    for blend_alpha in (False, True):
        model = InverseVolModel(lookback=21, blend_alpha=blend_alpha)
        expected = _legacy_inverse_vol(model, weights, returns)
        actual = model.adjust(weights, returns)
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-14)


def test_risk_parity_array_loop_matches_legacy_pandas_loop() -> None:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=120)
    columns = ["A", "B", "C", "D"]
    returns = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), 4)), index=dates, columns=columns)
    weights = pd.DataFrame(0.25, index=dates, columns=columns)
    weights.loc[dates[45:75], "D"] = -0.25
    weights.loc[dates[75:90], "C"] = 0.0
    model = RiskParityModel(lookback=20, rebalance_freq="BMS")

    expected = _legacy_risk_parity(model, weights, returns)
    actual = model.adjust(weights, returns)

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-14)


def _record_ledger(*, prepared: bool):
    dates = pd.bdate_range("2024-01-02", periods=4)
    instruments = ["A", "B"]
    specs = {instrument: ContractSpec.equity(instrument) for instrument in instruments}
    ledger = PortfolioLedger(
        initial_cash=10_000.0,
        instruments=instruments,
        contract_spec_resolver=specs.__getitem__,
    )
    if prepared:
        ledger.prepare_history(dates)
    for row, date in enumerate(dates):
        prices = {"A": 100.0 + row, "B": 50.0 - row}
        for instrument, price in prices.items():
            ledger.observe_mark(instrument, price)
        if row == 0:
            ledger.apply_fill(Fill("a-buy", "A", OrderSide.BUY, 10.0, 100.0, timestamp=date))
        elif row == 1:
            ledger.apply_fill(Fill("b-buy", "B", OrderSide.BUY, 5.0, 49.0, timestamp=date))
        elif row == 3:
            ledger.apply_fill(Fill("a-sell", "A", OrderSide.SELL, 4.0, 103.0, timestamp=date))
        nav, position_values = ledger.mark_to_market(prices)
        ledger.record(date=date, nav=nav, position_values=position_values, prices=prices)
    return ledger.result()


def test_preallocated_ledger_history_matches_append_history() -> None:
    expected = _record_ledger(prepared=False)
    actual = _record_ledger(prepared=True)
    for field in (
        "nav",
        "cash",
        "positions",
        "position_values",
        "weights",
        "book_weights",
        "exposure_values",
        "exposure_weights",
        "returns",
        "margin_by_instrument",
        "margin_used",
        "buying_power",
        "average_entry_price_history",
    ):
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if isinstance(expected_value, pd.Series):
            pd.testing.assert_series_equal(actual_value, expected_value)
        else:
            pd.testing.assert_frame_equal(actual_value, expected_value)


def test_ledger_resolves_each_immutable_contract_once() -> None:
    calls: dict[str, int] = {}

    def resolver(instrument: str) -> ContractSpec:
        calls[instrument] = calls.get(instrument, 0) + 1
        return ContractSpec.equity(instrument)

    ledger = PortfolioLedger(
        initial_cash=10_000.0,
        instruments=["A", "B"],
        contract_spec_resolver=resolver,
    )
    for _ in range(20):
        assert ledger.contract_spec("A").symbol == "A"
        assert ledger.is_valid_price(100.0, "B")
    assert calls == {"A": 1, "B": 1}


def test_default_contract_spec_lookup_reuses_immutable_instance() -> None:
    assert get_contract_spec("unknown-performance-symbol") is get_contract_spec(
        "UNKNOWN-PERFORMANCE-SYMBOL"
    )


def test_cached_portfolio_returns_preserve_tracking_error_math() -> None:
    rng = np.random.default_rng(23)
    return_weights = rng.normal(0.0, 0.1, (500, 12))
    returns = rng.normal(0.0, 0.01, (500, 12))
    benchmark = rng.normal(0.0, 0.01, 500)
    cached = np.array(
        [(return_weights[row] * returns[row]).sum() for row in range(len(return_weights))]
    )
    for end in range(21, len(cached)):
        legacy = np.array(
            [(return_weights[row] * returns[row]).sum() for row in range(end - 21, end)]
        )
        expected = np.std(legacy - benchmark[end - 21 : end], ddof=1)
        actual = np.std(cached[end - 21 : end] - benchmark[end - 21 : end], ddof=1)
        assert actual == expected
