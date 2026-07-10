# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Regression tests for the canonical strategy-data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtester.core import Backtester
from backtester.portfolio.instr_data import InstrumentData

INSTRUMENTS = ["AAPL", "MSFT"]


def _instrument_data(*, strategies: pd.DataFrame | None = None) -> InstrumentData:
    index = pd.bdate_range("2025-01-02", periods=4, tz="UTC")
    prices = pd.DataFrame(
        100.0,
        index=index,
        columns=pd.MultiIndex.from_product(
            [INSTRUMENTS, ["adj_close"]],
            names=["instrument", "price"],
        ),
    )
    metrics = pd.DataFrame(
        0.0,
        index=index,
        columns=pd.MultiIndex.from_product(
            [INSTRUMENTS, ["returns"]],
            names=["instrument", "metric"],
        ),
    )
    parameters = pd.DataFrame(
        0.0,
        index=index,
        columns=pd.MultiIndex.from_product(
            [INSTRUMENTS, ["units"]],
            names=["instrument", "parameter"],
        ),
    )
    if strategies is None:
        strategies = pd.DataFrame(index=index)
    return InstrumentData(
        group_data=pd.Series(["equity", "equity"], index=INSTRUMENTS),
        group_order=["equity"],
        strategies=strategies,
        prices=prices,
        metrics=metrics,
        parameters=parameters,
        _skip_validation=True,
    )


def _frame(data: InstrumentData, value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(value, index=data.get_dates(), columns=INSTRUMENTS)


def _backtester(data: InstrumentData) -> Backtester:
    backtester = object.__new__(Backtester)
    backtester.instruments_data = data
    backtester.strategy_name = "alpha"
    backtester.execution_mode = "weights"
    return backtester


def test_add_strategy_data_is_immutable_and_never_intersects_indexes() -> None:
    data = _instrument_data()
    signals = _frame(data, 1.0)
    original = signals.copy(deep=True)

    data.add_strategy_data("alpha", "signals", signals)

    assert_frame_equal(signals, original)
    assert data.strategies.index.equals(data.prices.index)
    assert data.strategies.columns.names == ["strategy", "field", "instrument"]

    before = data.strategies.copy(deep=True)
    with pytest.raises(ValueError, match="index must exactly match"):
        data.add_strategy_data("alpha", "weights", signals.iloc[:-1])
    assert_frame_equal(data.strategies, before)


def test_add_strategy_data_replaces_same_field_without_duplicate_columns() -> None:
    data = _instrument_data()
    data.add_strategy_data("alpha", "signals", _frame(data, 0.0))
    data.add_strategy_data("alpha", "signals", _frame(data, 2.0))

    assert not data.strategies.columns.has_duplicates
    assert (data.get_strategy_data("alpha", "signals") == 2.0).all().all()


def test_strategy_accessors_share_one_column_contract() -> None:
    data = _instrument_data()
    data.add_strategy_data("alpha", "signals", _frame(data, 1.0))
    data.add_strategy_data("alpha", "weights", _frame(data, 0.25))

    strategy_first = data.get_strategy("alpha", "signals", orientation="strategy_first")
    instrument_first = data.get_strategy("alpha", "signals")
    assert strategy_first.columns.names == ["strategy", "field", "instrument"]
    assert instrument_first.columns.names == ["instrument", "strategy", "field"]

    signals = data.get_feature("strategies", "alpha", "signals")
    assert list(signals.columns) == INSTRUMENTS
    all_signals = data.get_feature("strategies", level="signals")
    assert all_signals.columns.names == ["instrument", "strategy"]
    assert_frame_equal(all_signals, data.get_feature("signals"))

    aapl = data.get_instrument_data("AAPL")
    assert not aapl["strategies"].empty
    assert aapl["strategies"].columns.names == ["strategy", "field"]

    data.remove_instrument("AAPL")
    assert "AAPL" not in data.strategies.columns.get_level_values("instrument")
    assert "MSFT" in data.strategies.columns.get_level_values("instrument")


def test_legacy_instrument_first_frame_is_normalized_on_construction() -> None:
    base = _instrument_data()
    legacy = pd.DataFrame(
        1.0,
        index=base.get_dates(),
        columns=pd.MultiIndex.from_product(
            [INSTRUMENTS, ["legacy"], ["signals"]],
            names=["instrument", "strategy", "field"],
        ),
    )

    normalized = _instrument_data(strategies=legacy)

    assert normalized.strategies.columns.names == [
        "strategy",
        "field",
        "instrument",
    ]
    assert list(normalized.get_strategy_data("legacy", "signals").columns) == INSTRUMENTS


@pytest.mark.parametrize(
    ("columns", "strategy", "field"),
    [
        (
            pd.MultiIndex.from_product([INSTRUMENTS, ["legacy"], ["signals"]]),
            "legacy",
            "signals",
        ),
        (
            pd.MultiIndex.from_product([["canonical"], ["weights"], INSTRUMENTS]),
            "canonical",
            "weights",
        ),
    ],
)
def test_unnamed_legacy_strategy_layouts_are_inferred_from_universe(
    columns: pd.MultiIndex, strategy: str, field: str
) -> None:
    base = _instrument_data()
    unnamed = pd.DataFrame(1.0, index=base.get_dates(), columns=columns)

    normalized = _instrument_data(strategies=unnamed)

    assert normalized.strategies.columns.names == [
        "strategy",
        "field",
        "instrument",
    ]
    assert list(normalized.get_strategy_data(strategy, field).columns) == INSTRUMENTS


def test_canonicalize_strategies_honors_requested_orientation_with_a_copy() -> None:
    data = _instrument_data()
    data.add_strategy_data("alpha", "signals", _frame(data, 1.0))
    stored = data.strategies.copy(deep=True)

    instrument_first = data.canonicalize_strategies("instrument_first")
    strategy_first = data.canonicalize_strategies("strategy_first")

    assert instrument_first.columns.names == ["instrument", "strategy", "field"]
    assert strategy_first.columns.names == ["strategy", "field", "instrument"]
    instrument_first.iloc[0, 0] = 999.0
    assert_frame_equal(data.strategies, stored)


def test_all_zero_signals_and_weights_are_valid_flat_strategy_outputs() -> None:
    data = _instrument_data()
    backtester = _backtester(data)
    signals = _frame(data, 0.0)
    weights = _frame(data, 0.0)
    backtester._compute_signals = lambda: signals
    backtester._compute_weights = lambda: weights

    backtester._generate_signals()
    backtester._generate_weights()

    assert (data.get_strategy_data("alpha", "signals") == 0.0).all().all()
    assert (data.get_strategy_data("alpha", "weights") == 0.0).all().all()


def test_positions_use_the_full_fail_closed_strategy_output_contract() -> None:
    data = _instrument_data()
    backtester = _backtester(data)
    positions = _frame(data, 3.0)
    backtester._compute_positions = lambda: positions

    backtester._generate_positions()

    assert (data.get_strategy_data("alpha", "positions") == 3.0).all().all()
    sparse_positions = positions.copy()
    sparse_positions.iloc[0, 0] = np.nan
    backtester._compute_positions = lambda: sparse_positions
    with pytest.raises(ValueError, match="finite"):
        backtester._generate_positions()


def test_generic_mutators_cannot_bypass_the_strategy_column_schema() -> None:
    data = _instrument_data()
    data.add_strategy_data("alpha", "signals", _frame(data, 1.0))
    stored = data.strategies.copy(deep=True)

    with pytest.raises(ValueError, match="add_strategy_data"):
        data.add_feature("signals", _frame(data, 2.0), "strategies")
    with pytest.raises(ValueError, match="cannot infer strategy names"):
        data.add_instrument("TSLA", {"signals": pd.Series(1.0, index=data.get_dates())})

    assert_frame_equal(data.strategies, stored)
    assert data.strategies.columns.nlevels == 3


def test_strategy_output_validation_rejects_misaligned_or_unsafe_frames() -> None:
    data = _instrument_data()
    backtester = _backtester(data)
    valid = _frame(data, 0.0)

    with pytest.raises(ValueError, match="DataFrame"):
        backtester._validate_signals([0.0, 1.0])
    with pytest.raises(ValueError, match="exactly match the market-data index"):
        backtester._validate_signals(valid.iloc[:-1])
    with pytest.raises(ValueError, match="exactly match the strategy universe"):
        backtester._validate_weights(valid[["AAPL"]])

    non_numeric = valid.astype(object)
    non_numeric.iloc[0, 0] = "long"
    with pytest.raises(ValueError, match="must be numeric"):
        backtester._validate_signals(non_numeric)

    non_finite = valid.copy()
    non_finite.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        backtester._validate_weights(non_finite)


def test_strategy_output_column_order_is_normalized_without_input_mutation() -> None:
    data = _instrument_data()
    backtester = _backtester(data)
    reversed_frame = _frame(data, 1.0)[list(reversed(INSTRUMENTS))]
    original = reversed_frame.copy(deep=True)

    normalized = backtester._validate_signals(reversed_frame)

    assert list(normalized.columns) == INSTRUMENTS
    assert_frame_equal(reversed_frame, original)
