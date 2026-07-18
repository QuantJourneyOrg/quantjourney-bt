# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtester.portfolio import StrategyBook, StrategyBookResult
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.portfolio.weight_cost import FixedBpsWeightCostModel


def _dates(periods: int = 4) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=periods)


def test_static_book_applies_close_target_from_next_period() -> None:
    dates = _dates(3)
    result = StrategyBook(
        sleeves={
            "growth": pd.Series([100.0, 110.0, 121.0], index=dates),
            "stable": pd.Series([100.0, 100.0, 100.0], index=dates),
        },
        allocations={"growth": 0.5, "stable": 0.5},
        initial_capital=100_000.0,
    ).run()

    assert isinstance(result, StrategyBookResult)
    assert result.gross_returns.tolist() == pytest.approx([0.0, 0.05, 0.05])
    assert result.nav.tolist() == pytest.approx([100_000.0, 105_000.0, 110_250.0])
    assert result.rebalance_flags.all()
    np.testing.assert_allclose(
        result.actual_allocations.to_numpy(),
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    )
    pd.testing.assert_series_equal(
        result.cash + result.sleeve_values.sum(axis=1),
        result.nav,
        check_names=False,
    )


def test_book_without_scheduled_rebalance_allows_allocations_to_drift() -> None:
    dates = _dates(3)
    result = StrategyBook(
        sleeves={
            "growth": pd.Series([100.0, 110.0, 121.0], index=dates),
            "stable": pd.Series([100.0, 100.0, 100.0], index=dates),
        },
        allocations={"growth": 0.5, "stable": 0.5},
        rebalance_policy=RebalancePolicy(frequency=None),
    ).run()

    assert result.rebalance_flags.tolist() == [True, False, False]
    assert result.actual_allocations.iloc[1, 0] == pytest.approx(0.55 / 1.05)
    assert result.actual_allocations.iloc[1, 1] == pytest.approx(0.50 / 1.05)


def test_dynamic_allocations_are_point_in_time_and_never_backfilled() -> None:
    dates = _dates(3).tz_localize("UTC")
    sleeves = {
        "a": pd.Series([100.0, 110.0, 110.0], index=dates),
        "b": pd.Series([100.0, 100.0, 120.0], index=dates),
    }
    allocations = pd.DataFrame(
        {"a": [1.0, 0.0], "b": [0.0, 1.0]},
        index=pd.DatetimeIndex(
            [dates[0] - pd.Timedelta(days=1), dates[0] + pd.Timedelta(hours=12)]
        ),
    )

    result = StrategyBook(sleeves, allocations).run()

    assert result.target_allocations.iloc[0].to_dict() == {"a": 1.0, "b": 0.0}
    assert result.target_allocations.iloc[1].to_dict() == {"a": 0.0, "b": 1.0}
    # The new target is applied at the second close. The old A allocation
    # therefore earns the second row's return; B starts earning on row three.
    assert result.gross_returns.tolist() == pytest.approx([0.0, 0.10, 0.20])

    late_allocations = allocations.iloc[[1]]
    with pytest.raises(ValueError, match="initial point-in-time"):
        StrategyBook(sleeves, late_allocations).run()


def test_return_series_and_portfolio_data_like_nav_are_supported() -> None:
    dates = _dates(3)
    returns = pd.Series([0.0, 0.10, -0.05], index=dates)
    portfolio_data_like = SimpleNamespace(
        net_asset_value=pd.Series([50.0, 50.0, 55.0], index=dates)
    )

    result = StrategyBook(
        sleeves={"return_sleeve": returns, "nav_sleeve": portfolio_data_like},
        allocations={"return_sleeve": 0.5, "nav_sleeve": 0.5},
        series_kind={"return_sleeve": "returns", "nav_sleeve": "nav"},
    ).run()

    assert result.sleeve_nav["return_sleeve"].tolist() == pytest.approx([1.0, 1.1, 1.045])
    assert result.sleeve_nav["nav_sleeve"].tolist() == pytest.approx([50.0, 50.0, 55.0])


def test_initial_allocation_cost_reduces_first_nav_and_reconciles() -> None:
    dates = _dates(3)
    result = StrategyBook(
        sleeves={"cash_like": pd.Series(100.0, index=dates)},
        allocations={"cash_like": 1.0},
        weight_cost_model=FixedBpsWeightCostModel(total_bps=100.0),
        initial_capital=100_000.0,
    ).run()

    expected_nav = 100_000.0 / 1.01
    expected_cost = 100_000.0 - expected_nav
    assert result.costs.tolist() == pytest.approx([expected_cost, 0.0, 0.0])
    assert result.returns.tolist() == pytest.approx([expected_nav / 100_000.0 - 1.0, 0.0, 0.0])
    assert result.nav.tolist() == pytest.approx([expected_nav] * 3)
    assert result.turnover.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert result.stats["total_costs"] == pytest.approx(expected_cost)
    assert result.stats["underlying_security_netting"] is False
    assert result.transaction_costs is result.costs


@pytest.mark.parametrize(
    ("sleeve", "message"),
    [
        (pd.Series([100.0, 0.0], index=_dates(2)), "NAV must stay positive"),
        (pd.Series([0.0, -1.0], index=_dates(2)), "return <= -100%"),
    ],
)
def test_invalid_sleeve_paths_fail_loudly(sleeve: pd.Series, message: str) -> None:
    kind = "returns" if "return" in message else "nav"
    with pytest.raises(ValueError, match=message):
        StrategyBook(
            sleeves={"bad": sleeve},
            allocations={"bad": 1.0},
            series_kind=kind,
        ).run()


def test_unknown_allocation_sleeve_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sleeves"):
        StrategyBook(
            sleeves={"known": pd.Series(100.0, index=_dates(2))},
            allocations={"known": 1.0, "typo": 0.1},
        ).run()
