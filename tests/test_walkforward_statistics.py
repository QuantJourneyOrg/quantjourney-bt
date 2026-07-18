# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Regression tests for honest walk-forward statistical labels."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.folds.base import Fold
from backtester.walkforward.folds.purge import compute_pre_oos_purge, compute_purge_embargo
from backtester.walkforward.result import WalkForwardResult
from backtester.walkforward.runner import FoldRunner
from backtester.walkforward.statistics.deflated_sharpe import (
    _expected_max_sr,
    deflated_sharpe,
)
from backtester.walkforward.statistics.interpretation import interpret_metrics
from backtester.walkforward.statistics.pbo import (
    pbo_from_selected_ranks,
    selected_trial_rank_logit,
    walk_forward_top_k_rank_failure_rate,
)


def test_top_k_rank_failure_is_explicitly_not_exposed_as_pbo_only() -> None:
    logits = [1.0, -1.0, 2.0, -2.0]
    assert walk_forward_top_k_rank_failure_rate(logits) == pytest.approx(0.5)

    result = WalkForwardResult(
        folds=[],
        config_dict={},
        walk_forward_top_k_rank_failure_rate=0.5,
        rank_stability_available=True,
    )
    payload = result.to_dict()
    assert payload["walk_forward_top_k_rank_failure_rate"] == pytest.approx(0.5)
    assert result.pbo == pytest.approx(0.5)  # deprecated read alias
    assert "WF top-K rank failure" in result.summary()


def test_legacy_pbo_function_warns_and_delegates() -> None:
    with pytest.warns(DeprecationWarning, match="not canonical CSCV PBO"):
        value = pbo_from_selected_ranks([1.0, -1.0])
    assert value == pytest.approx(0.5)


def test_selected_trial_rank_logit_uses_oos_rank() -> None:
    assert selected_trial_rank_logit(1.0, [1.0, 0.5, -0.5]) > 0.0
    assert selected_trial_rank_logit(-0.5, [1.0, 0.5, -0.5]) < 0.0


def test_rank_stability_config_resolves_new_and_legacy_names() -> None:
    current = WalkForwardConfig(rank_stability_trials=8)
    legacy = WalkForwardConfig(pbo_trials=8)
    assert current.resolved_rank_stability_trials == 8
    assert legacy.resolved_rank_stability_trials == 8

    with pytest.raises(ValueError, match="disagree"):
        WalkForwardConfig(rank_stability_trials=8, pbo_trials=4)


def test_effective_trial_count_reduces_dsr_deflation() -> None:
    trials = [0.01, 0.03, 0.05, 0.07, 0.09]
    raw = deflated_sharpe(
        trials,
        n_trials=100,
        observed_sr=0.10,
        n_obs=252,
    )
    effective = deflated_sharpe(
        trials,
        n_trials=100,
        effective_n_trials=10.5,
        observed_sr=0.10,
        n_obs=252,
    )
    assert effective > raw

    with pytest.raises(ValueError, match="<= n_trials"):
        deflated_sharpe(trials, n_trials=5, effective_n_trials=6, n_obs=252)


@pytest.mark.parametrize(
    ("method", "label"),
    [
        ("pooled_walk_forward_dsr_style", "Pooled WF DSR-style"),
        ("probabilistic_sharpe_n1", "Probabilistic Sharpe N=1"),
    ],
)
def test_walk_forward_sharpe_method_is_serialized_and_labelled(
    method: str,
    label: str,
) -> None:
    result = WalkForwardResult(
        folds=[],
        config_dict={},
        deflated_sharpe=0.91,
        deflated_sharpe_method=method,
        dsr_raw_completed_trials=1,
        dsr_effective_trials=1.0,
    )

    assert result.to_dict()["deflated_sharpe_method"] == method
    assert label in result.summary()


def test_refined_expected_max_matches_documented_finite_examples() -> None:
    assert _expected_max_sr(1.0, 20) == pytest.approx(1.9007, abs=1e-4)
    assert _expected_max_sr(1.0, 500) == pytest.approx(3.0525, abs=1e-4)


def test_cscv_thresholds_are_not_applied_to_rolling_rank_failure() -> None:
    assert interpret_metrics({"walk_forward_top_k_rank_failure_rate": 0.05}) == []
    assert interpret_metrics({"pbo": 0.05}) == []


def test_pre_oos_percentage_has_honest_name_and_legacy_alias() -> None:
    current = WalkForwardConfig(extra_pre_oos_purge_pct=0.02)
    legacy = WalkForwardConfig(embargo_pct=0.03)
    assert current.resolved_extra_pre_oos_purge_pct == pytest.approx(0.02)
    assert legacy.resolved_extra_pre_oos_purge_pct == pytest.approx(0.03)

    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    kwargs = {
        "is_end": pd.Timestamp("2020-09-30"),
        "oos_start": pd.Timestamp("2020-10-01"),
        "purge_days": 5,
        "trading_dates": dates,
        "is_start": dates[0],
    }
    current_bounds = compute_pre_oos_purge(
        **kwargs,
        extra_pre_oos_purge_pct=0.02,
    )
    with pytest.warns(DeprecationWarning, match="not a post-test embargo"):
        legacy_bounds = compute_purge_embargo(**kwargs, embargo_pct=0.02)
    assert legacy_bounds == current_bounds


def _audit_fold() -> Fold:
    return Fold(
        fold_id=3,
        scheme="rolling",
        train_start=pd.Timestamp("2020-01-02"),
        train_end=pd.Timestamp("2020-06-30"),
        effective_is_end=pd.Timestamp("2020-06-25"),
        oos_start=pd.Timestamp("2020-07-01"),
        oos_end=pd.Timestamp("2020-09-30"),
        purge_start=pd.Timestamp("2020-06-26"),
        purge_end=pd.Timestamp("2020-06-30"),
    )


def test_per_fold_factory_receives_explicit_string_bounds() -> None:
    captured: dict[str, object] = {}

    def factory(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    nav = pd.Series([100_000.0], index=[pd.Timestamp("2020-01-02")])
    runner = FoldRunner(
        _audit_fold(),
        SimpleNamespace(net_asset_value=nav),
        backtester_factory=factory,
    )
    runner._build_fold_backtester({"lookback": 20})

    assert captured["train_start"] == "2020-01-02"
    assert captured["train_end"] == "2020-06-25"
    assert captured["oos_start"] == "2020-07-01"
    assert captured["oos_end"] == "2020-09-30"


def test_per_fold_refit_fails_closed_when_factory_returns_full_history() -> None:
    fold = _audit_fold()
    bounded_nav = pd.Series(
        [100_000.0, 101_000.0],
        index=[fold.train_start, fold.oos_end],
    )
    leaked_nav = pd.Series(
        [99_000.0, 102_000.0],
        index=[pd.Timestamp("2019-12-31"), pd.Timestamp("2020-10-01")],
    )
    runner = FoldRunner(
        fold,
        SimpleNamespace(net_asset_value=bounded_nav),
    )

    runner._validate_fold_portfolio_bounds(SimpleNamespace(net_asset_value=bounded_nav))
    with pytest.raises(ValueError, match="escaped requested date bounds"):
        runner._validate_fold_portfolio_bounds(SimpleNamespace(net_asset_value=leaked_nav))

    with pytest.raises(ValueError, match="NaT timestamps"):
        runner._validate_fold_portfolio_bounds(
            SimpleNamespace(net_asset_value=pd.Series([100_000.0], index=[pd.NaT]))
        )
