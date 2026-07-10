# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example WF 01 - Rolling Walk-Forward Validation
===============================================

Mode: weights + walk-forward.
Idea: run a normal SMA(50/200) trend strategy, then validate its temporal
robustness with a ROLLING walk-forward (fixed-length train and test windows
that slide forward), with purge/embargo gaps.
Universe: five large US technology stocks.

What this teaches: in-sample performance is not out-of-sample performance.
Walk-forward slices the NAV into successive train/test windows and reports how
much the strategy decays out of sample. Note: validating a single fixed-param
strategy's NAV is "slice diagnostics" — for genuine per-fold parameter refit,
pass a ``backtester_factory`` to ``WalkForwardEngine``.

Usage:
    ./strategy.sh example_wf_01_rolling_walkforward

    QJ_WF_MODE=per_fold_refit ./strategy.sh example_wf_01_rolling_walkforward
"""

import asyncio
import os

import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.walkforward import WalkForwardConfig, WalkForwardEngine
from backtester.walkforward.statistics.interpretation import interpret_metrics


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _wf_mode() -> str:
    raw = os.environ.get("QJ_WF_MODE", "").strip().lower()
    if raw in {"per_fold_refit", "refit", "true_oos"} or _env_flag("QJ_WF_REFIT"):
        return "per_fold_refit"
    return "slice_diagnostics"


class SMATrendForWF(Backtester):
    """SMA(50/200) long/cash trend, used as the walk-forward subject."""

    def _compute_signals(self) -> pd.DataFrame:
        fast = self.instruments_data.get_feature("SMA_50_close")
        slow = self.instruments_data.get_feature("SMA_200_close")
        valid = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(valid, 0.0)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


def _build_strategy(
    *,
    strategy_name: str = "ExampleWF01_RollingWalkForward",
    backtest_period: dict | None = None,
) -> SMATrendForWF:
    save_packet = _env_flag("QJ_WF_REPORT_PACKET")
    return SMATrendForWF(
        **_credentials(),
        strategy_name=strategy_name,
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period=backtest_period or {"start": "2012-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}},
        ],
        benchmark_symbol="^GSPC",
        benchmark_name="S&P 500 Index",
        show_text_reports=False,
        save_text_reports=save_packet,
        save_portfolio_plots=save_packet,
    )


async def main() -> None:
    mode = _wf_mode()
    strategy = _build_strategy()
    await strategy.run_strategy()

    # Walk-forward over the produced NAV.
    config = WalkForwardConfig(
        scheme="rolling",
        train_months=24,
        test_months=6,
        step_months=6,
        purge_days=5,
        embargo_pct=0.01,
    )
    engine_kwargs = {}
    if mode == "per_fold_refit":

        def factory(*, fold, train_start, train_end, oos_start, oos_end, **_) -> SMATrendForWF:
            return _build_strategy(
                strategy_name=f"ExampleWF01_RollingWalkForward_Fold{fold.fold_id:02d}",
                backtest_period={"start": train_start, "end": oos_end},
            )

        engine_kwargs["backtester_factory"] = factory

    engine = WalkForwardEngine(config=config, initial_capital=100_000, **engine_kwargs)
    if mode == "per_fold_refit":
        result = await engine.run_async(strategy.portfolio_data)
    else:
        result = engine.run(strategy.portfolio_data)

    print(result.summary())

    verdicts = interpret_metrics(
        {
            "overfit_ratio": result.overfit_ratio,
            "efficiency": result.efficiency,
            "sharpe_decay": result.sharpe_decay,
            # Context keys (no verdicts of their own): gate the lights so a
            # losing strategy or a tiny fold count never renders green.
            "composite_sharpe": result.oos_sharpe,
            "n_folds": result.n_folds,
        }
    )
    if result.mode == "slice_diagnostics":
        print(
            "\nNOTE: slice_diagnostics mode — the metrics above are IN-SAMPLE"
            "\nslices of ONE full-period run, not out-of-sample evidence."
            "\nSet QJ_WF_MODE=per_fold_refit for honest OOS validation."
        )
        print("\nWalk-forward traffic lights (IN-SAMPLE — indicative only):")
    else:
        print("\nWalk-forward traffic lights:")
    for v in verdicts:
        print(f"  {v}")


if __name__ == "__main__":
    asyncio.run(main())
