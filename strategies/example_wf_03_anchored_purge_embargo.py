# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example WF 03 - Anchored Walk-Forward With Purge & Embargo
==========================================================

Mode: weights + walk-forward.
Idea: validate a weekly RSI mean-reversion strategy with an ANCHORED
walk-forward, emphasising the purge and embargo gaps that prevent information
leakage between the training and test windows.
Universe: five liquid mega-cap stocks.

What this teaches: naive walk-forward can still leak — the last training bars
sit right next to the first test bars, and an indicator's warm-up or an
overlapping label can bleed test information backward. PURGE drops the training
bars closest to the test window; EMBARGO adds a buffer after it. This example
uses a generous purge/embargo so you can see their effect on the OOS metrics.

Usage:
    ./strategy.sh example_wf_03_anchored_purge_embargo

    QJ_WF_MODE=per_fold_refit ./strategy.sh example_wf_03_anchored_purge_embargo
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


class RSIReversionForWF(Backtester):
    """Weekly RSI(14) mean-reversion long/cash, walk-forward subject."""

    def _compute_signals(self) -> pd.DataFrame:
        rsi = self.instruments_data.get_feature("RSI_14_close")
        signals = pd.DataFrame(0.0, index=rsi.index, columns=rsi.columns)
        holding = pd.Series(False, index=rsi.columns)
        for date, row in rsi.iterrows():
            for inst, value in row.items():
                if pd.isna(value):
                    holding[inst] = False
                elif not holding[inst] and value < 35:
                    holding[inst] = True
                elif holding[inst] and value > 60:
                    holding[inst] = False
            signals.loc[date] = holding.astype(float)
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


def _build_strategy(
    *,
    strategy_name: str = "ExampleWF03_AnchoredPurgeEmbargo",
    backtest_period: dict | None = None,
) -> RSIReversionForWF:
    save_packet = _env_flag("QJ_WF_REPORT_PACKET")
    return RSIReversionForWF(
        **_credentials(),
        strategy_name=strategy_name,
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        backtest_period=backtest_period or {"start": "2012-01-01", "end": "2025-01-01"},
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency="W", weekday=4),
        indicators_config=[
            {"function": "RSI", "price_cols": ["close"], "params": {"periods": [14]}},
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

    config = WalkForwardConfig(
        scheme="anchored",
        train_months=24,
        test_months=6,
        step_months=6,
        purge_days=10,       # drop the 10 training days nearest the test window
        embargo_pct=0.02,    # 2% buffer after the test window
    )
    engine_kwargs = {}
    if mode == "per_fold_refit":
        def factory(*, fold, train_start, train_end, oos_start, oos_end, **_) -> RSIReversionForWF:
            return _build_strategy(
                strategy_name=f"ExampleWF03_AnchoredPurgeEmbargo_Fold{fold.fold_id:02d}",
                backtest_period={"start": train_start, "end": oos_end},
            )

        engine_kwargs["backtester_factory"] = factory

    engine = WalkForwardEngine(config=config, initial_capital=100_000, **engine_kwargs)
    if mode == "per_fold_refit":
        result = await engine.run_async(strategy.portfolio_data)
    else:
        result = engine.run(strategy.portfolio_data)

    print(result.summary())

    verdicts = interpret_metrics({
        "overfit_ratio": result.overfit_ratio,
        "efficiency": result.efficiency,
        "sharpe_decay": result.sharpe_decay,
    })
    print("\nWalk-forward traffic lights:")
    for v in verdicts:
        print(f"  {v}")


if __name__ == "__main__":
    asyncio.run(main())
