# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Native QuantJourney implementations for the five comparison strategies.

Runs entirely on the published ``quantjourney-bt`` package (>= 0.10.1): the
local OHLCV panel is converted into a ``/bt/prepare``-compatible payload, and
the strategies are expressed through the engine's normal ``_compute_signals``
/ ``_compute_weights`` hooks with an explicit rebalance calendar.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_DIR = Path(__file__).resolve().parent
if str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

from common import (  # noqa: E402
    CASH_BUFFER,
    DAILY_STRATEGIES,
    DUAL_MOMENTUM_TOP_N,
    EVALUATION_END,
    EVALUATION_START,
    INITIAL_CAPITAL,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP_RECENT,
    MOMENTUM_TOP_N,
    PRIOR_SESSION,
    RESULTS_DIR,
    RSI_ENTRY,
    RSI_EXIT,
    SMA_POSITION_CAP,
    STRATEGIES,
    TICKERS,
    VOLATILITY_LOOKBACK,
    VOLATILITY_TARGET,
    WARMUP_START,
    decision_flags,
    execution_dates,
    load_ohlcv,
    write_native_result,
)

from backtester import Backtester  # noqa: E402
from backtester.portfolio.rebalance import RebalancePolicy  # noqa: E402
from backtester.portfolio.weight_cost import FixedBpsWeightCostModel  # noqa: E402
from backtester.sample_data import _frame_payload, _series_payload  # noqa: E402


def _payload_from_panel(instruments: list[str]) -> dict:
    """Convert the shared OHLCV parquet into a /bt/prepare-compatible payload."""
    raw = load_ohlcv()
    dates = raw.index

    price_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    return_columns: list[pd.Series] = []

    for symbol in instruments:
        close = raw[(symbol, "close")].astype(float)
        ret = close.pct_change(fill_method=None).fillna(0.0)
        return_columns.append(ret)
        frame = pd.DataFrame(
            {
                (symbol, "open"): raw[(symbol, "open")].astype(float),
                (symbol, "high"): raw[(symbol, "high")].astype(float),
                (symbol, "low"): raw[(symbol, "low")].astype(float),
                (symbol, "close"): close,
                # The panel is split/dividend adjusted already.
                (symbol, "adj_close"): close,
                (symbol, "volume"): raw[(symbol, "volume")].astype(float),
            },
            index=dates,
        )
        price_frames.append(frame)

        synthetic_nav = (1.0 + ret).cumprod()
        drawdown = synthetic_nav / synthetic_nav.cummax() - 1.0
        metric_frames.append(
            pd.DataFrame(
                {
                    (symbol, "returns"): ret,
                    (symbol, "volatility"): ret.rolling(20, min_periods=2).std().fillna(0.0),
                    (symbol, "daily_pnl"): ret,
                    (symbol, "transaction_costs"): 0.0,
                    (symbol, "net_asset_value"): synthetic_nav,
                    (symbol, "gross_asset_value"): synthetic_nav,
                    (symbol, "daily_net_return"): ret,
                    (symbol, "drawdown"): drawdown,
                },
                index=dates,
            )
        )
        parameter_frames.append(
            pd.DataFrame(
                {
                    (symbol, "exchange"): 0.0,
                    (symbol, "units"): 0.0,
                    (symbol, "eligibility"): 1.0,
                    (symbol, "active"): 1.0,
                    (symbol, "forecasts"): 0.0,
                    (symbol, "is_trading_day"): 1.0,
                    (symbol, "day_type"): 1.0,
                },
                index=dates,
            )
        )

    prices_df = pd.concat(price_frames, axis=1)
    prices_df.columns = pd.MultiIndex.from_tuples(prices_df.columns, names=["instrument", "field"])
    metrics_df = pd.concat(metric_frames, axis=1)
    metrics_df.columns = pd.MultiIndex.from_tuples(
        metrics_df.columns, names=["instrument", "field"]
    )
    parameters_df = pd.concat(parameter_frames, axis=1)
    parameters_df.columns = pd.MultiIndex.from_tuples(
        parameters_df.columns, names=["instrument", "field"]
    )
    portfolio_nav = (
        1.0 + pd.concat(return_columns, axis=1).mean(axis=1)
    ).cumprod() * INITIAL_CAPITAL

    return {
        "session_id": "native-local-session",
        "dataset_id": "native-local-dataset",
        "instrument_names": list(instruments),
        "prices": _frame_payload(prices_df),
        "metrics": _frame_payload(metrics_df),
        "parameters": _frame_payload(parameters_df),
        "nav": _series_payload(portfolio_nav),
        "summary": {
            "source": "local-panel",
            "instruments": len(instruments),
            "dates": len(dates),
            "start": dates[0].date().isoformat(),
            "end": dates[-1].date().isoformat(),
        },
    }


class LocalPanelBacktester(Backtester):
    """Backtester fed from the shared benchmark panel instead of the API."""

    def __init__(self, **kwargs):
        self._benchmark_cash_buffer = float(kwargs.pop("benchmark_cash_buffer", 0.0))
        kwargs.setdefault("email", "local")
        kwargs.setdefault("password", "local")
        kwargs.setdefault("weight_cost_model", FixedBpsWeightCostModel(total_bps=0.0))
        super().__init__(**kwargs)
        self.TRANSACTION_COST_BPS = 0.0

    async def _fetch_market_data(self) -> None:
        self._api_response = _payload_from_panel(list(self.instruments))
        self.session_id = self._api_response["session_id"]
        self.dataset_id = self._api_response["dataset_id"]
        self._validate_data_completeness_response()

    async def _process_market_data(self) -> None:
        await super()._process_market_data()
        self.portfolio_data.cash_buffer = self._benchmark_cash_buffer


class NativeQJStrategy(LocalPanelBacktester):
    """Calculate strategy decisions through QJ's normal strategy hooks."""

    def __init__(self, *, native_strategy: str, **kwargs):
        self.native_strategy = native_strategy
        self.native_decisions = pd.DataFrame()
        super().__init__(**kwargs)

    def _monthly_flags(self) -> pd.Series:
        return decision_flags(self.native_strategy, self.data.close.index)

    def _compute_signals(self) -> pd.DataFrame:
        close = self.data.close
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        strategy = self.native_strategy

        if strategy == "01_sma_crossover":
            fast = self.data.SMA_50_close
            slow = self.data.SMA_200_close
            signals = ((fast > slow) & fast.notna() & slow.notna()).astype(float)
        elif strategy == "02_rsi_reversion":
            rsi = self.data.RSI_14_close
            state = pd.DataFrame(False, index=close.index, columns=close.columns)
            for i in range(1, len(close)):
                previous = state.iloc[i - 1].copy()
                current = previous.copy()
                values = rsi.iloc[i]
                current[(~previous) & (values < RSI_ENTRY)] = True
                current[previous & (values > RSI_EXIT)] = False
                current[values.isna()] = False
                state.iloc[i] = current
            signals = state.astype(float)
        elif strategy == "03_monthly_rebalance":
            signals = close.notna().astype(float)
        elif strategy == "04_momentum_voltarget":
            flags = self._monthly_flags()
            current = pd.Series(0.0, index=close.columns)
            for i in range(len(close)):
                if flags.iloc[i]:
                    current[:] = 0.0
                    if i >= MOMENTUM_LOOKBACK:
                        momentum = (
                            close.iloc[i - MOMENTUM_SKIP_RECENT] / close.iloc[i - MOMENTUM_LOOKBACK]
                            - 1.0
                        )
                        top = momentum.dropna().nlargest(MOMENTUM_TOP_N).index
                        current.loc[top] = 1.0
                signals.iloc[i] = current
        elif strategy == "05_dual_momentum":
            flags = self._monthly_flags()
            current = pd.Series(0.0, index=close.columns)
            for i in range(len(close)):
                if flags.iloc[i]:
                    current[:] = 0.0
                    if i >= MOMENTUM_LOOKBACK:
                        momentum = close.iloc[i] / close.iloc[i - MOMENTUM_LOOKBACK] - 1.0
                        top = momentum.dropna().nlargest(DUAL_MOMENTUM_TOP_N)
                        current.loc[top.loc[lambda values: values > 0.0].index] = 1.0
                signals.iloc[i] = current
        else:
            raise ValueError(f"Unknown native QJ strategy: {strategy}")

        decision_start = PRIOR_SESSION if strategy in DAILY_STRATEGIES else EVALUATION_START
        return signals.where(close.index.to_series().ge(decision_start), 0.0, axis=0)

    def _compute_weights(self) -> pd.DataFrame:
        close = self.data.close
        strategy = self.native_strategy
        active = self.signals > 0.0
        count = active.sum(axis=1).replace(0, np.nan)
        weights = active.astype(float).div(count, axis=0).fillna(0.0)

        if strategy == "01_sma_crossover":
            weights = weights.clip(upper=SMA_POSITION_CAP)
        elif strategy == "04_momentum_voltarget":
            returns = close.pct_change()
            flags = self._monthly_flags()
            scaled = pd.DataFrame(0.0, index=close.index, columns=close.columns)
            current = pd.Series(0.0, index=close.columns)
            for i in range(len(close)):
                if flags.iloc[i]:
                    current[:] = 0.0
                    selected = active.iloc[i]
                    if int(selected.sum()) == MOMENTUM_TOP_N:
                        raw = selected.astype(float) / MOMENTUM_TOP_N
                        portfolio_returns = (returns.iloc[i - VOLATILITY_LOOKBACK : i] * raw).sum(
                            axis=1
                        )
                        realised = float(portfolio_returns.std(ddof=1) * np.sqrt(252.0))
                        scale = min(VOLATILITY_TARGET / realised, 1.0) if realised > 0.01 else 1.0
                        current = raw * scale
                scaled.iloc[i] = current.reindex(scaled.columns).to_numpy()
            weights = scaled

        decision_start = PRIOR_SESSION if strategy in DAILY_STRATEGIES else EVALUATION_START
        weights.loc[weights.index < decision_start] = 0.0
        self.native_decisions = weights.copy()
        return weights


def _indicator_config(strategy: str) -> list[dict]:
    if strategy == "01_sma_crossover":
        return [{"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}}]
    if strategy == "02_rsi_reversion":
        return [{"function": "RSI", "price_cols": ["close"], "params": {"periods": [14]}}]
    return []


async def _run(strategy: str) -> dict:
    calendar = load_ohlcv().index
    policy = RebalancePolicy(
        frequency=None,
        calendar_dates=execution_dates(strategy, calendar),
    )
    native = NativeQJStrategy(
        native_strategy=strategy,
        strategy_name=f"native_compare_{strategy}",
        initial_capital=INITIAL_CAPITAL,
        instruments=TICKERS,
        backtest_period={"start": str(WARMUP_START.date()), "end": str(EVALUATION_END.date())},
        benchmark_cash_buffer=CASH_BUFFER,
        rebalance_policy=policy,
        max_position_size=1.0,
        indicators_config=_indicator_config(strategy),
        show_text_reports=False,
        save_text_reports=False,
        save_portfolio_plots=False,
        show_portfolio_plots=False,
        save_pdf_report=False,
        benchmark_symbol="^GSPC",
        skip_analysis=True,
        lite_init=True,
        reports_directory=str(RESULTS_DIR.parent / "native_reports"),
    )

    wall_started = time.perf_counter()
    await native.run_strategy()
    wall_seconds = time.perf_counter() - wall_started
    timings = getattr(native, "_timings", {})
    data_seconds = float(timings.get("data_processing_seconds") or 0.0)
    calculation_seconds = float(timings.get("calculation_seconds") or 0.0)
    core_seconds = data_seconds + calculation_seconds
    if core_seconds <= 0.0:
        core_seconds = wall_seconds

    result = write_native_result(
        engine="qj",
        strategy=strategy,
        nav=native.portfolio_data.net_asset_value,
        core_seconds=core_seconds,
        wall_seconds=wall_seconds,
        decision_weights=native.native_decisions,
        extra={
            "engine_mode": "native QJ indicators, strategy hooks, rebalance and accounting",
            "share_model": "fractional",
            "data_processing_seconds": timings.get("data_processing_seconds"),
            "indicator_seconds": timings.get("indicator_seconds"),
            "calculation_seconds": timings.get("calculation_seconds"),
        },
    )
    print(f"QJ native {strategy}: core={core_seconds:.4f}s NAV={result['final_nav']:,.6f}")
    return result


def run(strategy: str) -> dict:
    return asyncio.run(_run(strategy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=STRATEGIES)
    run(parser.parse_args().strategy)
