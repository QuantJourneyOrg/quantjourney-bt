# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Native Zipline implementations for the five comparison strategies."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from zipline import run_algorithm
from zipline.api import order_target_percent, set_commission, set_slippage, symbol
from zipline.finance.commission import NoCommission
from zipline.finance.slippage import NoSlippage

NATIVE_DIR = Path(__file__).resolve().parent
COMPARE_DIR = NATIVE_DIR.parent
ZIPLINE_DIR = COMPARE_DIR / "zipline"
for path in (ZIPLINE_DIR, COMPARE_DIR, NATIVE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(NATIVE_DIR) in sys.path:
    sys.path.remove(str(NATIVE_DIR))
sys.path.insert(0, str(NATIVE_DIR))

from common import (  # noqa: E402
    CASH_BUFFER,
    DUAL_MOMENTUM_TOP_N,
    EVALUATION_END,
    EVALUATION_START,
    INITIAL_CAPITAL,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP_RECENT,
    MOMENTUM_TOP_N,
    PRIOR_SESSION,
    RSI_ENTRY,
    RSI_EXIT,
    RSI_PERIOD,
    SMA_FAST,
    SMA_POSITION_CAP,
    SMA_SLOW,
    STRATEGIES,
    TICKERS,
    VOLATILITY_LOOKBACK,
    VOLATILITY_TARGET,
    WARMUP_START,
    write_native_result,
)
from custom_bundle import BUNDLE_NAME, ensure_ingested  # noqa: E402

_zipline_common_spec = importlib.util.spec_from_file_location(
    "qj_zipline_common", ZIPLINE_DIR / "common.py"
)
_zipline_common = importlib.util.module_from_spec(_zipline_common_spec)
assert _zipline_common_spec.loader is not None
_zipline_common_spec.loader.exec_module(_zipline_common)
MINIMAL_METRICS_SET = _zipline_common.MINIMAL_METRICS_SET
register_minimal_metrics = _zipline_common.register_minimal_metrics

ACTIVE_STRATEGY = ""
DECISIONS: dict[pd.Timestamp, dict[str, float]] = {}


def _session_date(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def initialize(context):
    context.assets = {ticker: symbol(ticker) for ticker in TICKERS}
    context.last_month = None
    context.previous_close = {ticker: None for ticker in TICKERS}
    context.rsi_gains = {ticker: [] for ticker in TICKERS}
    context.rsi_losses = {ticker: [] for ticker in TICKERS}
    context.rsi_avg_gain = {ticker: None for ticker in TICKERS}
    context.rsi_avg_loss = {ticker: None for ticker in TICKERS}
    context.rsi_state = {ticker: False for ticker in TICKERS}
    set_commission(NoCommission())
    set_slippage(NoSlippage())


def _update_native_rsi(context, data) -> dict[str, float | None]:
    values = {}
    for ticker, asset in context.assets.items():
        price = float(data.current(asset, "price"))
        previous = context.previous_close[ticker]
        context.previous_close[ticker] = price
        if previous is None:
            values[ticker] = None
            continue
        delta = price - previous
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if context.rsi_avg_gain[ticker] is None:
            context.rsi_gains[ticker].append(gain)
            context.rsi_losses[ticker].append(loss)
            if len(context.rsi_gains[ticker]) < RSI_PERIOD:
                values[ticker] = None
                continue
            context.rsi_avg_gain[ticker] = float(np.mean(context.rsi_gains[ticker]))
            context.rsi_avg_loss[ticker] = float(np.mean(context.rsi_losses[ticker]))
        else:
            context.rsi_avg_gain[ticker] = (
                context.rsi_avg_gain[ticker] * (RSI_PERIOD - 1) + gain
            ) / RSI_PERIOD
            context.rsi_avg_loss[ticker] = (
                context.rsi_avg_loss[ticker] * (RSI_PERIOD - 1) + loss
            ) / RSI_PERIOD
        avg_gain = context.rsi_avg_gain[ticker]
        avg_loss = context.rsi_avg_loss[ticker]
        if avg_loss == 0.0:
            values[ticker] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            values[ticker] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return values


def _sma_targets(context, data) -> dict[str, float]:
    active = []
    for ticker, asset in context.assets.items():
        history = data.history(asset, "price", SMA_SLOW, "1d")
        if len(history) < SMA_SLOW:
            continue
        fast = float(history.iloc[-SMA_FAST:].mean())
        slow = float(history.mean())
        if fast > slow:
            active.append(ticker)
    weight = min(1.0 / len(active), SMA_POSITION_CAP) if active else 0.0
    return {ticker: weight if ticker in active else 0.0 for ticker in TICKERS}


def _rsi_targets(context, rsi: dict[str, float | None]) -> dict[str, float]:
    for ticker, value in rsi.items():
        if value is None:
            continue
        if not context.rsi_state[ticker] and value < RSI_ENTRY:
            context.rsi_state[ticker] = True
        elif context.rsi_state[ticker] and value > RSI_EXIT:
            context.rsi_state[ticker] = False
    active = [ticker for ticker, enabled in context.rsi_state.items() if enabled]
    weight = 1.0 / len(active) if active else 0.0
    return {ticker: weight if ticker in active else 0.0 for ticker in TICKERS}


def _momentum_targets(context, data) -> dict[str, float]:
    momentum = {}
    for ticker, asset in context.assets.items():
        history = data.history(asset, "price", MOMENTUM_LOOKBACK + 1, "1d")
        if len(history) < MOMENTUM_LOOKBACK + 1:
            continue
        momentum[ticker] = (
            float(history.iloc[-1 - MOMENTUM_SKIP_RECENT]) / float(history.iloc[0]) - 1.0
        )
    if len(momentum) < MOMENTUM_TOP_N:
        return {ticker: 0.0 for ticker in TICKERS}
    top = sorted(momentum, key=momentum.get, reverse=True)[:MOMENTUM_TOP_N]
    raw_weight = 1.0 / MOMENTUM_TOP_N
    portfolio_returns = np.zeros(VOLATILITY_LOOKBACK, dtype=float)
    for ticker in top:
        history = data.history(context.assets[ticker], "price", VOLATILITY_LOOKBACK + 2, "1d")
        prior = history.iloc[:-1]
        returns = prior.pct_change().dropna().iloc[-VOLATILITY_LOOKBACK:]
        if len(returns) != VOLATILITY_LOOKBACK:
            return {name: 0.0 for name in TICKERS}
        portfolio_returns += returns.to_numpy(dtype=float) * raw_weight
    realised = float(np.std(portfolio_returns, ddof=1) * np.sqrt(252.0))
    scale = min(VOLATILITY_TARGET / realised, 1.0) if realised > 0.01 else 1.0
    return {ticker: raw_weight * scale if ticker in top else 0.0 for ticker in TICKERS}


def _dual_targets(context, data) -> dict[str, float]:
    momentum = {}
    for ticker, asset in context.assets.items():
        history = data.history(asset, "price", MOMENTUM_LOOKBACK + 1, "1d")
        if len(history) < MOMENTUM_LOOKBACK + 1:
            continue
        momentum[ticker] = float(history.iloc[-1]) / float(history.iloc[0]) - 1.0
    if len(momentum) < DUAL_MOMENTUM_TOP_N:
        return {ticker: 0.0 for ticker in TICKERS}
    top = sorted(momentum, key=momentum.get, reverse=True)[:DUAL_MOMENTUM_TOP_N]
    survivors = [ticker for ticker in top if momentum[ticker] > 0.0]
    weight = 1.0 / len(survivors) if survivors else 0.0
    return {ticker: weight if ticker in survivors else 0.0 for ticker in TICKERS}


def _submit_targets(context, data, targets: dict[str, float]) -> None:
    nav = float(context.portfolio.portfolio_value)
    rows = []
    for ticker, asset in context.assets.items():
        price = float(data.current(asset, "price"))
        current_value = float(context.portfolio.positions[asset].amount) * price
        target = float(targets[ticker]) * (1.0 - CASH_BUFFER)
        rows.append((target * nav - current_value, asset, target))
    rows.sort(key=lambda item: item[0])
    for _delta, asset, target in rows:
        if data.can_trade(asset):
            order_target_percent(asset, target)


def handle_data(context, data):
    date = _session_date(data.current_dt)
    rsi = _update_native_rsi(context, data)
    warm_rsi_targets = _rsi_targets(context, rsi) if ACTIVE_STRATEGY == "02_rsi_reversion" else None
    month = (date.year, date.month)
    first_session = month != context.last_month
    context.last_month = month
    decision_start = (
        PRIOR_SESSION
        if ACTIVE_STRATEGY in {"01_sma_crossover", "02_rsi_reversion"}
        else EVALUATION_START
    )
    if date < decision_start or date > EVALUATION_END:
        return

    if ACTIVE_STRATEGY == "01_sma_crossover":
        targets = _sma_targets(context, data)
    elif ACTIVE_STRATEGY == "02_rsi_reversion":
        targets = warm_rsi_targets
    elif first_session and ACTIVE_STRATEGY == "03_monthly_rebalance":
        targets = {ticker: 1.0 / len(TICKERS) for ticker in TICKERS}
    elif first_session and ACTIVE_STRATEGY == "04_momentum_voltarget":
        targets = _momentum_targets(context, data)
    elif first_session and ACTIVE_STRATEGY == "05_dual_momentum":
        targets = _dual_targets(context, data)
    else:
        return

    DECISIONS[date] = dict(targets)
    _submit_targets(context, data, targets)


def run(strategy: str) -> dict:
    ensure_ingested()
    register_minimal_metrics()
    global ACTIVE_STRATEGY, DECISIONS
    ACTIVE_STRATEGY = strategy
    DECISIONS = {}
    started = time.perf_counter()
    performance = run_algorithm(
        start=WARMUP_START,
        end=EVALUATION_END,
        initialize=initialize,
        handle_data=handle_data,
        capital_base=INITIAL_CAPITAL,
        bundle=BUNDLE_NAME,
        metrics_set=MINIMAL_METRICS_SET,
    )
    core_seconds = time.perf_counter() - started
    nav = performance["portfolio_value"].copy()
    decisions = pd.DataFrame.from_dict(DECISIONS, orient="index").reindex(columns=TICKERS)
    result = write_native_result(
        engine="zipline",
        strategy=strategy,
        nav=nav,
        core_seconds=core_seconds,
        decision_weights=decisions,
        extra={
            "engine_mode": "native Zipline history/event loop + target-percent orders",
            "share_model": "whole shares",
            "decision_count": len(decisions),
        },
    )
    print(f"Zipline native {strategy}: core={core_seconds:.4f}s NAV={result['final_nav']:,.6f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=STRATEGIES)
    run(parser.parse_args().strategy)
