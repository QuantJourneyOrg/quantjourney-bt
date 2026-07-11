# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Native VectorBT implementations for the five comparison strategies."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import vectorbt as vbt
from common import (
    CASH_BUFFER,
    DUAL_MOMENTUM_TOP_N,
    EVALUATION_END,
    EVALUATION_START,
    INITIAL_CAPITAL,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP_RECENT,
    MOMENTUM_TOP_N,
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
    decision_flags,
    load_close,
    write_native_result,
)


def _empty(close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def _sma_weights(close: pd.DataFrame) -> pd.DataFrame:
    fast = vbt.MA.run(close, window=SMA_FAST).ma
    slow = vbt.MA.run(close, window=SMA_SLOW).ma
    fast.columns = close.columns
    slow.columns = close.columns
    active = (fast > slow) & fast.notna() & slow.notna()
    count = active.sum(axis=1).replace(0, np.nan)
    return active.astype(float).div(count, axis=0).fillna(0.0).clip(upper=SMA_POSITION_CAP)


def _rsi_weights(close: pd.DataFrame) -> pd.DataFrame:
    # VectorBT's native RSI uses its own documented rolling/ewm semantics.
    rsi = vbt.RSI.run(close, window=RSI_PERIOD, ewm=False).rsi
    rsi.columns = close.columns
    state = pd.DataFrame(False, index=close.index, columns=close.columns)
    for i in range(1, len(close)):
        previous = state.iloc[i - 1].copy()
        current = previous.copy()
        values = rsi.iloc[i]
        current[(~previous) & (values < RSI_ENTRY)] = True
        current[previous & (values > RSI_EXIT)] = False
        current[values.isna()] = False
        state.iloc[i] = current
    count = state.sum(axis=1).replace(0, np.nan)
    return state.astype(float).div(count, axis=0).fillna(0.0)


def _monthly_equal_weights(close: pd.DataFrame) -> pd.DataFrame:
    weights = _empty(close)
    flags = decision_flags("03_monthly_rebalance", close.index)
    current = pd.Series(0.0, index=close.columns)
    for i in range(len(close)):
        if flags.iloc[i]:
            current[:] = 1.0 / len(close.columns)
        weights.iloc[i] = current
    return weights


def _momentum_vol_weights(close: pd.DataFrame) -> pd.DataFrame:
    weights = _empty(close)
    returns = close.pct_change()
    flags = decision_flags("04_momentum_voltarget", close.index)
    current = pd.Series(0.0, index=close.columns)
    for i in range(len(close)):
        if flags.iloc[i]:
            current[:] = 0.0
            if i >= MOMENTUM_LOOKBACK:
                momentum = (
                    close.iloc[i - MOMENTUM_SKIP_RECENT] / close.iloc[i - MOMENTUM_LOOKBACK] - 1.0
                )
                top = momentum.dropna().nlargest(MOMENTUM_TOP_N).index
                if len(top) == MOMENTUM_TOP_N:
                    raw = pd.Series(0.0, index=close.columns)
                    raw.loc[top] = 1.0 / MOMENTUM_TOP_N
                    recent = returns.iloc[i - VOLATILITY_LOOKBACK : i]
                    portfolio_returns = (recent * raw).sum(axis=1)
                    realised = float(portfolio_returns.std(ddof=1) * np.sqrt(252.0))
                    scale = min(VOLATILITY_TARGET / realised, 1.0) if realised > 0.01 else 1.0
                    current = raw * scale
        weights.iloc[i] = current
    return weights


def _dual_momentum_weights(close: pd.DataFrame) -> pd.DataFrame:
    weights = _empty(close)
    flags = decision_flags("05_dual_momentum", close.index)
    current = pd.Series(0.0, index=close.columns)
    for i in range(len(close)):
        if flags.iloc[i]:
            current[:] = 0.0
            if i >= MOMENTUM_LOOKBACK:
                momentum = close.iloc[i] / close.iloc[i - MOMENTUM_LOOKBACK] - 1.0
                top = momentum.dropna().nlargest(DUAL_MOMENTUM_TOP_N)
                survivors = top.loc[lambda values: values > 0.0].index
                if len(survivors):
                    current.loc[survivors] = 1.0 / len(survivors)
        weights.iloc[i] = current
    return weights


BUILDERS = {
    "01_sma_crossover": _sma_weights,
    "02_rsi_reversion": _rsi_weights,
    "03_monthly_rebalance": _monthly_equal_weights,
    "04_momentum_voltarget": _momentum_vol_weights,
    "05_dual_momentum": _dual_momentum_weights,
}


def run(strategy: str) -> dict:
    close = load_close()
    started = time.perf_counter()
    decisions = BUILDERS[strategy](close)
    flags = decision_flags(strategy, close.index)
    execution_flags = flags.shift(1, fill_value=False).astype(bool)
    execution_weights = decisions.shift(1).fillna(0.0) * (1.0 - CASH_BUFFER)

    close_eval = close.loc[EVALUATION_START:EVALUATION_END]
    size = pd.DataFrame(np.nan, index=close_eval.index, columns=TICKERS)
    active_execution = execution_flags.reindex(close_eval.index).fillna(False)
    aligned_execution_weights = execution_weights.reindex(close_eval.index)
    size.loc[active_execution] = aligned_execution_weights.loc[active_execution]
    portfolio = vbt.Portfolio.from_orders(
        close_eval,
        size=size,
        size_type="targetpercent",
        init_cash=INITIAL_CAPITAL,
        fees=0.0,
        slippage=0.0,
        freq="1D",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
    )
    core_seconds = time.perf_counter() - started
    result = write_native_result(
        engine="vectorbt",
        strategy=strategy,
        nav=portfolio.value(),
        core_seconds=core_seconds,
        decision_weights=decisions,
        extra={
            "engine_mode": "native indicators/arrays + Portfolio.from_orders",
            "share_model": "fractional",
            "decision_count": int(flags.loc[EVALUATION_START:].sum()),
            "execution_count": int(active_execution.sum()),
        },
    )
    print(f"VectorBT native {strategy}: core={core_seconds:.4f}s NAV={result['final_nav']:,.6f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=STRATEGIES)
    run(parser.parse_args().strategy)
