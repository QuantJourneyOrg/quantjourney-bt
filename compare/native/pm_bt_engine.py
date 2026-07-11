# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Native pmorissette/bt implementations for the comparison strategies."""

from __future__ import annotations

import argparse
import time

import bt
import numpy as np
import pandas as pd
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
    VOLATILITY_LOOKBACK,
    VOLATILITY_TARGET,
    decision_flags,
    load_close,
    write_native_result,
)


def _blank(close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def _sma(close: pd.DataFrame) -> pd.DataFrame:
    fast = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    slow = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    active = (fast > slow) & fast.notna() & slow.notna()
    count = active.sum(axis=1).replace(0, np.nan)
    return active.astype(float).div(count, axis=0).fillna(0.0).clip(upper=SMA_POSITION_CAP)


def _wilder_rsi(close: pd.DataFrame) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _blank(close).replace(0.0, np.nan)
    avg_loss = _blank(close).replace(0.0, np.nan)
    avg_gain.iloc[RSI_PERIOD] = gain.iloc[1 : RSI_PERIOD + 1].mean(axis=0)
    avg_loss.iloc[RSI_PERIOD] = loss.iloc[1 : RSI_PERIOD + 1].mean(axis=0)
    for i in range(RSI_PERIOD + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (RSI_PERIOD - 1) + gain.iloc[i]) / RSI_PERIOD
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (RSI_PERIOD - 1) + loss.iloc[i]) / RSI_PERIOD
    rs = avg_gain.divide(avg_loss.replace(0.0, np.nan))
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    return rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)


def _rsi(close: pd.DataFrame) -> pd.DataFrame:
    rsi = _wilder_rsi(close)
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


def _monthly_equal(close: pd.DataFrame) -> pd.DataFrame:
    weights = _blank(close)
    flags = decision_flags("03_monthly_rebalance", close.index)
    current = pd.Series(0.0, index=close.columns)
    for i in range(len(close)):
        if flags.iloc[i]:
            current[:] = 1.0 / len(close.columns)
        weights.iloc[i] = current
    return weights


def _momentum_vol(close: pd.DataFrame) -> pd.DataFrame:
    weights = _blank(close)
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
                    portfolio_returns = (returns.iloc[i - VOLATILITY_LOOKBACK : i] * raw).sum(
                        axis=1
                    )
                    realised = float(portfolio_returns.std(ddof=1) * np.sqrt(252.0))
                    scale = min(VOLATILITY_TARGET / realised, 1.0) if realised > 0.01 else 1.0
                    current = raw * scale
        weights.iloc[i] = current
    return weights


def _dual_momentum(close: pd.DataFrame) -> pd.DataFrame:
    weights = _blank(close)
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
    "01_sma_crossover": _sma,
    "02_rsi_reversion": _rsi,
    "03_monthly_rebalance": _monthly_equal,
    "04_momentum_voltarget": _momentum_vol,
    "05_dual_momentum": _dual_momentum,
}


def run(strategy: str) -> dict:
    close = load_close()
    started = time.perf_counter()
    decisions = BUILDERS[strategy](close)
    flags = decision_flags(strategy, close.index)
    execution_flags = flags.shift(1, fill_value=False).astype(bool)
    execution_weights = decisions.shift(1).fillna(0.0) * (1.0 - CASH_BUFFER)
    close_eval = close.loc[EVALUATION_START:EVALUATION_END]
    execution_dates = list(execution_flags.loc[lambda values: values].index)

    native_strategy = bt.Strategy(
        f"native_{strategy}",
        [
            bt.algos.RunOnDate(*execution_dates),
            bt.algos.WeighTarget(execution_weights),
            bt.algos.Rebalance(),
        ],
    )
    backtest = bt.Backtest(
        native_strategy,
        close_eval,
        initial_capital=INITIAL_CAPITAL,
        integer_positions=False,
        progress_bar=False,
    )
    result_set = bt.run(backtest, progress_bar=False)
    core_seconds = time.perf_counter() - started
    nav = result_set.prices.iloc[:, 0] * (INITIAL_CAPITAL / 100.0)
    result = write_native_result(
        engine="pm_bt",
        strategy=strategy,
        nav=nav,
        core_seconds=core_seconds,
        decision_weights=decisions,
        extra={
            "engine_mode": "independent pandas strategy + WeighTarget/Rebalance",
            "share_model": "fractional",
            "decision_count": int(flags.loc[EVALUATION_START:].sum()),
            "execution_count": len(execution_dates),
        },
    )
    print(
        f"pmorissette/bt native {strategy}: core={core_seconds:.4f}s NAV={result['final_nav']:,.6f}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=STRATEGIES)
    run(parser.parse_args().strategy)
