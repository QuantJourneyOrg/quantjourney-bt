# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Native Backtrader implementations for the five comparison strategies."""

from __future__ import annotations

import argparse
import time

import backtrader as bt
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
    TICKERS,
    VOLATILITY_LOOKBACK,
    VOLATILITY_TARGET,
    load_ohlcv,
    write_native_result,
)


class NativeStrategy(bt.Strategy):
    params = dict(strategy_key=None)

    def __init__(self):
        self.values: dict[pd.Timestamp, float] = {}
        self.decisions: dict[pd.Timestamp, dict[str, float]] = {}
        self.rsi_state = {ticker: False for ticker in TICKERS}
        self.fast = {}
        self.slow = {}
        self.rsi = {}
        if self.p.strategy_key == "01_sma_crossover":
            for data in self.datas:
                self.fast[data._name] = bt.indicators.SMA(data.close, period=SMA_FAST)
                self.slow[data._name] = bt.indicators.SMA(data.close, period=SMA_SLOW)
        elif self.p.strategy_key == "02_rsi_reversion":
            for data in self.datas:
                self.rsi[data._name] = bt.indicators.RSI(data.close, period=RSI_PERIOD)

    def _date(self, ago: int = 0) -> pd.Timestamp:
        return pd.Timestamp(self.datas[0].datetime.date(ago))

    def _is_previous_first_session(self) -> bool:
        if len(self) < 3:
            return False
        previous = self._date(-1)
        before = self._date(-2)
        return (previous.year, previous.month) != (before.year, before.month)

    def _daily_targets(self) -> dict[str, float]:
        strategy = self.p.strategy_key
        if strategy == "01_sma_crossover":
            active = [
                data._name
                for data in self.datas
                if self.fast[data._name][-1] > self.slow[data._name][-1]
            ]
            weight = min(1.0 / len(active), SMA_POSITION_CAP) if active else 0.0
            return {ticker: weight if ticker in active else 0.0 for ticker in TICKERS}

        for data in self.datas:
            value = float(self.rsi[data._name][-1])
            if not self.rsi_state[data._name] and value < RSI_ENTRY:
                self.rsi_state[data._name] = True
            elif self.rsi_state[data._name] and value > RSI_EXIT:
                self.rsi_state[data._name] = False
        active = [ticker for ticker, enabled in self.rsi_state.items() if enabled]
        weight = 1.0 / len(active) if active else 0.0
        return {ticker: weight if ticker in active else 0.0 for ticker in TICKERS}

    def _monthly_targets(self) -> dict[str, float]:
        strategy = self.p.strategy_key
        if strategy == "03_monthly_rebalance":
            return {ticker: 1.0 / len(TICKERS) for ticker in TICKERS}

        if len(self) < MOMENTUM_LOOKBACK + 2:
            return {ticker: 0.0 for ticker in TICKERS}

        if strategy == "04_momentum_voltarget":
            momentum = {
                data._name: (
                    float(data.close[-1 - MOMENTUM_SKIP_RECENT])
                    / float(data.close[-1 - MOMENTUM_LOOKBACK])
                    - 1.0
                )
                for data in self.datas
            }
            top = [
                name
                for name, _value in sorted(
                    momentum.items(), key=lambda item: item[1], reverse=True
                )[:MOMENTUM_TOP_N]
            ]
            raw_weight = 1.0 / MOMENTUM_TOP_N
            portfolio_returns = np.zeros(VOLATILITY_LOOKBACK, dtype=float)
            for data in self.datas:
                if data._name not in top:
                    continue
                returns = []
                # Match the published contract: estimate volatility from the
                # 63 completed returns immediately before the decision bar.
                for offset in range(VOLATILITY_LOOKBACK + 1, 1, -1):
                    previous = float(data.close[-1 - offset])
                    current = float(data.close[-offset])
                    returns.append(current / previous - 1.0)
                portfolio_returns += np.asarray(returns) * raw_weight
            realised = float(np.std(portfolio_returns, ddof=1) * np.sqrt(252.0))
            scale = min(VOLATILITY_TARGET / realised, 1.0) if realised > 0.01 else 1.0
            return {ticker: raw_weight * scale if ticker in top else 0.0 for ticker in TICKERS}

        momentum = {
            data._name: float(data.close[-1]) / float(data.close[-1 - MOMENTUM_LOOKBACK]) - 1.0
            for data in self.datas
        }
        top = sorted(momentum, key=momentum.get, reverse=True)[:DUAL_MOMENTUM_TOP_N]
        survivors = [ticker for ticker in top if momentum[ticker] > 0.0]
        weight = 1.0 / len(survivors) if survivors else 0.0
        return {ticker: weight if ticker in survivors else 0.0 for ticker in TICKERS}

    def _execute(self, targets: dict[str, float]) -> None:
        nav = float(self.broker.getvalue())
        orders = []
        for data in self.datas:
            target = float(targets[data._name]) * (1.0 - CASH_BUFFER)
            current_value = float(self.getposition(data).size * data.close[0])
            orders.append((target * nav - current_value, data, target))
        orders.sort(key=lambda item: item[0])
        for _delta, data, target in orders:
            self.order_target_percent(data=data, target=target)

    def next(self):
        current_date = self._date()
        if EVALUATION_START <= current_date <= EVALUATION_END:
            self.values[current_date] = float(self.broker.getvalue())
        if len(self) < 2:
            return

        decision_date = self._date(-1)
        strategy = self.p.strategy_key
        warm_rsi_targets = self._daily_targets() if strategy == "02_rsi_reversion" else None
        if current_date < EVALUATION_START or current_date > EVALUATION_END:
            return

        if strategy in {"01_sma_crossover", "02_rsi_reversion"}:
            targets = warm_rsi_targets or self._daily_targets()
        else:
            if not self._is_previous_first_session():
                return
            targets = self._monthly_targets()
        self.decisions[decision_date] = dict(targets)
        self._execute(targets)


def run(strategy: str) -> dict:
    raw = load_ohlcv()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_coc(True)
    for ticker in TICKERS:
        frame = raw[ticker].copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        cerebro.adddata(bt.feeds.PandasData(dataname=frame, name=ticker))
    cerebro.addstrategy(NativeStrategy, strategy_key=strategy)

    started = time.perf_counter()
    native = cerebro.run()[0]
    core_seconds = time.perf_counter() - started
    nav = pd.Series(native.values, dtype=float).sort_index()
    decisions = pd.DataFrame.from_dict(native.decisions, orient="index").reindex(columns=TICKERS)
    result = write_native_result(
        engine="backtrader",
        strategy=strategy,
        nav=nav,
        core_seconds=core_seconds,
        decision_weights=decisions,
        extra={
            "engine_mode": "native indicators/event loop + order_target_percent",
            "share_model": "whole shares",
            "decision_count": len(decisions),
        },
    )
    print(f"Backtrader native {strategy}: core={core_seconds:.4f}s NAV={result['final_nav']:,.6f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=STRATEGIES)
    run(parser.parse_args().strategy)
