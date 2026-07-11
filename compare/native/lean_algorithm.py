# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""LEAN-native strategy implementations for the cross-engine benchmark."""

# ruff: noqa: F403, F405

import csv

import numpy as np
from AlgorithmImports import *

ACTIVE_STRATEGY = "__NATIVE_STRATEGY__"


class NativeStrategyAlgorithm(QCAlgorithm):
    strategy_key = ""
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    evaluation_start = "2016-01-04"
    prior_session = "2015-12-31"
    cash_buffer = 0.001

    def initialize(self):
        self.strategy_key = ACTIVE_STRATEGY
        self.set_start_date(2015, 1, 2)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)
        self.symbols = {}
        self.fast = {}
        self.slow = {}
        self.rsi_indicators = {}
        self.rsi_state = {ticker: False for ticker in self.tickers}
        self.close_history = {ticker: [] for ticker in self.tickers}
        self.last_month = None
        self.pending_targets = None

        for ticker in self.tickers:
            security = self.add_equity(ticker, Resolution.DAILY)
            security.set_fee_model(ConstantFeeModel(0))
            security.set_slippage_model(NullSlippageModel())
            self.symbols[ticker] = security.symbol
            if self.strategy_key == "01_sma_crossover":
                self.fast[ticker] = self.sma(security.symbol, 50, Resolution.DAILY)
                self.slow[ticker] = self.sma(security.symbol, 200, Resolution.DAILY)
            elif self.strategy_key == "02_rsi_reversion":
                self.rsi_indicators[ticker] = self.rsi(
                    security.symbol,
                    14,
                    MovingAverageType.WILDERS,
                    Resolution.DAILY,
                )

        self.settings.minimum_order_margin_portfolio_percentage = 0
        calendar_symbol = self.symbols[self.tickers[0]]
        self.schedule.on(
            self.date_rules.every_day(calendar_symbol),
            self.time_rules.before_market_close(calendar_symbol, 20),
            self.execute_pending_targets,
        )

        self.nav_file = open("/Results/native_nav.csv", "w", buffering=1)
        self.nav_file.write("date,nav\n")
        self.decision_file = open("/Results/native_decisions.csv", "w", buffering=1)
        self.decision_writer = csv.writer(self.decision_file)
        self.decision_writer.writerow(["date"] + self.tickers)

    def _sma_targets(self):
        if not all(self.slow[ticker].is_ready for ticker in self.tickers):
            return {ticker: 0.0 for ticker in self.tickers}
        active = [
            ticker
            for ticker in self.tickers
            if self.fast[ticker].current.value > self.slow[ticker].current.value
        ]
        weight = min(1.0 / len(active), 0.25) if active else 0.0
        return {ticker: weight if ticker in active else 0.0 for ticker in self.tickers}

    def _rsi_targets(self):
        for ticker in self.tickers:
            indicator = self.rsi_indicators[ticker]
            if not indicator.is_ready:
                continue
            value = float(indicator.current.value)
            if not self.rsi_state[ticker] and value < 30.0:
                self.rsi_state[ticker] = True
            elif self.rsi_state[ticker] and value > 70.0:
                self.rsi_state[ticker] = False
        active = [ticker for ticker in self.tickers if self.rsi_state[ticker]]
        weight = 1.0 / len(active) if active else 0.0
        return {ticker: weight if ticker in active else 0.0 for ticker in self.tickers}

    def _momentum_vol_targets(self):
        momentum = {}
        for ticker in self.tickers:
            history = self.close_history[ticker]
            if len(history) < 253:
                continue
            momentum[ticker] = history[-22] / history[-253] - 1.0
        if len(momentum) < 3:
            return {ticker: 0.0 for ticker in self.tickers}
        top = sorted(momentum, key=momentum.get, reverse=True)[:3]
        raw_weight = 1.0 / 3.0
        portfolio_returns = np.zeros(63)
        for ticker in top:
            prior = np.asarray(self.close_history[ticker][:-1][-64:], dtype=float)
            if len(prior) != 64:
                return {name: 0.0 for name in self.tickers}
            portfolio_returns += np.diff(prior) / prior[:-1] * raw_weight
        realised = float(np.std(portfolio_returns, ddof=1) * np.sqrt(252.0))
        scale = min(0.15 / realised, 1.0) if realised > 0.01 else 1.0
        return {
            ticker: raw_weight * scale if ticker in top else 0.0 for ticker in self.tickers
        }

    def _dual_targets(self):
        momentum = {}
        for ticker in self.tickers:
            history = self.close_history[ticker]
            if len(history) < 253:
                continue
            momentum[ticker] = history[-1] / history[-253] - 1.0
        if len(momentum) < 2:
            return {ticker: 0.0 for ticker in self.tickers}
        top = sorted(momentum, key=momentum.get, reverse=True)[:2]
        survivors = [ticker for ticker in top if momentum[ticker] > 0.0]
        weight = 1.0 / len(survivors) if survivors else 0.0
        return {ticker: weight if ticker in survivors else 0.0 for ticker in self.tickers}

    def _build_targets(self, first_session):
        if self.strategy_key == "01_sma_crossover":
            return self._sma_targets()
        if self.strategy_key == "02_rsi_reversion":
            return self._rsi_targets()
        if not first_session:
            return None
        if self.strategy_key == "03_monthly_rebalance":
            return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
        if self.strategy_key == "04_momentum_voltarget":
            return self._momentum_vol_targets()
        if self.strategy_key == "05_dual_momentum":
            return self._dual_targets()
        raise ValueError("Unknown native strategy: " + self.strategy_key)

    def on_data(self, data):
        date = self.time.strftime("%Y-%m-%d")
        for ticker in self.tickers:
            price = float(self.securities[self.symbols[ticker]].price)
            if price > 0.0:
                self.close_history[ticker].append(price)

        if date >= self.evaluation_start:
            self.nav_file.write(f"{date},{float(self.portfolio.total_portfolio_value):.12f}\n")

        month = (self.time.year, self.time.month)
        first_session = month != self.last_month
        self.last_month = month
        warm_rsi_targets = (
            self._rsi_targets() if self.strategy_key == "02_rsi_reversion" else None
        )
        decision_start = (
            self.prior_session
            if self.strategy_key in {"01_sma_crossover", "02_rsi_reversion"}
            else self.evaluation_start
        )
        if date < decision_start:
            return

        targets = warm_rsi_targets
        if targets is None:
            targets = self._build_targets(first_session)
        if targets is None:
            return
        self.pending_targets = dict(targets)
        self.decision_writer.writerow([date] + [targets[ticker] for ticker in self.tickers])

    def execute_pending_targets(self):
        if self.pending_targets is None:
            return
        nav = float(self.portfolio.total_portfolio_value)
        orders = []
        for ticker in self.tickers:
            symbol = self.symbols[ticker]
            price = float(self.securities[symbol].price)
            if price <= 0.0:
                continue
            target = float(self.pending_targets[ticker]) * (1.0 - self.cash_buffer)
            desired = int(target * nav / price)
            current = int(self.portfolio[symbol].quantity)
            delta = desired - current
            if delta:
                orders.append((delta, symbol))
        orders.sort(key=lambda item: item[0])
        for delta, symbol in orders:
            self.market_on_close_order(symbol, delta)
        self.pending_targets = None

    def on_end_of_algorithm(self):
        self.nav_file.flush()
        self.nav_file.close()
        self.decision_file.flush()
        self.decision_file.close()


class SMACrossoverAlgorithm(NativeStrategyAlgorithm):
    pass


class RSIMeanReversionAlgorithm(NativeStrategyAlgorithm):
    pass


class MonthlyEqualWeightAlgorithm(NativeStrategyAlgorithm):
    pass


class MomentumVolTargetAlgorithm(NativeStrategyAlgorithm):
    pass


class DualMomentumAlgorithm(NativeStrategyAlgorithm):
    pass
