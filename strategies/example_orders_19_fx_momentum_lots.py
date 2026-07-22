# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 19 - FX Momentum with Standard Lots
==================================================

Mode: orders.
Order type: MARKET.
Idea: trade four USD-quoted spot-FX pairs in the direction of six-month
momentum. Position size is an integer number of standard lots, constrained by
ATR risk and a per-pair notional cap.
Universe: EURUSD, GBPUSD, AUDUSD and NZDUSD provider spot-FX proxies.

The ContractSpec is supplied automatically by ``/bt/prepare``: one unit means
one 100,000-base-currency lot. Current accounting does not enforce FX margin,
overnight swap/financing, broker leverage, or bid/ask quotes. The universe is
limited to XXX/USD pairs so PnL is already denominated in the base portfolio
currency.

Usage:
    ./strategy.sh example_orders_19_fx_momentum_lots
"""

import asyncio
import math
import os

import numpy as np
import pandas as pd

from backtester import Backtester
from backtester.execution.commission import FixedBpsCommission
from backtester.execution.slippage import FixedBpsSlippage


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


def average_true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    previous_close = close.shift(1)
    components = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
        keys=["range", "high_gap", "low_gap"],
    )
    true_range = components.T.groupby(level=1, sort=False).max().T
    return true_range.rolling(window, min_periods=window).mean()


def size_standard_lots(
    *,
    nav: float,
    price: float,
    atr: float,
    multiplier: float,
    lot_size: float,
    risk_fraction: float = 0.005,
    notional_cap: float = 0.20,
) -> int:
    """Size whole FX lots by stop-distance risk and gross notional."""
    values = (nav, price, atr, multiplier, lot_size)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        return 0
    risk_per_lot = atr * multiplier * lot_size
    notional_per_lot = price * multiplier * lot_size
    by_risk = math.floor(nav * risk_fraction / risk_per_lot)
    by_notional = math.floor(nav * notional_cap / notional_per_lot)
    return max(0, min(by_risk, by_notional))


class FXMomentumLots(Backtester):
    """Momentum strategy using API-provided spot-FX lot specifications."""

    LOOKBACK = 126
    ATR_WINDOW = 20

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._atr_cache = None

    def _has_pending(self, instrument: str) -> bool:
        return any(
            order.instrument == instrument and order.is_active
            for order in self.fill_engine.pending_orders
        )

    def _compute_orders(self, date, bars, current_positions, nav) -> None:
        close = self.instruments_data.get_feature("adj_close")
        if date not in close.index:
            return
        index = close.index.get_loc(date)
        if not isinstance(index, (int, np.integer)) or index < self.LOOKBACK:
            return

        high = self.instruments_data.get_feature("high")
        low = self.instruments_data.get_feature("low")
        if self._atr_cache is None or not self._atr_cache.index.equals(close.index):
            self._atr_cache = average_true_range(high, low, close, self.ATR_WINDOW)
        atr = self._atr_cache

        for instrument in self.instruments:
            if self._has_pending(instrument):
                continue
            current = float(current_positions.get(instrument, 0.0))
            current_price = float(bars[instrument].close)
            past_price = close[instrument].iloc[index - self.LOOKBACK]
            current_atr = atr.loc[date, instrument]
            if pd.isna(past_price) or past_price <= 0.0 or pd.isna(current_atr):
                continue

            direction = 1 if current_price > past_price else -1
            spec = self._contract_spec(instrument)
            lots = size_standard_lots(
                nav=float(nav),
                price=current_price,
                atr=float(current_atr),
                multiplier=float(spec.multiplier),
                lot_size=float(spec.lot_size),
            )
            target = float(direction * lots)
            delta = target - current
            if abs(delta) > 1e-12:
                self.order_market(instrument, delta, tag="fx_momentum_target")


async def main() -> None:
    strategy = FXMomentumLots(
        **_credentials(),
        strategy_name="ExampleOrders19_FXMomentumLots",
        strategy_type="Long / Short FX",
        initial_capital=1_000_000,
        instruments=["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"],
        backtest_period={"start": "2010-01-04", "end": "2026-01-01"},
        benchmark_symbol="DX-Y.NYB",
        benchmark_name="US Dollar Index",
        source="yfinance",
        execution_mode="orders",
        indicators_config=[],
        slippage_model=FixedBpsSlippage(bps=0.5),
        commission_scheme=FixedBpsCommission(bps=0.1),
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
