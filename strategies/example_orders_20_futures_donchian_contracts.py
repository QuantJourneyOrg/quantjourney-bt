# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Orders 20 - Futures Donchian Contracts
==============================================

Mode: orders.
Order type: MARKET.
Idea: a diversified 55-day Donchian breakout across index, energy, and metal
futures. Orders are whole contracts sized by ATR risk, gross notional, and the
reference initial margin carried in ContractSpec.

Yahoo ``=F`` data are provider-managed continuous series. The order engine
honors contract multipliers for sizing and PnL, but it does not yet select
dated contracts, control rolls, round fills to ticks, post variation margin,
or enforce current broker buying power. Cash accounting therefore uses full
notional, while the margin figure here is only an additional sizing cap.

Usage:
    ./strategy.sh example_orders_20_futures_donchian_contracts
"""

import asyncio
import math
import os

import numpy as np
import pandas as pd

from backtester import Backtester
from backtester.execution.commission import PerShareCommission
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


def size_futures_contracts(
    *,
    nav: float,
    price: float,
    atr: float,
    multiplier: float,
    margin: float,
    risk_fraction: float = 0.005,
    notional_cap: float = 0.25,
    margin_cap: float = 0.15,
) -> int:
    """Size whole contracts using independent risk and exposure ceilings."""
    values = (nav, price, atr, multiplier)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        return 0
    limits = [
        math.floor(nav * risk_fraction / (atr * multiplier)),
        math.floor(nav * notional_cap / (price * multiplier)),
    ]
    if np.isfinite(margin) and margin > 0.0:
        limits.append(math.floor(nav * margin_cap / margin))
    return max(0, min(limits))


class FuturesDonchianContracts(Backtester):
    """Donchian breakout using API-provided continuous-futures specs."""

    BREAKOUT_WINDOW = 55
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
        if not isinstance(index, (int, np.integer)) or index < self.BREAKOUT_WINDOW:
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
            prior_high = high[instrument].iloc[index - self.BREAKOUT_WINDOW : index].max()
            prior_low = low[instrument].iloc[index - self.BREAKOUT_WINDOW : index].min()
            current_atr = atr.loc[date, instrument]
            if pd.isna(prior_high) or pd.isna(prior_low) or pd.isna(current_atr):
                continue

            if current_price > prior_high:
                direction = 1
            elif current_price < prior_low:
                direction = -1
            else:
                direction = int(np.sign(current))

            spec = self._contract_spec(instrument)
            contracts = size_futures_contracts(
                nav=float(nav),
                price=current_price,
                atr=float(current_atr),
                multiplier=float(spec.multiplier),
                margin=float(spec.margin),
            )
            target = float(direction * contracts)
            delta = target - current
            if abs(delta) > 1e-12:
                self.order_market(instrument, delta, tag="futures_donchian_target")


async def main() -> None:
    strategy = FuturesDonchianContracts(
        **_credentials(),
        strategy_name="ExampleOrders20_FuturesDonchianContracts",
        strategy_type="Long / Short Futures",
        initial_capital=2_000_000,
        instruments=["MES=F", "MNQ=F", "CL=F", "GC=F"],
        backtest_period={"start": "2010-01-01", "end": "2026-01-01"},
        source="yfinance",
        execution_mode="orders",
        indicators_config=[],
        slippage_model=FixedBpsSlippage(bps=1.0),
        # Quantity-based fee proxy: $2.50 per contract, with no percentage cap.
        commission_scheme=PerShareCommission(
            cost_per_share=2.50,
            min_per_order=2.50,
            max_pct=0.0,
        ),
        benchmark_symbol="SPY",
        benchmark_name="S&P 500 ETF",
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
