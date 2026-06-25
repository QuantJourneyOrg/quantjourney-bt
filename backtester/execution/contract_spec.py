"""
ContractSpec — Instrument-level specification for multi-asset backtesting.

Defines how each instrument converts price changes to PnL:
  - Equity: PnL = shares × Δprice
  - Futures: PnL = contracts × Δprice × multiplier
  - FX: PnL = lots × Δprice × lot_size (pip-based optional)
  - Crypto: PnL = units × Δprice (same as equity, fractional lots)

Usage:
    spec = ContractSpec.equity("AAPL")
    spec = ContractSpec.future("ES", multiplier=50, tick_size=0.25, margin=15_000)
    spec = ContractSpec.fx("EURUSD", lot_size=100_000, pip_size=0.0001)
    spec = ContractSpec.crypto("BTCUSD")

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class AssetClass(str, Enum):
    EQUITY = "equity"
    FUTURE = "future"
    FX = "fx"
    CRYPTO = "crypto"
    OPTION = "option"


@dataclass(frozen=True)
class ContractSpec:
    """Immutable specification for a tradeable instrument."""

    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY

    # Multiplier: 1 for equity/crypto, contract multiplier for futures
    multiplier: float = 1.0

    # Tick size (minimum price increment)
    tick_size: float = 0.01

    # Margin required per contract (futures) or per lot (FX)
    margin: float = 0.0

    # Lot / contract size: 100_000 for FX, 1 for equity
    lot_size: float = 1.0

    # Pip size for FX pairs (e.g., 0.0001 for EURUSD, 0.01 for USDJPY)
    pip_size: float = 0.0001

    # Currency of the instrument quote (for multi-currency PnL)
    quote_currency: str = "USD"

    # Inverse contract (crypto perps quoted in USD, settled in coin)
    inverse: bool = False

    # ── Factory methods ──

    @classmethod
    def equity(cls, symbol: str, **kwargs) -> ContractSpec:
        defaults = dict(
            asset_class=AssetClass.EQUITY,
            multiplier=1.0,
            tick_size=0.01,
            lot_size=1.0,
            margin=0.0,
        )
        defaults.update(kwargs)
        return cls(symbol=symbol, **defaults)

    @classmethod
    def future(
        cls,
        symbol: str,
        multiplier: float,
        tick_size: float = 0.25,
        margin: float = 0.0,
        **kwargs,
    ) -> ContractSpec:
        defaults = dict(
            asset_class=AssetClass.FUTURE,
            multiplier=multiplier,
            tick_size=tick_size,
            lot_size=1.0,
            margin=margin,
        )
        defaults.update(kwargs)
        return cls(symbol=symbol, **defaults)

    @classmethod
    def fx(
        cls,
        symbol: str,
        lot_size: float = 100_000.0,
        pip_size: float = 0.0001,
        **kwargs,
    ) -> ContractSpec:
        defaults = dict(
            asset_class=AssetClass.FX,
            multiplier=1.0,
            tick_size=pip_size,
            lot_size=lot_size,
            pip_size=pip_size,
            margin=0.0,
        )
        defaults.update(kwargs)
        return cls(symbol=symbol, **defaults)

    @classmethod
    def crypto(cls, symbol: str, **kwargs) -> ContractSpec:
        defaults = dict(
            asset_class=AssetClass.CRYPTO,
            multiplier=1.0,
            tick_size=0.01,
            lot_size=1.0,
            margin=0.0,
        )
        defaults.update(kwargs)
        return cls(symbol=symbol, **defaults)

    # ── PnL helpers ──

    def notional(self, quantity: float, price: float) -> float:
        """Notional value of a position."""
        if self.inverse:
            return abs(quantity) * self.multiplier / price if price != 0 else 0.0
        return abs(quantity) * price * self.multiplier * self.lot_size

    def pnl(self, quantity: float, entry_price: float, exit_price: float) -> float:
        """
        PnL for a round-trip trade.

        Equity : quantity × (exit - entry)
        Futures: quantity × (exit - entry) × multiplier
        FX     : quantity × (exit - entry) × lot_size
        Crypto : quantity × (exit - entry) [or inverse formula]
        """
        delta = exit_price - entry_price
        if self.inverse:
            # Inverse perp: PnL in quote = quantity × multiplier × (1/entry - 1/exit)
            if entry_price == 0 or exit_price == 0:
                return 0.0
            return quantity * self.multiplier * (1.0 / entry_price - 1.0 / exit_price)
        return quantity * delta * self.multiplier * self.lot_size

    def margin_required(self, quantity: float, price: float) -> float:
        """Initial margin required for a position."""
        if self.margin > 0:
            return abs(quantity) * self.margin
        # Equity: full notional
        return self.notional(quantity, price)

    def round_price(self, price: float) -> float:
        """Round price to nearest tick."""
        if self.tick_size <= 0:
            return price
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def round_quantity(self, quantity: float) -> float:
        """Round quantity to nearest lot (integer for equity/futures, fractional for crypto)."""
        if self.asset_class == AssetClass.CRYPTO:
            return quantity  # fractional OK
        return float(int(quantity))  # integer lots for equity/futures/FX


# ── Registry ─────────────────────────────────────────────────────────

# Pre-built specs for common instruments
COMMON_SPECS: Dict[str, ContractSpec] = {
    # US Equity Index Futures
    "ES": ContractSpec.future("ES", multiplier=50, tick_size=0.25, margin=15_840),
    "NQ": ContractSpec.future("NQ", multiplier=20, tick_size=0.25, margin=21_000),
    "YM": ContractSpec.future("YM", multiplier=5, tick_size=1.0, margin=11_000),
    "RTY": ContractSpec.future("RTY", multiplier=50, tick_size=0.10, margin=8_000),
    # Micro futures
    "MES": ContractSpec.future("MES", multiplier=5, tick_size=0.25, margin=1_584),
    "MNQ": ContractSpec.future("MNQ", multiplier=2, tick_size=0.25, margin=2_100),
    # Treasuries
    "ZN": ContractSpec.future("ZN", multiplier=1000, tick_size=1 / 64, margin=2_200),
    "ZB": ContractSpec.future("ZB", multiplier=1000, tick_size=1 / 32, margin=4_400),
    # Commodities
    "CL": ContractSpec.future("CL", multiplier=1000, tick_size=0.01, margin=7_000),
    "GC": ContractSpec.future("GC", multiplier=100, tick_size=0.10, margin=10_000),
    # FX majors
    "EURUSD": ContractSpec.fx("EURUSD", pip_size=0.0001),
    "GBPUSD": ContractSpec.fx("GBPUSD", pip_size=0.0001),
    "USDJPY": ContractSpec.fx("USDJPY", pip_size=0.01),
    "AUDUSD": ContractSpec.fx("AUDUSD", pip_size=0.0001),
    # Crypto
    "BTCUSD": ContractSpec.crypto("BTCUSD"),
    "ETHUSD": ContractSpec.crypto("ETHUSD"),
}


def get_contract_spec(symbol: str) -> ContractSpec:
    """
    Look up a ContractSpec by symbol.

    Falls back to default equity spec if not in the registry.
    """
    return COMMON_SPECS.get(symbol, ContractSpec.equity(symbol))
