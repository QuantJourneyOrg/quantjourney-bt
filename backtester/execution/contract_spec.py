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

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AssetClass(str, Enum):  # noqa: UP042 - preserve pre-3.11 Enum string semantics
    EQUITY = "equity"
    FUTURE = "future"
    FX = "fx"
    CRYPTO = "crypto"
    OPTION = "option"


class UnsupportedCurrencyConversionError(ValueError):
    """Raised when an FX amount cannot be expressed in portfolio currency."""


def _currency_code(value: object, *, field_name: str) -> str:
    """Return a normalized non-empty currency code or fail explicitly."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip().upper()
    if not normalized or not normalized.isalnum():
        raise ValueError(
            f"{field_name} must be a non-empty alphanumeric currency code, got {value!r}"
        )
    return normalized


@dataclass(frozen=True)
class ContractSpec:
    """Immutable specification for a tradeable instrument.

    Execution support status
    ────────────────────────
    Orders mode and execution-aware target-weight mode honor ``multiplier``,
    ``lot_size``, ``inverse``, quantity rounding and per-instrument margin.
    ``tick_size`` is available to explicit order authors through
    ``round_price()`` but generic fills are not silently tick-rounded.

    ``pip_size`` remains descriptive. Multi-currency cash conversion is not
    implemented, so FX contracts fail closed unless ``quote_currency`` equals
    the portfolio settlement currency. Fast weight mode uses contract specs
    only for reporting proxies; use ``weight_execution='orders'`` when
    quantities, fills and collateral must be contract-aware.
    """

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

    # Tradable quantity increment, independent of economic contract size.
    # Examples: 1 share, 1 futures contract, 0.01 standard FX lot.
    quantity_step: float | None = 1.0
    min_quantity: float = 0.0

    # Pip size for FX pairs (e.g., 0.0001 for EURUSD, 0.01 for USDJPY)
    pip_size: float = 0.0001

    # Currency of the instrument quote (for multi-currency PnL)
    quote_currency: str = "USD"

    # Descriptive metadata supplied by the market-data contract.
    base_currency: str | None = None
    exchange: str | None = None
    calendar: str | None = None
    instrument_type: str | None = None
    provider_symbol: str | None = None
    continuous: bool = False

    # Inverse contract (crypto perps quoted in USD, settled in coin)
    inverse: bool = False

    def __post_init__(self) -> None:
        """Validate economic inputs before they can affect sizing or PnL."""
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.asset_class, AssetClass):
            raise TypeError(
                "asset_class must be an AssetClass value; use "
                "contract_spec_from_mapping() to parse external metadata"
            )

        positive_fields = ("multiplier", "tick_size", "lot_size", "pip_size")
        non_negative_fields = ("margin", "min_quantity")
        for name in positive_fields + non_negative_fields:
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool):
                raise TypeError(f"{name} must be numeric, not bool")
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be numeric, got {raw_value!r}") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite, got {raw_value!r}")
            if name in positive_fields and numeric <= 0.0:
                raise ValueError(f"{name} must be greater than zero, got {numeric}")
            if name in non_negative_fields and numeric < 0.0:
                raise ValueError(f"{name} must be non-negative, got {numeric}")
            object.__setattr__(self, name, numeric)

        if self.quantity_step is not None:
            if isinstance(self.quantity_step, bool):
                raise TypeError("quantity_step must be numeric or None, not bool")
            try:
                quantity_step = float(self.quantity_step)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"quantity_step must be numeric or None, got {self.quantity_step!r}"
                ) from exc
            if not math.isfinite(quantity_step) or quantity_step <= 0.0:
                raise ValueError(
                    "quantity_step must be finite and greater than zero, "
                    f"got {self.quantity_step!r}"
                )
            object.__setattr__(self, "quantity_step", quantity_step)

        if not isinstance(self.continuous, bool):
            raise TypeError("continuous must be bool")
        if not isinstance(self.inverse, bool):
            raise TypeError("inverse must be bool")

        object.__setattr__(
            self,
            "quote_currency",
            _currency_code(self.quote_currency, field_name="quote_currency"),
        )
        if self.base_currency is not None:
            object.__setattr__(
                self,
                "base_currency",
                _currency_code(self.base_currency, field_name="base_currency"),
            )

    def validate_settlement_currency(self, settlement_currency: str) -> None:
        """Fail if FX quote amounts require an unavailable currency conversion."""
        if self.asset_class != AssetClass.FX:
            return
        settlement = _currency_code(settlement_currency, field_name="settlement_currency")
        if self.quote_currency != settlement:
            raise UnsupportedCurrencyConversionError(
                f"FX contract {self.symbol!r} produces notional and PnL in "
                f"{self.quote_currency}, but the portfolio settles in {settlement}. "
                "Currency conversion is not implemented; use a matching portfolio "
                "currency or an instrument quoted in the portfolio currency."
            )

    # ── Factory methods ──

    @classmethod
    def equity(cls, symbol: str, **kwargs) -> ContractSpec:
        defaults = dict(
            asset_class=AssetClass.EQUITY,
            multiplier=1.0,
            tick_size=0.01,
            lot_size=1.0,
            quantity_step=1.0,
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
            quantity_step=1.0,
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
            quantity_step=0.01,
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
            quantity_step=None,
            margin=0.0,
        )
        defaults.update(kwargs)
        return cls(symbol=symbol, **defaults)

    # ── PnL helpers ──

    def notional(self, quantity: float, price: float) -> float:
        """Absolute notional exposure of a position."""
        if self.inverse:
            return abs(quantity) * self.multiplier / price if price != 0 else 0.0
        return abs(quantity) * abs(price) * self.multiplier * self.lot_size

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
        """Round toward zero to the executable quantity increment."""
        numeric = float(quantity)
        if not math.isfinite(numeric):
            raise ValueError("quantity must be finite")
        step = self.quantity_step
        if step is None:
            rounded = numeric
        else:
            numeric_step = float(step)
            if not math.isfinite(numeric_step) or numeric_step <= 0.0:
                raise ValueError("quantity_step must be finite and positive")
            tolerance = math.copysign(1e-12, numeric) if numeric else 0.0
            rounded = math.trunc(numeric / numeric_step + tolerance) * numeric_step
            rounded = round(rounded, 12)
        if abs(rounded) + 1e-12 < float(self.min_quantity):
            return 0.0
        return float(rounded)


# ── Registry ─────────────────────────────────────────────────────────

# Pre-built specs for common instruments
COMMON_SPECS: dict[str, ContractSpec] = {
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
    "NG": ContractSpec.future("NG", multiplier=10_000, tick_size=0.001, margin=7_500),
    "GC": ContractSpec.future("GC", multiplier=100, tick_size=0.10, margin=10_000),
    "SI": ContractSpec.future("SI", multiplier=5_000, tick_size=0.005, margin=14_000),
    "HG": ContractSpec.future("HG", multiplier=25_000, tick_size=0.0005, margin=6_000),
    # Grains and livestock (prices are quoted in exchange price points)
    "ZC": ContractSpec.future("ZC", multiplier=50, tick_size=0.25, margin=2_500),
    "ZS": ContractSpec.future("ZS", multiplier=50, tick_size=0.25, margin=3_500),
    "ZW": ContractSpec.future("ZW", multiplier=50, tick_size=0.25, margin=3_000),
    "LE": ContractSpec.future("LE", multiplier=400, tick_size=0.025, margin=2_500),
    "HE": ContractSpec.future("HE", multiplier=400, tick_size=0.025, margin=2_500),
    # FX majors
    "EURUSD": ContractSpec.fx("EURUSD", pip_size=0.0001, base_currency="EUR"),
    "GBPUSD": ContractSpec.fx("GBPUSD", pip_size=0.0001, base_currency="GBP"),
    "USDJPY": ContractSpec.fx("USDJPY", pip_size=0.01, base_currency="USD", quote_currency="JPY"),
    "AUDUSD": ContractSpec.fx("AUDUSD", pip_size=0.0001, base_currency="AUD"),
    "NZDUSD": ContractSpec.fx("NZDUSD", pip_size=0.0001, base_currency="NZD"),
    "USDCAD": ContractSpec.fx("USDCAD", pip_size=0.0001, base_currency="USD", quote_currency="CAD"),
    "USDCHF": ContractSpec.fx("USDCHF", pip_size=0.0001, base_currency="USD", quote_currency="CHF"),
    # Crypto
    "BTCUSD": ContractSpec.crypto("BTCUSD"),
    "ETHUSD": ContractSpec.crypto("ETHUSD"),
}


def contract_spec_from_mapping(symbol: str, values: Mapping[str, Any]) -> ContractSpec:
    """Build and strictly validate a spec from the ``/bt/prepare`` contract."""

    def _require_fields(asset_name: str, *field_names: str) -> None:
        missing = [name for name in field_names if name not in values or values[name] is None]
        if missing:
            raise ValueError(
                f"Invalid contract spec for {symbol!r}: {asset_name} metadata "
                f"requires explicit field(s): {', '.join(missing)}"
            )

    def _symbol_fx_pair() -> tuple[str, str] | None:
        """Return an FX pair only when the external symbol is unambiguous."""
        canonical = str(symbol).strip().upper()
        if canonical.endswith("=X"):
            canonical = canonical[:-2]
        for separator in ("/", "_", "-"):
            parts = canonical.split(separator)
            if len(parts) == 2 and all(len(part) == 3 and part.isalpha() for part in parts):
                return parts[0], parts[1]
        if len(canonical) == 6 and canonical.isalpha():
            return canonical[:3], canonical[3:]
        return None

    def _number(name: str, default: float) -> float:
        if name not in values:
            return float(default)
        value = values[name]
        if value is None or isinstance(value, bool):
            raise ValueError(
                f"Invalid contract spec for {symbol!r}: {name} must be a finite "
                f"number, got {value!r}"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid contract spec for {symbol!r}: {name} must be numeric, got {value!r}"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"Invalid contract spec for {symbol!r}: {name} must be finite, got {value!r}"
            )
        return numeric

    def _boolean(name: str, default: bool) -> bool:
        if name not in values:
            return default
        value = values[name]
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise ValueError(
            f"Invalid contract spec for {symbol!r}: {name} must be a boolean, got {value!r}"
        )

    _require_fields("instrument", "asset_class")
    asset_class_value = values["asset_class"]
    if isinstance(asset_class_value, AssetClass):
        raw_asset_class = asset_class_value.value
    elif isinstance(asset_class_value, str) and asset_class_value.strip():
        raw_asset_class = asset_class_value.strip().lower()
    else:
        raise ValueError(
            f"Invalid contract spec for {symbol!r}: asset_class must be a "
            f"non-empty string, got {asset_class_value!r}"
        )
    asset_aliases = {
        "futures": "future",
        "future_continuous": "future",
        "spot_fx": "fx",
        "currency": "fx",
    }
    raw_asset_class = asset_aliases.get(raw_asset_class, raw_asset_class)
    try:
        asset_class = AssetClass(raw_asset_class)
    except ValueError as exc:
        supported = ", ".join(member.value for member in AssetClass)
        raise ValueError(
            f"Invalid contract spec for {symbol!r}: unsupported asset_class "
            f"{asset_class_value!r}; expected one of: {supported}"
        ) from exc

    if asset_class == AssetClass.FX:
        _require_fields("FX", "base_currency", "quote_currency", "lot_size")
    elif asset_class == AssetClass.FUTURE:
        _require_fields("future", "multiplier")

    quote_currency = (
        "USD"
        if "quote_currency" not in values
        else _currency_code(values["quote_currency"], field_name="quote_currency")
    )
    common = {
        "quote_currency": quote_currency,
        "base_currency": (
            _currency_code(values["base_currency"], field_name="base_currency")
            if values.get("base_currency") is not None
            else None
        ),
        "exchange": str(values["exchange"]) if values.get("exchange") else None,
        "calendar": str(values["calendar"]) if values.get("calendar") else None,
        "instrument_type": (
            str(values["instrument_type"]) if values.get("instrument_type") else None
        ),
        "provider_symbol": str(values.get("provider_symbol") or symbol),
        "continuous": _boolean("continuous", False),
        "inverse": _boolean("inverse", False),
    }
    if asset_class == AssetClass.FX:
        base_currency = common["base_currency"]
        if base_currency == quote_currency:
            raise ValueError(
                f"Invalid contract spec for {symbol!r}: base_currency and "
                f"quote_currency must differ for FX, got {base_currency}"
            )
        symbol_pair = _symbol_fx_pair()
        if symbol_pair is not None and symbol_pair != (
            base_currency,
            quote_currency,
        ):
            raise ValueError(
                f"Invalid contract spec for {symbol!r}: FX symbol implies "
                f"{symbol_pair[0]}/{symbol_pair[1]}, but metadata declares "
                f"{base_currency}/{quote_currency}"
            )
    if values.get("quantity_step") is not None:
        common["quantity_step"] = _number("quantity_step", 1.0)
    if values.get("min_quantity") is not None:
        common["min_quantity"] = _number("min_quantity", 0.0)
    tick_size = _number("tick_size", 0.01)
    margin = _number("margin", 0.0)
    multiplier = _number("multiplier", 1.0)
    lot_size = _number("lot_size", 1.0)

    if asset_class == AssetClass.FUTURE:
        return ContractSpec.future(
            symbol,
            multiplier=multiplier,
            tick_size=tick_size,
            margin=margin,
            lot_size=lot_size,
            **common,
        )
    if asset_class == AssetClass.FX:
        quote = common["quote_currency"]
        pip_size = _number("pip_size", 0.01 if quote == "JPY" else 0.0001)
        return ContractSpec.fx(
            symbol,
            lot_size=lot_size,
            pip_size=pip_size,
            tick_size=_number("tick_size", pip_size),
            margin=margin,
            multiplier=multiplier,
            **common,
        )
    if asset_class == AssetClass.CRYPTO:
        return ContractSpec.crypto(
            symbol,
            multiplier=multiplier,
            tick_size=tick_size,
            margin=margin,
            lot_size=lot_size,
            **common,
        )
    return ContractSpec(
        symbol=symbol,
        asset_class=asset_class,
        multiplier=multiplier,
        tick_size=tick_size,
        margin=margin,
        lot_size=lot_size,
        pip_size=_number("pip_size", 0.0001),
        **common,
    )


def get_contract_spec(symbol: str) -> ContractSpec:
    """
    Look up a ContractSpec by symbol.

    Falls back to default equity spec if not in the registry.
    """
    key = str(symbol).strip().upper()
    canonical = key[:-2] if key.endswith(("=F", "=X")) else key
    return COMMON_SPECS.get(key, COMMON_SPECS.get(canonical, ContractSpec.equity(key)))
