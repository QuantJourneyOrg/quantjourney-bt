"""Execution Engine - Order Types, Fill Engine, Slippage, Commission.

Provides institutional-grade order management for backtesting:
  Market, Limit, Stop, StopTrail, OCO, Bracket orders
  with pluggable slippage models and commission schemes.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import importlib
from typing import Any

from backtester.execution.commission import (
    CommissionConfig,
    CommissionScheme,
    FixedBpsCommission,
    PerShareCommission,
    TieredCommission,
    ZeroCommission,
)
from backtester.execution.contract_spec import (
    COMMON_SPECS,
    AssetClass,
    ContractSpec,
    UnsupportedCurrencyConversionError,
    contract_spec_from_mapping,
    get_contract_spec,
)
from backtester.execution.fill_engine import FillEngine
from backtester.execution.order_types import (
    BarData,
    BracketSpec,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from backtester.execution.slippage import (
    FixedBpsSlippage,
    MarketImpactSlippage,
    NoSlippage,
    SlippageModel,
    VolatilitySlippage,
)

__all__ = [
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "Order",
    "Fill",
    "BarData",
    "BracketSpec",
    "FillEngine",
    "ExecutionSimulator",
    "BatchSubmissionResult",
    "TargetWeightOrderExecutor",
    "SlippageModel",
    "NoSlippage",
    "FixedBpsSlippage",
    "VolatilitySlippage",
    "MarketImpactSlippage",
    "CommissionScheme",
    "ZeroCommission",
    "PerShareCommission",
    "FixedBpsCommission",
    "TieredCommission",
    "CommissionConfig",
    "AssetClass",
    "ContractSpec",
    "UnsupportedCurrencyConversionError",
    "contract_spec_from_mapping",
    "get_contract_spec",
    "COMMON_SPECS",
]


def __getattr__(name: str) -> Any:
    """Load simulator types lazily to keep accounting imports acyclic."""
    if name in {
        "BatchSubmissionResult",
        "ExecutionSimulator",
        "TargetWeightOrderExecutor",
    }:
        return getattr(importlib.import_module("backtester.execution.simulator"), name)
    raise AttributeError(name)
