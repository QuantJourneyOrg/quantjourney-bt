"""
Execution Engine — Order Types, Fill Engine, Slippage, Commission.

Provides institutional-grade order management for backtesting:
  Market, Limit, Stop, StopTrail, OCO, Bracket orders
  with pluggable slippage models and commission schemes.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from backtester.execution.order_types import (
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Order,
    Fill,
    BarData,
    BracketSpec,
)
from backtester.execution.fill_engine import FillEngine
from backtester.execution.slippage import (
    SlippageModel,
    NoSlippage,
    FixedBpsSlippage,
    VolatilitySlippage,
    MarketImpactSlippage,
)
from backtester.execution.commission import (
    CommissionScheme,
    ZeroCommission,
    PerShareCommission,
    FixedBpsCommission,
    TieredCommission,
    CommissionConfig,
)
from backtester.execution.contract_spec import (
    AssetClass,
    ContractSpec,
    get_contract_spec,
    COMMON_SPECS,
)

__all__ = [
    "OrderType", "OrderSide", "OrderStatus", "TimeInForce", "Order", "Fill", "BarData",
    "BracketSpec", "FillEngine",
    "SlippageModel", "NoSlippage", "FixedBpsSlippage", "VolatilitySlippage",
    "MarketImpactSlippage",
    "CommissionScheme", "ZeroCommission", "PerShareCommission",
    "FixedBpsCommission", "TieredCommission", "CommissionConfig",
    "AssetClass", "ContractSpec", "get_contract_spec", "COMMON_SPECS",
]
