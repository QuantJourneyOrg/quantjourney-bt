"""
backtester.risk — Pluggable Risk Models
========================================

Sits between ``_compute_weights()`` and the ``RebalanceEngine``
in the Backtester pipeline::

    signals → weights → **RiskModel.adjust(weights)** → rebalance → execution

Each model is a callable that receives raw target weights and asset
returns, and returns adjusted weights.  Models can be composed via
``RiskModelChain``.

Usage in strategy constructor::

    from backtester.risk import VolTargetModel, PositionLimitModel, RiskModelChain

    strategy = MyStrategy(
        ...,
        risk_model=RiskModelChain([
            VolTargetModel(target_vol=0.15, lookback=63, max_leverage=2.0),
            PositionLimitModel(max_weight=0.40),
        ]),
    )

The strategy's ``_compute_weights()`` returns *raw* weights (e.g.
proportional to alpha). The risk model adjusts them *before*
RebalanceEngine decides which days to trade.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from backtester.risk.base import RiskModel, RiskModelChain
from backtester.risk.inverse_vol import InverseVolModel
from backtester.risk.position_limit import PositionLimitModel
from backtester.risk.pre_trade import PreTradeDecision, PreTradeResult, PreTradeRisk
from backtester.risk.risk_parity import RiskParityModel
from backtester.risk.vol_target import VolTargetModel

__all__ = [
    "RiskModel",
    "RiskModelChain",
    "VolTargetModel",
    "InverseVolModel",
    "RiskParityModel",
    "PositionLimitModel",
    "PreTradeDecision",
    "PreTradeResult",
    "PreTradeRisk",
]
