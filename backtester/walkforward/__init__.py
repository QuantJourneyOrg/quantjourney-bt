"""
Walk-Forward Validation & Parameter Optimization Framework.

Public API::

    from backtester.walkforward import (
        WalkForwardConfig,
        WalkForwardEngine,
        WalkForwardResult,
        FoldResult,
    )

    config = WalkForwardConfig(scheme="rolling", train_months=24, test_months=6)
    engine = WalkForwardEngine(config=config)
    result = engine.run(portfolio_data=pd_data)
    print(result.summary())

See ADR-002 for full design documentation.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.engine import WalkForwardEngine
from backtester.walkforward.result import FoldResult, WalkForwardResult
from backtester.walkforward.folds.base import Fold

__all__ = [
    "WalkForwardConfig",
    "WalkForwardEngine",
    "WalkForwardResult",
    "FoldResult",
    "Fold",
]
