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

Supports rolling, expanding, anchored, and purged/embargoed fold schemes,
grid-search and Optuna parameter optimization, and overfitting diagnostics
(deflated Sharpe ratio, PBO). See strategies/example_wf_*.py for usage.

Copyright (c) 2026 QuantJourney.
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
