"""
backtester.engines package

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

# Optional imports — avoid hard failure if deps are missing
try:
    from .performance import StrategyPerformanceAnalysis
except Exception:
    StrategyPerformanceAnalysis = None  # type: ignore
