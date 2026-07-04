"""
backtester.engines package

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

# Optional imports — avoid hard failure if deps are missing
try:
    from .performance import StrategyPerformanceAnalysis
except Exception:
    StrategyPerformanceAnalysis = None  # type: ignore
