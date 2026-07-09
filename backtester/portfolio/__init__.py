"""
backtester.portfolio package
-------------------------------

Exports core data containers and facades using lazy import.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "InstrumentData",
    "PortfolioData",
    "InstrumentCalculations",
    "PortfolioCalculations",
    "InstrumentPlots",
    "PortfolioPlots",
    "WeightCostBreakdown",
    "WeightCostModel",
    "FixedBpsWeightCostModel",
]


def __getattr__(name: str) -> Any:
    if name == "InstrumentData":
        return importlib.import_module("backtester.portfolio.instr_data").InstrumentData
    if name == "PortfolioData":
        return importlib.import_module("backtester.portfolio.portf_data").PortfolioData
    if name == "InstrumentCalculations":
        return importlib.import_module("backtester.portfolio.instr_calc").InstrumentCalculations
    if name == "PortfolioCalculations":
        return importlib.import_module("backtester.portfolio.portf_calc").PortfolioCalculations
    if name == "InstrumentPlots":
        return importlib.import_module("backtester.portfolio.instrument_plots").InstrumentPlots
    if name == "PortfolioPlots":
        return importlib.import_module("backtester.portfolio.portfolio_plots").PortfolioPlots
    if name == "WeightCostBreakdown":
        return importlib.import_module("backtester.portfolio.weight_cost").WeightCostBreakdown
    if name == "WeightCostModel":
        return importlib.import_module("backtester.portfolio.weight_cost").WeightCostModel
    if name == "FixedBpsWeightCostModel":
        return importlib.import_module("backtester.portfolio.weight_cost").FixedBpsWeightCostModel
    raise AttributeError(name)
