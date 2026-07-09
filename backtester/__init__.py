"""
backtester — QuantJourney Backtester
----------------------------------------------

Standalone strategy backtesting package. Fetches market data from the
QuantJourney API; all computation (signals, weights, performance, plots)
runs locally.

Usage:
    from backtester import Backtester

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from backtester.version import __version__

__all__ = [
    "Backtester",
    "CloudBacktester",          # deprecated alias
    "InstrumentData",
    "PortfolioData",
    "InstrumentCalculations",
    "PortfolioCalculations",
    "ReportingFrequency",
    "ReportingFrequencyConfig",
]


def __getattr__(name: str) -> Any:
    if name == "Backtester":
        return importlib.import_module("backtester.core").Backtester
    if name == "CloudBacktester":
        warnings.warn(
            "CloudBacktester is deprecated — use Backtester instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.import_module("backtester.core").Backtester
    if name == "InstrumentData":
        return importlib.import_module("backtester.portfolio.instr_data").InstrumentData
    if name == "PortfolioData":
        return importlib.import_module("backtester.portfolio.portf_data").PortfolioData
    if name == "InstrumentCalculations":
        return importlib.import_module("backtester.portfolio.instr_calc").InstrumentCalculations
    if name == "PortfolioCalculations":
        return importlib.import_module("backtester.portfolio.portf_calc").PortfolioCalculations
    if name == "ReportingFrequency":
        return importlib.import_module("backtester.reporting_frequency").ReportingFrequency
    if name == "ReportingFrequencyConfig":
        return importlib.import_module("backtester.reporting_frequency").ReportingFrequencyConfig
    raise AttributeError(name)
