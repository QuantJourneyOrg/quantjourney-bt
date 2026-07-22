"""Lazy public exports for the reporting engines.

Importing a lightweight engine such as :mod:`backtester.engines.blotter`
must not initialize matplotlib or plotting helpers.  The optional export
retains its historical ``None`` fallback when a reporting dependency is
unavailable, but is resolved only when callers request it.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "StrategyPerformanceAnalysis": (".performance", "StrategyPerformanceAnalysis"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve optional reporting exports without penalizing core imports."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    try:
        value = getattr(importlib.import_module(module_name, __name__), attribute)
    except Exception:
        value = None
    globals()[name] = value
    return value
