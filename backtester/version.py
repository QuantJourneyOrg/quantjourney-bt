"""
Package version helpers for QuantJourney Backtester.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from importlib import metadata


def get_version() -> str:
    """Return the installed package version, falling back in editable/dev trees."""
    try:
        return metadata.version("quantjourney-bt")
    except metadata.PackageNotFoundError:
        return "0.8.0"


__version__ = get_version()
