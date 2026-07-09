"""
Package version helpers for QuantJourney Backtester.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from importlib import metadata


def get_version() -> str:
    """Return the installed package version, falling back in editable/dev trees."""
    try:
        return metadata.version("quantjourney-bt")
    except metadata.PackageNotFoundError:
        return "0.9.1"


__version__ = get_version()
