"""
Package version helpers for QuantJourney Backtester.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path


def get_version() -> str:
    """Return the installed package version, falling back in editable/dev trees."""
    source_pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if source_pyproject.is_file():
        try:
            project = tomllib.loads(source_pyproject.read_text(encoding="utf-8"))
            return str(project["project"]["version"])
        except (KeyError, OSError, tomllib.TOMLDecodeError):
            pass
    try:
        return metadata.version("quantjourney-bt")
    except metadata.PackageNotFoundError:
        return "0.10.1"


__version__ = get_version()
