"""
Standalone logger for backtester.

Matches the output format of the main QuantJourney logger but uses
only the stdlib logging module.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

import logging
import os
import sys
from datetime import datetime


def _resolve_log_level() -> int:
    raw = os.environ.get("QJ_LOG_LEVEL", "INFO").strip().upper()
    aliases = {
        "TRACE": "DEBUG",
        "WARN": "WARNING",
        "QUIET": "ERROR",
        "SILENT": "CRITICAL",
    }
    name = aliases.get(raw, raw)
    return getattr(logging, name, logging.INFO)


class _QJFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.upper()
        # Trim the module path to last two parts for readability
        parts = record.name.split(".")
        short = ".".join(parts[-2:]) if len(parts) >= 2 else record.name
        return f"[{ts}] [{level}] [{short}] {record.getMessage()}"


def _build_logger() -> logging.Logger:
    _logger = logging.getLogger("backtester")
    level = _resolve_log_level()
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_QJFormatter())
        handler.setLevel(level)
        _logger.addHandler(handler)
        _logger.propagate = False
    else:
        for handler in _logger.handlers:
            handler.setLevel(level)
    _logger.setLevel(level)
    return _logger


logger = _build_logger()
