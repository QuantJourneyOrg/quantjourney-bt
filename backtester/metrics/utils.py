"""
        Utilities for creating and customizing plots in the QuantJourney Framework.
        ---------------------------------------------------------

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from typing import Any

from dateutil.relativedelta import relativedelta

from backtester.utils.logger import logger


def get_nested_value(data: dict, path: str, default: Any = 0) -> Any:
    """
    Get value from nested dictionary using dot notation path.
    """
    try:
        # Special handling for specific paths
        if path == "start_date":
            return data["returns_index"][0].strftime("%Y-%m-%d")
        if path == "end_date":
            return data["returns_index"][-1].strftime("%Y-%m-%d")
        if path == "duration":
            start = data["returns_index"][0]
            end = data["returns_index"][-1]
            duration = relativedelta(end, start)
            return f"{duration.years}Y {duration.months}M"
        if path == "risk_free_rate":
            return data.get("config", {}).get("risk_free_rate", 0) * 100

        # Handle array indices in path
        keys = path.split(".")
        current = data
        for key in keys:
            if key.isdigit():  # Handle array indices
                current = current[int(key)]
            elif isinstance(current, dict):
                current = current.get(key, default)
            elif hasattr(current, "iloc") and key == "latest":  # Handle pandas latest value
                current = current.iloc[-1]
            else:
                return default

        return current
    except Exception as e:
        logger.error(f"Error getting nested value for path {path}: {str(e)}")
        return default
