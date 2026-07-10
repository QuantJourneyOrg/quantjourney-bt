"""
Purge & embargo computation — shared across all fold schemes.

Purge removes the last N trading days from IS to prevent label leakage.
Embargo removes an additional fraction of IS length for serial correlation.
If ``max_holding_period_days`` is set, purge = max(purge_days, holding).

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd


def compute_purge_embargo(
    is_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    purge_days: int,
    embargo_pct: float,
    trading_dates: pd.DatetimeIndex,
    is_start: pd.Timestamp,
    max_holding_period_days: int | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Compute effective IS end after purge & embargo.

    Returns:
        (effective_is_end, purge_start, purge_end)

        - effective_is_end: last IS trading day to include.
        - purge_start: first excluded trading day.
        - purge_end: last excluded trading day (day before oos_start).
    """
    # IS trading days
    is_dates = trading_dates[(trading_dates >= is_start) & (trading_dates <= is_end)]
    is_length = len(is_dates)

    if is_length == 0:
        return is_end, is_end, is_end

    # Effective purge days
    effective_purge = purge_days
    if max_holding_period_days is not None:
        effective_purge = max(effective_purge, max_holding_period_days)

    # Embargo: additional trading days removed from end of IS
    embargo_days = max(0, int(embargo_pct * is_length))

    total_remove = effective_purge + embargo_days

    if total_remove == 0:
        # No purge/embargo: an empty exclusion window is reported as None,
        # never as is_dates[-0] == is_dates[0] (which claimed the whole IS
        # window was purged).
        return is_dates[-1], None, None

    if total_remove >= is_length:
        # Edge case: purge + embargo exceeds IS window
        effective_is_end = is_dates[0]
        purge_start = is_dates[0]
    else:
        # Remove last `total_remove` trading days from IS
        effective_is_end = is_dates[-(total_remove + 1)]
        purge_start = is_dates[-total_remove]

    # Purge end = last trading day before oos_start
    pre_oos = trading_dates[trading_dates < oos_start]
    purge_end = pre_oos[-1] if len(pre_oos) > 0 else oos_start

    return effective_is_end, purge_start, purge_end
