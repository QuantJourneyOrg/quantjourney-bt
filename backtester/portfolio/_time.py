"""
Time-index alignment helpers for portfolio series.

Copyright (c) 2026 QuantJourney.
Updated: 07.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd


def normalize_time_index_like(
    obj: pd.Series | pd.DataFrame,
    target_index: pd.Index,
) -> pd.Series | pd.DataFrame:
    """Return a copy whose DatetimeIndex timezone matches ``target_index``."""
    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    if not isinstance(target_index, pd.DatetimeIndex):
        return out

    if target_index.tz is not None:
        out.index = (
            out.index.tz_convert(target_index.tz)
            if out.index.tz is not None
            else out.index.tz_localize(target_index.tz)
        )
    elif out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


def reindex_time_like(
    obj: pd.Series | pd.DataFrame,
    target_index: pd.Index,
    *args,
    **kwargs,
) -> pd.Series | pd.DataFrame:
    """Timezone-normalize ``obj`` before reindexing to ``target_index``."""
    return normalize_time_index_like(obj, target_index).reindex(
        index=target_index,
        *args,  # noqa: B026
        **kwargs,
    )
