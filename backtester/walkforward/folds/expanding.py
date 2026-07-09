"""
Expanding-window fold scheme.

IS always starts at the earliest available date and grows each fold.
OOS slides forward by ``step_months``.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from dateutil.relativedelta import relativedelta

from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.folds.base import Fold
from backtester.walkforward.folds.purge import compute_purge_embargo


class ExpandingFoldScheme:
    """IS starts at data start and expands each fold."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self._cfg = config

    def generate_folds(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        trading_dates: pd.DatetimeIndex,
    ) -> List[Fold]:
        folds: List[Fold] = []
        step = relativedelta(months=self._cfg.effective_step_months)
        train_delta = relativedelta(months=self._cfg.train_months)
        test_delta = relativedelta(months=self._cfg.test_months)
        min_train_delta = relativedelta(months=self._cfg.min_train_months)

        fold_id = 0
        # First OOS starts after min_train_months from data start
        oos_cursor = start + train_delta

        while oos_cursor <= end:
            # IS: always from data start to just before OOS
            train_start_dt = start
            train_end_dt = oos_cursor - pd.Timedelta(days=1)
            oos_start_dt = oos_cursor
            oos_end_dt = min(oos_cursor + test_delta - pd.Timedelta(days=1), end)

            # Snap to trading dates
            train_dates = trading_dates[
                (trading_dates >= train_start_dt) & (trading_dates <= train_end_dt)
            ]
            oos_dates = trading_dates[
                (trading_dates >= oos_start_dt) & (trading_dates <= oos_end_dt)
            ]

            # Skip if insufficient IS data
            if len(train_dates) < max(1, len(trading_dates[
                (trading_dates >= train_start_dt)
                & (trading_dates <= train_start_dt + min_train_delta)
            ])):
                oos_cursor += step
                continue

            if len(oos_dates) == 0:
                oos_cursor += step
                continue

            t_start = train_dates[0]
            t_end = train_dates[-1]
            o_start = oos_dates[0]
            o_end = oos_dates[-1]

            eff_is_end, purge_start, purge_end = compute_purge_embargo(
                is_end=t_end,
                oos_start=o_start,
                purge_days=self._cfg.purge_days,
                embargo_pct=self._cfg.embargo_pct,
                trading_dates=trading_dates,
                is_start=t_start,
                max_holding_period_days=self._cfg.max_holding_period_days,
            )

            folds.append(
                Fold(
                    fold_id=fold_id,
                    scheme="expanding",
                    train_start=t_start,
                    train_end=t_end,
                    oos_start=o_start,
                    oos_end=o_end,
                    effective_is_end=eff_is_end,
                    purge_start=purge_start,
                    purge_end=purge_end,
                )
            )
            fold_id += 1
            oos_cursor += step

        return folds
