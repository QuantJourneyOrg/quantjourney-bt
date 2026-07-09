"""
Anchored-window fold scheme.

Same as expanding, but IS starts at a fixed anchor date (data start).
IS width grows from ``train_months`` onward — first fold has exactly
``train_months`` of IS, subsequent folds add ``step_months`` each.

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


class AnchoredFoldScheme:
    """
    IS anchored at data start, grows by step_months each fold.

    Identical to ExpandingFoldScheme in the current implementation
    (anchor = data start). Separated for future extension where
    anchor can be a user-specified date (e.g. for burn-in periods).
    """

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

        anchor = start  # future: configurable anchor date
        fold_id = 0
        oos_cursor = anchor + train_delta

        while oos_cursor <= end:
            train_start_dt = anchor
            train_end_dt = oos_cursor - pd.Timedelta(days=1)
            oos_start_dt = oos_cursor
            oos_end_dt = min(oos_cursor + test_delta - pd.Timedelta(days=1), end)

            train_dates = trading_dates[
                (trading_dates >= train_start_dt) & (trading_dates <= train_end_dt)
            ]
            oos_dates = trading_dates[
                (trading_dates >= oos_start_dt) & (trading_dates <= oos_end_dt)
            ]

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
                    scheme="anchored",
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
