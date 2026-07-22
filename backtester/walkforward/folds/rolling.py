"""
Rolling-window fold scheme.

IS window has fixed length ``train_months``.  Each successive fold
slides forward by ``step_months`` (default = ``test_months``).

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import pandas as pd
from dateutil.relativedelta import relativedelta

from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.folds.base import Fold
from backtester.walkforward.folds.purge import compute_pre_oos_purge


class RollingFoldScheme:
    """Fixed-width rolling IS window with non-overlapping OOS."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self._cfg = config

    def generate_folds(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        trading_dates: pd.DatetimeIndex,
    ) -> list[Fold]:
        folds: list[Fold] = []
        step = relativedelta(months=self._cfg.effective_step_months)
        train_delta = relativedelta(months=self._cfg.train_months)
        test_delta = relativedelta(months=self._cfg.test_months)
        min_train_delta = relativedelta(months=self._cfg.min_train_months)

        fold_id = 0
        oos_cursor = start + train_delta  # first OOS start candidate

        while oos_cursor <= end:
            train_start_dt = oos_cursor - train_delta
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
            min_train_dates = trading_dates[
                (trading_dates >= train_start_dt)
                & (trading_dates <= train_start_dt + min_train_delta)
            ]
            if len(train_dates) < len(min_train_dates) or len(train_dates) == 0:
                oos_cursor += step
                continue

            if len(oos_dates) == 0:
                oos_cursor += step
                continue

            t_start = train_dates[0]
            t_end = train_dates[-1]
            o_start = oos_dates[0]
            o_end = oos_dates[-1]

            eff_is_end, purge_start, purge_end = compute_pre_oos_purge(
                is_end=t_end,
                oos_start=o_start,
                purge_days=self._cfg.purge_days,
                extra_pre_oos_purge_pct=self._cfg.resolved_extra_pre_oos_purge_pct,
                trading_dates=trading_dates,
                is_start=t_start,
                max_holding_period_days=self._cfg.max_holding_period_days,
            )

            folds.append(
                Fold(
                    fold_id=fold_id,
                    scheme="rolling",
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
