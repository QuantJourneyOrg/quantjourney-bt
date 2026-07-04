"""
Walk-Forward Persistence — checkpoint save/load for resumable runs.

Stores completed FoldResult objects as JSON (without heavy Series data)
so that a crashed run can resume from the last completed fold.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from backtester.utils.logger import logger


def save_checkpoint(
    checkpoint_dir: str,
    completed_results: Dict[int, Any],  # Dict[fold_id, FoldResult]
) -> None:
    """
    Save completed fold results to a checkpoint file.

    Only saves lightweight metadata (no pd.Series). The fold can be
    re-identified by fold_id + fingerprint for validation.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = path / "wf_checkpoint.json"

    data = {}
    for fold_id, fr in completed_results.items():
        data[str(fold_id)] = {
            "fold_id": fr.fold.fold_id,
            "scheme": fr.fold.scheme,
            "train_start": str(fr.fold.train_start),
            "oos_end": str(fr.fold.oos_end),
            "is_sharpe": fr.is_sharpe,
            "oos_sharpe": fr.oos_sharpe,
            "fingerprint": fr.fingerprint,
            "completed": True,
        }

    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(
    checkpoint_dir: str,
) -> Dict[int, Any]:
    """
    Load checkpoint. Returns dict of fold_id → checkpoint data.

    Note: Returns lightweight dicts, not full FoldResult objects.
    The engine uses these to skip re-computation of completed folds.
    Currently returns empty dict (full FoldResult reconstruction from
    checkpoint requires storing the OOS returns, which is Phase 2+).
    """
    checkpoint_file = Path(checkpoint_dir) / "wf_checkpoint.json"
    if not checkpoint_file.exists():
        return {}

    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        logger.info(f"[WalkForward] Found checkpoint with {len(data)} completed folds")
        # For MVP: we log checkpoint existence but re-run all folds
        # Full resume requires serialising OOS returns
        return {}
    except Exception as e:
        logger.warning(f"[WalkForward] Failed to load checkpoint: {e}")
        return {}
