"""
Fold schemes subpackage — polymorphic fold generation with purge & embargo.

Usage::

    from backtester.walkforward.folds import fold_scheme_factory, Fold

    scheme = fold_scheme_factory("rolling", config)
    folds = scheme.generate_folds(start, end, trading_dates)

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.folds.anchored import AnchoredFoldScheme
from backtester.walkforward.folds.base import Fold, FoldScheme
from backtester.walkforward.folds.expanding import ExpandingFoldScheme
from backtester.walkforward.folds.purge import compute_purge_embargo
from backtester.walkforward.folds.rolling import RollingFoldScheme

__all__ = [
    "Fold",
    "FoldScheme",
    "RollingFoldScheme",
    "ExpandingFoldScheme",
    "AnchoredFoldScheme",
    "compute_purge_embargo",
    "fold_scheme_factory",
]


def fold_scheme_factory(config: WalkForwardConfig) -> FoldScheme:
    """Instantiate the correct FoldScheme from config.scheme."""
    _registry = {
        "rolling": RollingFoldScheme,
        "expanding": ExpandingFoldScheme,
        "anchored": AnchoredFoldScheme,
        # "cpcv" added in Phase 10
    }
    cls = _registry.get(config.scheme)
    if cls is None:
        raise ValueError(f"Unknown fold scheme {config.scheme!r}. Available: {list(_registry)}")
    return cls(config)
