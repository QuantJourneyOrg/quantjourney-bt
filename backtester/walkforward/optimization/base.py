"""
Optimizer Protocol — common interface for all optimizers.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from backtester.walkforward.optimization.result import OptimizationResult


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for parameter optimizers."""

    async def optimize(
        self,
        backtester_factory: Callable[..., Any],
        train_start: str,
        train_end: str,
        base_config: dict[str, Any],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OptimizationResult:
        """
        Run optimization on the IS window.

        Args:
            backtester_factory: Callable that produces a fresh backtester.
            train_start: IS window start date.
            train_end: IS window end date.
            base_config: Strategy config to override with param combos.
            progress_callback: Optional real-time progress reporting.
            cancel_check: Optional cancellation callback.

        Returns:
            OptimizationResult with best params and all trial results.
        """
        ...
