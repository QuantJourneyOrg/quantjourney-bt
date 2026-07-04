"""
WalkForwardEngine — orchestrator for walk-forward validation.

Responsibilities (and only these):
    1. Generate folds via the appropriate FoldScheme.
    2. Dispatch FoldRunner for each fold (sequential or parallel).
    3. Aggregate OOS results via statistics subpackage.
    4. Build and return WalkForwardResult.

All computation is delegated — the engine is a thin loop.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

import pandas as pd

from backtester.utils.logger import logger
from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.folds import fold_scheme_factory
from backtester.walkforward.runner import FoldRunner
from backtester.walkforward.result import FoldResult, WalkForwardResult
from backtester.walkforward.statistics.aggregation import (
    aggregate_oos_returns,
    compute_composite_metrics,
)
from backtester.walkforward.statistics.overfit import (
    aggregate_overfit_ratio,
    aggregate_efficiency,
    sharpe_decay,
)
from backtester.walkforward.statistics.deflated_sharpe import deflated_sharpe
from backtester.walkforward.statistics.pbo import probability_of_backtest_overfitting
from backtester.walkforward.statistics.interpretation import interpret_metrics
from backtester.walkforward.persistence import save_checkpoint, load_checkpoint


class WalkForwardEngine:
    """
    Orchestrates multi-fold walk-forward validation.

    Usage::

        from backtester.walkforward import WalkForwardEngine, WalkForwardConfig

        config = WalkForwardConfig(scheme="rolling", train_months=24, test_months=6)
        engine = WalkForwardEngine(config=config)
        result = engine.run(portfolio_data=pd_data)
        print(result.summary())
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        *,
        blotter: Any = None,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.0,
        checkpoint_dir: Optional[str] = None,
        backtester_factory: Optional[Callable[..., Any]] = None,
        optimizer: Any = None,
        base_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._config = config
        self._blotter = blotter
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate
        self._checkpoint_dir = checkpoint_dir
        self._backtester_factory = backtester_factory
        self._optimizer = optimizer
        self._base_config = dict(base_config or {})

    def run(
        self,
        portfolio_data: Any,  # PortfolioData
        *,
        resume: bool = False,
    ) -> WalkForwardResult:
        """
        Execute walk-forward validation.

        Args:
            portfolio_data: Full-period PortfolioData.
            resume: If True and checkpoint_dir is set, resume from last checkpoint.

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        # 1. Extract trading dates from NAV index
        trading_dates = portfolio_data.net_asset_value.index.sort_values()
        if len(trading_dates) < 2:
            raise ValueError("Insufficient data for walk-forward (need >= 2 trading days)")

        start = trading_dates[0]
        end = trading_dates[-1]

        if self._config.verbose:
            logger.info(
                f"[WalkForward] Starting {self._config.scheme} WF: "
                f"{start.date()} → {end.date()}, "
                f"train={self._config.train_months}m, test={self._config.test_months}m, "
                f"purge={self._config.purge_days}d, embargo={self._config.embargo_pct:.0%}"
            )

        # 2. Generate folds
        scheme = fold_scheme_factory(self._config)
        folds = scheme.generate_folds(start, end, trading_dates)

        if not folds:
            raise ValueError(
                f"No valid folds generated for {self._config.scheme} scheme "
                f"with train={self._config.train_months}m, test={self._config.test_months}m "
                f"over {start.date()} → {end.date()}"
            )

        if self._config.verbose:
            logger.info(f"[WalkForward] Generated {len(folds)} folds")

        # 3. Resume from checkpoint if requested
        completed_results: dict[int, FoldResult] = {}
        if resume and self._checkpoint_dir:
            completed_results = load_checkpoint(self._checkpoint_dir)
            if completed_results and self._config.verbose:
                logger.info(
                    f"[WalkForward] Resumed {len(completed_results)} folds from checkpoint"
                )

        # 4. Execute folds
        fold_results: list[FoldResult] = []

        for fold in folds:
            # Skip already-completed folds
            if fold.fold_id in completed_results:
                fold_results.append(completed_results[fold.fold_id])
                continue

            if self._config.verbose:
                logger.info(
                    f"[WalkForward] Fold {fold.fold_id}: "
                    f"IS {fold.train_start.date()} → {fold.effective_is_end.date()}, "
                    f"OOS {fold.oos_start.date()} → {fold.oos_end.date()}"
                )

            runner = FoldRunner(
                fold=fold,
                portfolio_data=portfolio_data,
                blotter=self._blotter,
                initial_capital=self._initial_capital,
                risk_free_rate=self._risk_free_rate,
                backtester_factory=self._backtester_factory,
                optimizer=self._optimizer,
                base_config=self._base_config,
            )
            result = runner.run()
            fold_results.append(result)

            # Checkpoint
            if self._checkpoint_dir:
                completed_results[fold.fold_id] = result
                save_checkpoint(self._checkpoint_dir, completed_results)

        # 5. Aggregate
        wf_result = self._aggregate(fold_results)

        if self._config.verbose:
            dsr_str = f", DSR={wf_result.deflated_sharpe:.2f}" if wf_result.deflated_sharpe is not None else ""
            pbo_str = f", PBO={wf_result.pbo:.2f}" if wf_result.pbo is not None else ""
            logger.info(
                f"[WalkForward] Complete: OOS Sharpe={wf_result.oos_sharpe:.2f}, "
                f"Overfit Ratio={wf_result.overfit_ratio:.2f}, "
                f"Efficiency={wf_result.efficiency:.2f}"
                f"{dsr_str}{pbo_str}"
            )

        return wf_result

    # ── Aggregation ───────────────────────────────────────────────────

    def _aggregate(self, fold_results: list[FoldResult]) -> WalkForwardResult:
        """Build WalkForwardResult from completed fold results."""

        # Concatenate OOS returns
        oos_returns_list = [fr.oos_returns for fr in fold_results if not fr.oos_returns.empty]
        oos_returns, oos_nav = aggregate_oos_returns(oos_returns_list)

        # Composite metrics from concatenated returns
        composite = compute_composite_metrics(
            oos_returns, risk_free_rate=self._risk_free_rate
        )

        # Overfit diagnostics
        is_sharpes = [fr.is_sharpe for fr in fold_results]
        oos_sharpes = [fr.oos_sharpe for fr in fold_results]
        is_cagrs = [fr.is_cagr for fr in fold_results]
        oos_cagrs = [fr.oos_cagr for fr in fold_results]

        or_val = aggregate_overfit_ratio(is_sharpes, oos_sharpes)
        eff_val = aggregate_efficiency(is_cagrs, oos_cagrs)
        decay = sharpe_decay(oos_sharpes)

        # ── Deflated Sharpe Ratio ─────────────────────────────────────
        dsr_val = None
        if self._config.compute_deflated_sharpe and len(is_sharpes) >= 2:
            # n_trials = total optimizer evaluations (or n_folds if no optimizer)
            optimizer_evals = [
                fr.optimizer_n_evals
                for fr in fold_results
                if fr.optimizer_n_evals is not None and fr.optimizer_n_evals > 0
            ]
            n_trials = sum(optimizer_evals) if optimizer_evals else len(fold_results)
            dsr_val = deflated_sharpe(is_sharpes, n_trials)

        # ── Probability of Backtest Overfitting ───────────────────────
        pbo_val = None
        if self._config.compute_pbo and len(fold_results) >= 4:
            pbo_val = probability_of_backtest_overfitting(
                is_sharpes, oos_sharpes, seed=self._config.seed,
            )

        # Collect warnings
        all_warnings = []
        for fr in fold_results:
            all_warnings.extend(fr.sanity_warnings)

        # Add aggregate-level warnings
        if decay < -0.05:
            all_warnings.append("Sharpe decay slope is strongly negative — alpha may be decaying")
        elif decay < -0.01:
            all_warnings.append("Sharpe decay slope is negative — mild alpha decay detected")

        if or_val > 2.5:
            all_warnings.append(f"Aggregate overfit ratio {or_val:.1f} > 2.5 — likely overfit")

        neg_folds = sum(1 for s in oos_sharpes if s < self._config.min_oos_sharpe)
        if neg_folds > 0:
            all_warnings.append(
                f"{neg_folds}/{len(fold_results)} folds have OOS Sharpe "
                f"below {self._config.min_oos_sharpe}"
            )

        # Fingerprint
        fp_payload = json.dumps(
            {
                "config": self._config.to_dict(),
                "n_folds": len(fold_results),
                "oos_sharpe": composite["sharpe"],
                "fold_fingerprints": [fr.fingerprint for fr in fold_results],
            },
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(fp_payload.encode()).hexdigest()[:16]

        return WalkForwardResult(
            folds=fold_results,
            config_dict=self._config.to_dict(),
            oos_sharpe=composite["sharpe"],
            oos_cagr=composite["cagr"],
            oos_max_dd=composite["max_dd"],
            oos_returns=oos_returns,
            oos_nav=oos_nav,
            overfit_ratio=or_val,
            efficiency=eff_val,
            sharpe_decay=decay,
            deflated_sharpe=dsr_val,
            pbo=pbo_val,
            fingerprint=fingerprint,
            warnings=all_warnings,
        )
