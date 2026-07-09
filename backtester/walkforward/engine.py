"""
WalkForwardEngine — orchestrator for walk-forward validation.

Responsibilities (and only these):
    1. Generate folds via the appropriate FoldScheme.
    2. Dispatch FoldRunner for each fold (sequential or parallel).
    3. Aggregate OOS results via statistics subpackage.
    4. Build and return WalkForwardResult.

All computation is delegated — the engine is a thin loop.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from backtester.utils.logger import logger
from backtester.walkforward.config import WalkForwardConfig
from backtester.walkforward.folds import fold_scheme_factory
from backtester.walkforward.runner import FoldRunner
from backtester.walkforward.result import FoldResult, WalkForwardResult
from backtester.walkforward.statistics.aggregation import (
    aggregate_oos_returns,
    bootstrap_sharpe_ci,
    compute_composite_metrics,
)
from backtester.walkforward.statistics.overfit import (
    aggregate_overfit_ratio,
    aggregate_efficiency,
    sharpe_decay,
)
from backtester.walkforward.statistics.deflated_sharpe import deflated_sharpe
from backtester.walkforward.statistics.pbo import pbo_from_selected_ranks
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

    @property
    def mode(self) -> str:
        return "per_fold_refit" if self._backtester_factory is not None else "slice_diagnostics"

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
                f"mode={self.mode}, "
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
                pbo_trials=self._config.pbo_trials,
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

    async def run_async(
        self,
        portfolio_data: Any,  # PortfolioData
        *,
        resume: bool = False,
    ) -> WalkForwardResult:
        """
        Execute walk-forward validation from an active event loop.

        This is equivalent to ``run()`` but awaits per-fold backtester refits when
        ``backtester_factory`` returns an async strategy.
        """
        trading_dates = portfolio_data.net_asset_value.index.sort_values()
        if len(trading_dates) < 2:
            raise ValueError("Insufficient data for walk-forward (need >= 2 trading days)")

        start = trading_dates[0]
        end = trading_dates[-1]

        if self._config.verbose:
            logger.info(
                f"[WalkForward] Starting {self._config.scheme} WF: "
                f"{start.date()} → {end.date()}, "
                f"mode={self.mode}, "
                f"train={self._config.train_months}m, test={self._config.test_months}m, "
                f"purge={self._config.purge_days}d, embargo={self._config.embargo_pct:.0%}"
            )

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

        completed_results: dict[int, FoldResult] = {}
        if resume and self._checkpoint_dir:
            completed_results = load_checkpoint(self._checkpoint_dir)
            if completed_results and self._config.verbose:
                logger.info(
                    f"[WalkForward] Resumed {len(completed_results)} folds from checkpoint"
                )

        fold_results: list[FoldResult] = []

        for fold in folds:
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
                pbo_trials=self._config.pbo_trials,
            )
            result = await runner.run_async()
            fold_results.append(result)

            if self._checkpoint_dir:
                completed_results[fold.fold_id] = result
                save_checkpoint(self._checkpoint_dir, completed_results)

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

        # Collect warnings
        all_warnings: list[str] = []

        # Failed folds (empty NAV window / refit crash) carry NaN metrics
        # and must be excluded from all aggregates — never averaged in.
        ok_folds = [fr for fr in fold_results if fr.fold_status == "ok"]
        failed_folds = [fr for fr in fold_results if fr.fold_status != "ok"]
        if failed_folds:
            failed_ids = ", ".join(str(fr.fold.fold_id) for fr in failed_folds)
            all_warnings.append(
                f"{len(failed_folds)}/{len(fold_results)} folds FAILED "
                f"(ids: {failed_ids}) — excluded from aggregate metrics"
            )
            logger.warning(f"[WalkForward] {all_warnings[-1]}")

        # Concatenate OOS returns
        oos_returns_list = [fr.oos_returns for fr in ok_folds if not fr.oos_returns.empty]
        if oos_returns_list:
            combined_index = pd.concat(oos_returns_list).index
            if combined_index.duplicated().any():
                # Mirror the aggregation.py log warning into the result's
                # warnings so it survives into summary() output.
                all_warnings.append(
                    "Overlapping OOS windows (step < test): duplicated dates "
                    "are averaged across folds, which can bias the composite "
                    "Sharpe upward; prefer step_months >= test_months"
                )
        oos_returns, oos_nav = aggregate_oos_returns(oos_returns_list)

        # Composite metrics from concatenated returns
        if oos_returns.empty:
            nan = float("nan")
            composite = {"sharpe": nan, "cagr": nan, "max_dd": nan, "volatility": nan}
        else:
            composite = compute_composite_metrics(
                oos_returns, risk_free_rate=self._risk_free_rate
            )

        # Bootstrap CI for the composite Sharpe (stationary block
        # bootstrap, seeded from config.seed for reproducibility).
        sharpe_ci = None
        if not oos_returns.empty:
            sharpe_ci = bootstrap_sharpe_ci(
                oos_returns,
                seed=self._config.seed,
                risk_free_rate=self._risk_free_rate,
            )

        # Overfit diagnostics (ok folds only)
        is_sharpes = [fr.is_sharpe for fr in ok_folds]
        oos_sharpes = [fr.oos_sharpe for fr in ok_folds]
        is_cagrs = [fr.is_cagr for fr in ok_folds]
        oos_cagrs = [fr.oos_cagr for fr in ok_folds]

        if ok_folds:
            or_val = aggregate_overfit_ratio(is_sharpes, oos_sharpes)
            eff_val = aggregate_efficiency(is_cagrs, oos_cagrs)
            decay = sharpe_decay(oos_sharpes)
        else:
            or_val = float("nan")
            eff_val = float("nan")
            decay = float("nan")

        # ── Deflated Sharpe Ratio (Bailey & López de Prado 2014) ──────
        # Candidate: the aggregated OOS daily return series (T, skew, raw
        # kurtosis, and per-day Sharpe all come from the SAME series).
        # Trial population: per-trial IS objective values pooled across
        # folds — the same population defines both √V[SR] and N, so the
        # E[max SR] deflation is internally consistent.  Objective values
        # are annualized Sharpes, so they are de-annualized (√252) to the
        # daily units of the candidate.  No optimizer → N = 1 → the DSR
        # honestly reduces to the PSR of the OOS Sharpe.  Folds are NOT
        # trials and are never used as the trial population.
        #
        # Mode caveats:
        # - slice_diagnostics: the "OOS" series is an in-sample slice of
        #   one full-period run — DSR would bless in-sample performance,
        #   so it is reported as unavailable (None) with a reason.
        # - per_fold_refit: the concatenated OOS series spans folds with
        #   potentially DIFFERENT best_params. The DSR then measures the
        #   whole tune-then-trade process, not one fixed parameterization.
        dsr_val = None
        dsr_reason = None
        if self.mode == "slice_diagnostics":
            dsr_reason = (
                "in-sample; DSR not meaningful without independent trials"
            )
        elif self._config.compute_deflated_sharpe and len(oos_returns) >= 3:
            ret_std = float(oos_returns.std(ddof=1))
            if ret_std > 0.0:
                ann = math.sqrt(252.0)
                rfr_daily = self._risk_free_rate / 252.0
                sr_daily = (float(oos_returns.mean()) - rfr_daily) / ret_std
                skew = float(oos_returns.skew())
                kurt_raw = float(oos_returns.kurt()) + 3.0  # pandas kurt() is excess

                pooled_trials = [
                    v / ann
                    for fr in ok_folds
                    for v in (fr.optimizer_trial_values or [])
                    if v is not None and np.isfinite(v)
                ]
                optimizer_used = any(
                    fr.optimizer_n_evals for fr in ok_folds if fr.optimizer_n_evals
                )
                if optimizer_used and not pooled_trials:
                    all_warnings.append(
                        "DSR computed without multiple-testing deflation (N=1): "
                        "optimizer was used but per-trial objective values are "
                        "unavailable"
                    )

                dsr_val = deflated_sharpe(
                    pooled_trials,
                    n_trials=len(pooled_trials) if pooled_trials else 1,
                    observed_sr=sr_daily,
                    n_obs=len(oos_returns),
                    skewness=skew,
                    kurtosis=kurt_raw,
                )

        # ── Probability of Backtest Overfitting (rank-based) ──────────
        # Real PBO needs the IS-selected trial's OOS rank among candidate
        # trials per fold (WalkForwardConfig.pbo_trials).  Without that
        # data PBO is reported as unavailable — never as a reassuring 0.
        pbo_val = None
        pbo_reason = None
        if self._config.compute_pbo:
            logits = [
                fr.pbo_selected_logit
                for fr in fold_results
                if fr.pbo_selected_logit is not None
            ]
            if len(logits) >= 4:
                pbo_val = pbo_from_selected_ranks(logits)
                if math.isnan(pbo_val):
                    pbo_val = None
                    pbo_reason = "no finite selection-rank logits across folds"
            elif logits:
                pbo_reason = (
                    f"only {len(logits)} folds have per-trial OOS ranks (need >= 4)"
                )
            else:
                pbo_reason = (
                    "requires per-trial OOS evaluation; set "
                    "WalkForwardConfig.pbo_trials=K (>=2) with an optimizer to "
                    "evaluate top-K trials OOS per fold"
                )
            if pbo_val is None and pbo_reason:
                all_warnings.append(f"PBO unavailable: {pbo_reason}")
        if self._backtester_factory is None:
            all_warnings.append(
                "Walk-forward mode=slice_diagnostics: metrics are computed from "
                "a full-period NAV; pass backtester_factory for per-fold refit."
            )
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
                f"{neg_folds}/{len(ok_folds)} folds have OOS Sharpe "
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
            sharpe_ci_5pct=sharpe_ci[0] if sharpe_ci is not None else None,
            sharpe_ci_95pct=sharpe_ci[1] if sharpe_ci is not None else None,
            overfit_ratio=or_val,
            efficiency=eff_val,
            sharpe_decay=decay,
            deflated_sharpe=dsr_val,
            deflated_sharpe_reason=dsr_reason,
            pbo=pbo_val,
            pbo_available=pbo_val is not None,
            pbo_reason=pbo_reason,
            fingerprint=fingerprint,
            warnings=all_warnings,
            mode=self.mode,
        )
