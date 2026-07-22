"""
FoldRunner — Command-pattern executor for a single walk-forward fold.

Runs IS and OOS phases, extracts metrics, and returns a ``FoldResult``.
Can be dispatched by the engine sequentially or in parallel.

Design: The runner operates on a *lightweight* metric-extraction path.
It uses ``PortfolioCalculations`` directly on sliced data rather than
going through the full report pipeline (which generates plots, PDFs, etc.).

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from backtester.reporting_frequency import infer_periods_per_year
from backtester.utils.logger import logger
from backtester.walkforward.folds.base import Fold
from backtester.walkforward.result import FoldResult
from backtester.walkforward.statistics.overfit import efficiency, overfit_ratio
from backtester.walkforward.statistics.pbo import selected_trial_rank_logit


class FoldRunner:
    """
    Executes a single fold: IS metrics, OOS metrics, diagnostics.

    Without a ``backtester_factory`` the runner does NOT call Backtester
    (which would re-fetch data and run the full pipeline). Instead it
    operates on pre-computed PortfolioData by slicing to IS / OOS windows
    — slice_diagnostics mode; the "OOS" slices are still in-sample.

    When a ``backtester_factory`` IS provided (per_fold_refit mode), the
    runner re-runs the strategy per fold — optionally optimizing on the
    IS window first — and computes metrics from that fold's own NAV.
    See ``_build_fold_backtester`` for a leakage caveat that strategy
    authors must respect.
    """

    def __init__(
        self,
        fold: Fold,
        portfolio_data: Any,  # PortfolioData — avoid circular import
        *,
        blotter: Any = None,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.0,
        backtester_factory: Callable[..., Any] | None = None,
        optimizer: Any = None,
        base_config: dict[str, Any] | None = None,
        rank_stability_trials: int | None = None,
        pbo_trials: int = 0,
    ) -> None:
        self._fold = fold
        self._portfolio_data = portfolio_data
        self._blotter = blotter
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate
        self._backtester_factory = backtester_factory
        self._optimizer = optimizer
        self._base_config = dict(base_config or {})
        self._rank_stability_trials = (
            rank_stability_trials if rank_stability_trials is not None else pbo_trials
        )

    def run(self) -> FoldResult:
        """Execute fold and return FoldResult (fold_status='failed' on refit crash)."""
        portfolio_data = self._portfolio_data
        opt_meta: dict[str, Any] = {}
        if self._backtester_factory is not None:
            try:
                portfolio_data, opt_meta = self._run_fold_refit()
            except RuntimeError as exc:
                if "active event loop" in str(exc):
                    raise  # usage error — caller must use run_async()
                logger.error(f"[WalkForward] Fold {self._fold.fold_id} refit FAILED: {exc}")
                return self._failed_result(f"refit failed: {exc}")
            except Exception as exc:
                logger.error(f"[WalkForward] Fold {self._fold.fold_id} refit FAILED: {exc}")
                return self._failed_result(f"refit failed: {exc}")

        return self._build_result(portfolio_data, opt_meta)

    async def run_async(self) -> FoldResult:
        """Execute fold from an active event loop and return FoldResult."""
        portfolio_data = self._portfolio_data
        opt_meta: dict[str, Any] = {}
        if self._backtester_factory is not None:
            try:
                portfolio_data, opt_meta = await self._run_fold_refit_async()
            except Exception as exc:
                logger.error(f"[WalkForward] Fold {self._fold.fold_id} refit FAILED: {exc}")
                return self._failed_result(f"refit failed: {exc}")

        return self._build_result(portfolio_data, opt_meta)

    def _build_result(self, portfolio_data: Any, opt_meta: dict[str, Any]) -> FoldResult:
        best_params = opt_meta.get("best_params")
        optimizer_n_evals = opt_meta.get("optimizer_n_evals")
        optimizer_best_objective = opt_meta.get("optimizer_best_objective")
        optimizer_trial_values = opt_meta.get("optimizer_trial_values")
        rank_stability_candidate_oos = opt_meta.get("rank_stability_candidate_oos")
        rank_stability_selected_logit = opt_meta.get("rank_stability_selected_logit")

        is_metrics = self._compute_metrics_for_window(
            self._fold.train_start,
            self._fold.effective_is_end,
            portfolio_data=portfolio_data,
        )
        oos_metrics = self._compute_metrics_for_window(
            self._fold.oos_start,
            self._fold.oos_end,
            portfolio_data=portfolio_data,
        )

        # Build OOS returns and NAV
        oos_returns = self._get_returns_for_window(
            self._fold.oos_start, self._fold.oos_end, portfolio_data=portfolio_data
        )
        oos_nav = (1.0 + oos_returns).cumprod() if not oos_returns.empty else pd.Series(dtype=float)

        # Diagnostics
        oos_sr = oos_metrics.get("sharpe", float("nan"))
        is_sr = is_metrics.get("sharpe", float("nan"))
        is_cagr = is_metrics.get("cagr", float("nan"))
        oos_cagr = oos_metrics.get("cagr", float("nan"))

        # Failed fold: empty/too-short NAV window → NaN metrics, never
        # silent zeros (Sharpe 0.0 would be indistinguishable from a
        # genuinely flat strategy).
        fold_failed = oos_returns.empty or not np.isfinite(oos_sr) or not np.isfinite(is_sr)
        if fold_failed:
            or_val = float("nan")
            eff = float("nan")
        else:
            or_val = overfit_ratio(is_sr, oos_sr)
            eff = efficiency(is_cagr, oos_cagr)

        # Sanity warnings
        warnings = []
        if fold_failed:
            warnings.append(
                f"Fold {self._fold.fold_id}: FAILED — empty or insufficient "
                "NAV window; metrics are NaN and the fold is excluded from "
                "aggregates"
            )
        if self._backtester_factory is None:
            warnings.append(
                f"Fold {self._fold.fold_id}: slice diagnostics mode — metrics "
                "are sliced from the full-period NAV (no per-fold refit)"
            )
        if opt_meta.get("optimizer_all_failed"):
            warnings.append(
                f"Fold {self._fold.fold_id}: ALL optimizer trials failed — "
                "fold ran with base config (unoptimized)"
            )
        if oos_sr < 0:
            warnings.append(f"Fold {self._fold.fold_id}: OOS Sharpe {oos_sr:.2f} is negative")
        if or_val > 2.5:
            warnings.append(f"Fold {self._fold.fold_id}: overfit ratio {or_val:.1f} > 2.5")

        # Fingerprint for this fold
        fp = self._compute_fold_fingerprint(is_metrics, oos_metrics)

        nan = float("nan")
        return FoldResult(
            fold=self._fold,
            # IS
            is_sharpe=is_sr,
            is_cagr=is_cagr,
            is_max_dd=is_metrics.get("max_dd", nan),
            is_volatility=is_metrics.get("volatility", nan),
            is_n_trades=is_metrics.get("n_trades", 0),
            is_win_rate=is_metrics.get("win_rate", nan),
            is_profit_factor=is_metrics.get("profit_factor", nan),
            is_avg_holding_days=is_metrics.get("avg_holding_days", nan),
            is_turnover_ann=is_metrics.get("turnover_ann", nan),
            # OOS
            oos_sharpe=oos_sr,
            oos_cagr=oos_cagr,
            oos_max_dd=oos_metrics.get("max_dd", nan),
            oos_volatility=oos_metrics.get("volatility", nan),
            oos_n_trades=oos_metrics.get("n_trades", 0),
            oos_win_rate=oos_metrics.get("win_rate", nan),
            oos_profit_factor=oos_metrics.get("profit_factor", nan),
            oos_avg_holding_days=oos_metrics.get("avg_holding_days", nan),
            oos_turnover_ann=oos_metrics.get("turnover_ann", nan),
            # OOS time series
            oos_returns=oos_returns,
            oos_nav=oos_nav,
            # Diagnostics
            overfit_ratio=or_val,
            efficiency=eff,
            sanity_warnings=warnings,
            fingerprint=fp,
            fold_status="failed" if fold_failed else "ok",
            best_params=best_params,
            optimizer_n_evals=optimizer_n_evals,
            optimizer_best_objective=optimizer_best_objective,
            optimizer_trial_values=optimizer_trial_values,
            rank_stability_candidate_oos=rank_stability_candidate_oos,
            rank_stability_selected_logit=rank_stability_selected_logit,
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _failed_result(self, reason: str) -> FoldResult:
        """All-NaN FoldResult with fold_status='failed' — never silent zeros."""
        nan = float("nan")
        empty = pd.Series(dtype=float)
        return FoldResult(
            fold=self._fold,
            is_sharpe=nan,
            is_cagr=nan,
            is_max_dd=nan,
            is_volatility=nan,
            is_n_trades=0,
            is_win_rate=nan,
            is_profit_factor=nan,
            is_avg_holding_days=nan,
            is_turnover_ann=nan,
            oos_sharpe=nan,
            oos_cagr=nan,
            oos_max_dd=nan,
            oos_volatility=nan,
            oos_n_trades=0,
            oos_win_rate=nan,
            oos_profit_factor=nan,
            oos_avg_holding_days=nan,
            oos_turnover_ann=nan,
            oos_returns=empty,
            oos_nav=empty,
            overfit_ratio=nan,
            efficiency=nan,
            sanity_warnings=[f"Fold {self._fold.fold_id}: FAILED — {reason}"],
            fingerprint=self._compute_fold_fingerprint({}, {}),
            fold_status="failed",
        )

    def _get_returns_for_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        portfolio_data: Any = None,
    ) -> pd.Series:
        """Extract daily returns for a date window."""
        pdata = portfolio_data if portfolio_data is not None else self._portfolio_data
        nav = pdata.net_asset_value
        returns = nav.pct_change()
        window_returns = returns.loc[(returns.index >= start) & (returns.index <= end)].dropna()
        if len(window_returns) < 1:
            return pd.Series(dtype=float)
        return window_returns

    def _compute_metrics_for_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        portfolio_data: Any = None,
    ) -> dict[str, float]:
        """
        Compute key metrics for a date window using PortfolioCalculations.

        Uses lightweight direct computation rather than the full report pipeline.
        """
        pdata = portfolio_data if portfolio_data is not None else self._portfolio_data
        returns = self._get_returns_for_window(start, end, portfolio_data=portfolio_data)

        if returns.empty or len(returns) < 2:
            # Empty/too-short window → NaN, NOT zeros: a fabricated
            # Sharpe of 0.0 is indistinguishable from a flat strategy.
            nan = float("nan")
            return {
                "sharpe": nan,
                "cagr": nan,
                "max_dd": nan,
                "volatility": nan,
                "n_trades": 0,
                "win_rate": nan,
                "profit_factor": nan,
                "avg_holding_days": nan,
                "turnover_ann": nan,
            }

        n_days = len(returns)
        periods_per_year = max(
            int(
                getattr(
                    pdata,
                    "periods_per_year",
                    infer_periods_per_year(pdata.net_asset_value.index),
                )
            ),
            1,
        )
        years = n_days / periods_per_year

        # CAGR
        total_return = (1.0 + returns).prod() - 1.0
        cagr = (1.0 + total_return) ** (1.0 / max(years, 1e-9)) - 1.0

        # Volatility
        vol = returns.std() * np.sqrt(periods_per_year)

        # Sharpe
        rfr_daily = self._risk_free_rate / periods_per_year
        excess = returns.mean() - rfr_daily
        sharpe = excess / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0

        # Max drawdown
        nav = (1.0 + returns).cumprod()
        running_max = nav.cummax()
        dd = (nav - running_max) / running_max
        max_dd = float(dd.min())

        # Trade analytics (if blotter available). Defaults are NaN, not
        # 0.0 — a missing blotter must render "n/a", never a fake zero.
        # avg_holding_days is not computed on this lightweight path at
        # all, so it is always NaN (was: hardcoded 0.0).
        nan = float("nan")
        n_trades = 0
        win_rate = nan
        profit_factor = nan
        avg_holding_days = nan
        turnover_ann = nan

        if self._blotter is not None:
            try:
                trades_df = self._blotter.get_trades_dataframe()
                if trades_df is not None and not trades_df.empty:
                    # Filter trades within window
                    timestamp_col = next(
                        (c for c in ("Timestamp", "timestamp") if c in trades_df.columns),
                        None,
                    )
                    if timestamp_col is not None:
                        ts_col = pd.to_datetime(trades_df[timestamp_col], utc=True)
                        start_utc = pd.Timestamp(start)
                        end_utc = pd.Timestamp(end)
                        start_utc = (
                            start_utc.tz_localize("UTC")
                            if start_utc.tzinfo is None
                            else start_utc.tz_convert("UTC")
                        )
                        end_utc = (
                            end_utc.tz_localize("UTC")
                            if end_utc.tzinfo is None
                            else end_utc.tz_convert("UTC")
                        )
                        mask = (ts_col >= start_utc) & (ts_col <= end_utc)
                        window_trades = trades_df[mask]
                        n_trades = len(window_trades)
                        turnover_ann = 0.0  # no trades → genuinely zero turnover

                        if n_trades > 0:
                            # Win rate from positive PnL trades
                            pnl_col = next(
                                (c for c in ("PnL", "pnl") if c in window_trades.columns),
                                None,
                            )
                            if pnl_col is not None:
                                wins = (window_trades[pnl_col] > 0).sum()
                                win_rate = wins / n_trades if n_trades > 0 else 0.0

                                gross_profit = window_trades.loc[
                                    window_trades[pnl_col] > 0, pnl_col
                                ].sum()
                                gross_loss = abs(
                                    window_trades.loc[window_trades[pnl_col] < 0, pnl_col].sum()
                                )
                                profit_factor = (
                                    gross_profit / gross_loss if gross_loss > 0 else float("inf")
                                )

                            # Annualized dollar turnover — unified with the
                            # reproducibility.py definition:
                            #   (total notional traded / 2) / avg NAV, annualized.
                            # 1.0 = the whole portfolio replaced once per year.
                            notional_col = next(
                                (
                                    c
                                    for c in ("TradeValue", "dollar_amount", "trade_value")
                                    if c in window_trades.columns
                                ),
                                None,
                            )
                            if notional_col is not None:
                                total_notional = window_trades[notional_col].abs().sum()
                                nav_series = pdata.net_asset_value
                                nav_window = nav_series.loc[
                                    (nav_series.index >= start) & (nav_series.index <= end)
                                ]
                                avg_nav = float(nav_window.mean()) if len(nav_window) else nan
                                turnover_ann = (
                                    (total_notional / 2.0) / avg_nav / max(years, 1e-9)
                                    if avg_nav and avg_nav > 0
                                    else nan
                                )

            except Exception as exc:
                # Blotter incompatible with this lightweight path: report
                # NaN (rendered n/a), never a plausible-looking zero.
                logger.warning(
                    f"[WalkForward] Fold {self._fold.fold_id}: trade analytics "
                    f"failed ({exc}); trade metrics set to NaN"
                )
                n_trades = 0
                win_rate = nan
                profit_factor = nan
                avg_holding_days = nan
                turnover_ann = nan

        return {
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "max_dd": max_dd,
            "volatility": float(vol),
            "n_trades": n_trades,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_holding_days": float(avg_holding_days),
            "turnover_ann": float(turnover_ann),
        }

    def _run_fold_refit(self) -> tuple[Any, dict[str, Any]]:
        opt_meta: dict[str, Any] = {}
        best_params: dict[str, Any] = {}

        if self._optimizer is not None:
            opt_result = self._run_async(
                self._optimizer.optimize(
                    self._backtester_factory,
                    self._fold.train_start.strftime("%Y-%m-%d"),
                    self._fold.effective_is_end.strftime("%Y-%m-%d"),
                    self._base_config,
                )
            )
            best_params = dict(getattr(opt_result, "best_params", {}) or {})
            opt_meta = self._optimizer_meta(opt_result, best_params)

            # Opt-in rolling rank stability: evaluate top-K IS trials OOS.
            if self._rank_stability_trials >= 2:
                candidate_oos, lam = self._evaluate_rank_stability_candidates(opt_result)
                opt_meta["rank_stability_candidate_oos"] = candidate_oos
                opt_meta["rank_stability_selected_logit"] = lam

        bt = self._build_fold_backtester(best_params)
        self._run_backtester(bt)
        self._blotter = getattr(bt, "blotter", None)
        pdata = getattr(bt, "portfolio_data", None)
        if pdata is None:
            raise ValueError("backtester_factory result must expose portfolio_data after run")
        self._validate_fold_portfolio_bounds(pdata)
        return pdata, opt_meta

    async def _run_fold_refit_async(self) -> tuple[Any, dict[str, Any]]:
        opt_meta: dict[str, Any] = {}
        best_params: dict[str, Any] = {}

        if self._optimizer is not None:
            opt_result_or_awaitable = self._optimizer.optimize(
                self._backtester_factory,
                self._fold.train_start.strftime("%Y-%m-%d"),
                self._fold.effective_is_end.strftime("%Y-%m-%d"),
                self._base_config,
            )
            if inspect.isawaitable(opt_result_or_awaitable):
                opt_result = await opt_result_or_awaitable
            else:
                opt_result = opt_result_or_awaitable
            best_params = dict(getattr(opt_result, "best_params", {}) or {})
            opt_meta = self._optimizer_meta(opt_result, best_params)

            if self._rank_stability_trials >= 2:
                candidate_oos, lam = await self._evaluate_rank_stability_candidates_async(
                    opt_result
                )
                opt_meta["rank_stability_candidate_oos"] = candidate_oos
                opt_meta["rank_stability_selected_logit"] = lam

        bt = self._build_fold_backtester(best_params)
        await self._run_backtester_async(bt)
        self._blotter = getattr(bt, "blotter", None)
        pdata = getattr(bt, "portfolio_data", None)
        if pdata is None:
            raise ValueError("backtester_factory result must expose portfolio_data after run")
        self._validate_fold_portfolio_bounds(pdata)
        return pdata, opt_meta

    @staticmethod
    def _optimizer_meta(opt_result: Any, best_params: dict[str, Any]) -> dict[str, Any]:
        """Extract fold-level optimizer metadata from an OptimizationResult."""
        return {
            "best_params": best_params or None,
            "optimizer_n_evals": getattr(opt_result, "n_evaluated", None),
            "optimizer_best_objective": getattr(opt_result, "best_objective", None),
            "optimizer_trial_values": FoldRunner._extract_trial_values(opt_result),
            "optimizer_all_failed": bool(getattr(opt_result, "all_trials_failed", False)),
        }

    @staticmethod
    def _extract_trial_values(opt_result: Any) -> list[float] | None:
        """Completed-trial objective values — the DSR trial population."""
        trials = getattr(opt_result, "trials", None) or []
        vals = [
            float(t.value)
            for t in trials
            if getattr(t, "state", "COMPLETE") == "COMPLETE"
            and np.isfinite(getattr(t, "value", np.nan))
        ]
        if vals:
            return vals
        # Fallback for optimizers that only populate all_results
        df = getattr(opt_result, "all_results", None)
        if df is not None and "objective" in getattr(df, "columns", []):
            arr = pd.to_numeric(df["objective"], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return [float(v) for v in arr]
        return None

    # ── Rolling top-K OOS rank-stability evaluation (opt-in) ─────────

    def _rank_stability_top_trials(self, opt_result: Any) -> list[Any]:
        top_k = getattr(opt_result, "top_k", None)
        if top_k is None:
            return []
        top = top_k(self._rank_stability_trials)
        return top if len(top) >= 2 else []

    def _rank_stability_logit_from_scores(self, oos_scores: list[float]) -> float | None:
        """λ of the IS-selected trial (top_k[0]) among candidate OOS scores."""
        if not oos_scores:
            return None
        selected = oos_scores[0]
        finite_scores = [float(s) for s in oos_scores if np.isfinite(s)]
        if not np.isfinite(selected) or len(finite_scores) < 2:
            # Failed candidate backtests are not real OOS ranks. Including
            # them as -inf would make any surviving selected trial look
            # artificially strong.
            if not finite_scores:
                message = (
                    f"[WalkForward] Fold {self._fold.fold_id}: ALL "
                    f"{len(oos_scores)} rank-stability candidate backtests failed — "
                    "selection rank meaningless; logit set to NaN"
                )
            else:
                message = (
                    f"[WalkForward] Fold {self._fold.fold_id}: only "
                    f"{len(finite_scores)}/{len(oos_scores)} rank-stability candidate "
                    "backtests produced finite OOS scores — selection rank "
                    "not computable; logit set to NaN"
                )
            logger.warning(message)
            return float("nan")
        return selected_trial_rank_logit(float(selected), finite_scores)

    def _oos_score_for_params(self, pdata: Any) -> float:
        """OOS Sharpe of a candidate backtest (higher = better rank)."""
        metrics = self._compute_metrics_for_window(
            self._fold.oos_start, self._fold.oos_end, portfolio_data=pdata
        )
        return float(metrics["sharpe"])

    def _evaluate_rank_stability_candidates(
        self, opt_result: Any
    ) -> tuple[list[float] | None, float | None]:
        """
        Re-backtest the optimizer's top-K IS trials over the fold window
        and score each on the OOS slice (by OOS Sharpe; the ranking
        assumes higher-is-better regardless of the IS objective
        direction).  Failed candidates score -inf (worst rank).
        """
        top = self._rank_stability_top_trials(opt_result)
        if not top:
            return None, None

        oos_scores: list[float] = []
        for tr in top:
            try:
                bt = self._build_fold_backtester(dict(tr.params))
                self._run_backtester(bt)
                pdata = getattr(bt, "portfolio_data", None)
                if pdata is None:
                    raise ValueError("candidate backtester exposes no portfolio_data")
                self._validate_fold_portfolio_bounds(pdata)
                oos_scores.append(self._oos_score_for_params(pdata))
            except Exception:
                oos_scores.append(float("-inf"))
        return oos_scores, self._rank_stability_logit_from_scores(oos_scores)

    async def _evaluate_rank_stability_candidates_async(
        self, opt_result: Any
    ) -> tuple[list[float] | None, float | None]:
        """Async twin of ``_evaluate_rank_stability_candidates``."""
        top = self._rank_stability_top_trials(opt_result)
        if not top:
            return None, None

        oos_scores: list[float] = []
        for tr in top:
            try:
                bt = self._build_fold_backtester(dict(tr.params))
                await self._run_backtester_async(bt)
                pdata = getattr(bt, "portfolio_data", None)
                if pdata is None:
                    raise ValueError("candidate backtester exposes no portfolio_data")
                self._validate_fold_portfolio_bounds(pdata)
                oos_scores.append(self._oos_score_for_params(pdata))
            except Exception:
                oos_scores.append(float("-inf"))
        return oos_scores, self._rank_stability_logit_from_scores(oos_scores)

    def _build_fold_backtester(self, best_params: dict[str, Any]) -> Any:
        """
        Instantiate the per-fold backtester over [train_start, oos_end].

        LEAKAGE WARNING for strategy authors: the fold backtester runs
        over the FULL fold window (IS + OOS). Any strategy that computes
        full-window statistics — z-score/min-max normalization, whole-
        sample means or volatilities, fitted scalers — leaks OOS data
        into the IS fit even under per-fold refit. Signal features must
        use rolling or train-only (expanding, anchored at train_start)
        statistics for the OOS metrics to be honest.
        """
        assert self._backtester_factory is not None
        train_start = self._fold.train_start.strftime("%Y-%m-%d")
        train_end = self._fold.effective_is_end.strftime("%Y-%m-%d")
        oos_start = self._fold.oos_start.strftime("%Y-%m-%d")
        oos_end = self._fold.oos_end.strftime("%Y-%m-%d")
        config = {
            **self._base_config,
            **best_params,
            "backtest_period": {"start": train_start, "end": oos_end},
        }
        try:
            return self._backtester_factory(
                fold=self._fold,
                train_start=train_start,
                train_end=train_end,
                oos_start=oos_start,
                oos_end=oos_end,
                params=best_params,
                base_config=self._base_config,
            )
        except TypeError:
            return self._backtester_factory(**config)

    def _validate_fold_portfolio_bounds(self, portfolio_data: Any) -> None:
        """Fail closed when a custom factory ignores the requested fold dates."""
        nav = getattr(portfolio_data, "net_asset_value", None)
        if nav is None or not hasattr(nav, "index") or len(nav.index) == 0:
            raise ValueError(
                "per-fold refit produced no dated net_asset_value; fold bounds cannot be audited"
            )

        index = pd.DatetimeIndex(nav.index)
        if index.hasnans:
            raise ValueError(
                "per-fold refit produced NaT timestamps; fold bounds cannot be audited"
            )
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        actual_start = pd.Timestamp(index.min()).normalize()
        actual_end = pd.Timestamp(index.max()).normalize()

        requested_start = pd.Timestamp(self._fold.train_start)
        requested_end = pd.Timestamp(self._fold.oos_end)
        if requested_start.tzinfo is not None:
            requested_start = requested_start.tz_convert("UTC").tz_localize(None)
        if requested_end.tzinfo is not None:
            requested_end = requested_end.tz_convert("UTC").tz_localize(None)
        requested_start = requested_start.normalize()
        requested_end = requested_end.normalize()

        if actual_start < requested_start or actual_end > requested_end:
            raise ValueError(
                "per-fold refit escaped requested date bounds: "
                f"requested [{requested_start.date()}, {requested_end.date()}], "
                f"produced [{actual_start.date()}, {actual_end.date()}]. "
                "Propagate the factory train_start/oos_end strings into "
                "backtest_period."
            )

    def _run_backtester(self, bt: Any) -> None:
        runner = getattr(bt, "run_strategy", None) or getattr(bt, "run", None)
        if runner is None:
            raise ValueError("backtester_factory result must provide run_strategy() or run()")
        result = runner()
        if inspect.isawaitable(result):
            self._run_async(result)

    async def _run_backtester_async(self, bt: Any) -> None:
        runner = getattr(bt, "run_strategy", None) or getattr(bt, "run", None)
        if runner is None:
            raise ValueError("backtester_factory result must provide run_strategy() or run()")
        result = runner()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _run_async(awaitable: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise RuntimeError("WalkForward per-fold refit cannot run inside an active event loop")

    def _compute_fold_fingerprint(self, is_metrics: dict, oos_metrics: dict) -> str:
        """Deterministic hash for this fold's config + results."""
        payload = json.dumps(
            {
                "fold_id": self._fold.fold_id,
                "scheme": self._fold.scheme,
                "train_start": str(self._fold.train_start),
                "train_end": str(self._fold.train_end),
                "oos_start": str(self._fold.oos_start),
                "oos_end": str(self._fold.oos_end),
                "is_sharpe": is_metrics.get("sharpe"),
                "oos_sharpe": oos_metrics.get("sharpe"),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
