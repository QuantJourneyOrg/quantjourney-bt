"""
FoldRunner — Command-pattern executor for a single walk-forward fold.

Runs IS and OOS phases, extracts metrics, and returns a ``FoldResult``.
Can be dispatched by the engine sequentially or in parallel.

Design: The runner operates on a *lightweight* metric-extraction path.
It uses ``PortfolioCalculations`` directly on sliced data rather than
going through the full report pipeline (which generates plots, PDFs, etc.).

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
from typing import Any, Callable, Dict, Optional
import asyncio
import inspect

import numpy as np
import pandas as pd

from backtester.walkforward.folds.base import Fold
from backtester.walkforward.result import FoldResult
from backtester.walkforward.statistics.overfit import overfit_ratio, efficiency


class FoldRunner:
    """
    Executes a single fold: IS metrics, OOS metrics, diagnostics.

    The runner does NOT call Backtester (which would re-fetch data
    and run the full pipeline). Instead it operates on pre-computed
    PortfolioData by slicing to IS / OOS windows and computing metrics
    via PortfolioCalculations.

    For optimization-enabled WF (Phase 6+), the runner will accept
    a ``backtester_factory`` and re-run the strategy per fold.
    """

    def __init__(
        self,
        fold: Fold,
        portfolio_data: Any,  # PortfolioData — avoid circular import
        *,
        blotter: Any = None,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.0,
        backtester_factory: Optional[Callable[..., Any]] = None,
        optimizer: Any = None,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._fold = fold
        self._portfolio_data = portfolio_data
        self._blotter = blotter
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate
        self._backtester_factory = backtester_factory
        self._optimizer = optimizer
        self._base_config = dict(base_config or {})

    def run(self) -> FoldResult:
        """Execute fold and return FoldResult."""
        portfolio_data = self._portfolio_data
        best_params = None
        optimizer_n_evals = None
        optimizer_best_objective = None
        if self._backtester_factory is not None:
            portfolio_data, opt_meta = self._run_fold_refit()
            best_params = opt_meta.get("best_params")
            optimizer_n_evals = opt_meta.get("optimizer_n_evals")
            optimizer_best_objective = opt_meta.get("optimizer_best_objective")

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
        oos_sr = oos_metrics.get("sharpe", 0.0)
        is_sr = is_metrics.get("sharpe", 0.0)
        or_val = overfit_ratio(is_sr, oos_sr)

        is_cagr = is_metrics.get("cagr", 0.0)
        oos_cagr = oos_metrics.get("cagr", 0.0)
        eff = efficiency(is_cagr, oos_cagr)

        # Sanity warnings
        warnings = []
        if self._backtester_factory is None:
            warnings.append(
                "Walk-forward result is slice diagnostics, not true out-of-sample: "
                "provide backtester_factory for per-fold refit."
            )
        if oos_sr < 0:
            warnings.append(
                f"Fold {self._fold.fold_id}: OOS Sharpe {oos_sr:.2f} is negative"
            )
        if or_val > 2.5:
            warnings.append(
                f"Fold {self._fold.fold_id}: overfit ratio {or_val:.1f} > 2.5"
            )

        # Fingerprint for this fold
        fp = self._compute_fold_fingerprint(is_metrics, oos_metrics)

        return FoldResult(
            fold=self._fold,
            # IS
            is_sharpe=is_sr,
            is_cagr=is_cagr,
            is_max_dd=is_metrics.get("max_dd", 0.0),
            is_volatility=is_metrics.get("volatility", 0.0),
            is_n_trades=is_metrics.get("n_trades", 0),
            is_win_rate=is_metrics.get("win_rate", 0.0),
            is_profit_factor=is_metrics.get("profit_factor", 0.0),
            is_avg_holding_days=is_metrics.get("avg_holding_days", 0.0),
            is_turnover_ann=is_metrics.get("turnover_ann", 0.0),
            # OOS
            oos_sharpe=oos_sr,
            oos_cagr=oos_cagr,
            oos_max_dd=oos_metrics.get("max_dd", 0.0),
            oos_volatility=oos_metrics.get("volatility", 0.0),
            oos_n_trades=oos_metrics.get("n_trades", 0),
            oos_win_rate=oos_metrics.get("win_rate", 0.0),
            oos_profit_factor=oos_metrics.get("profit_factor", 0.0),
            oos_avg_holding_days=oos_metrics.get("avg_holding_days", 0.0),
            oos_turnover_ann=oos_metrics.get("turnover_ann", 0.0),
            # OOS time series
            oos_returns=oos_returns,
            oos_nav=oos_nav,
            # Diagnostics
            overfit_ratio=or_val,
            efficiency=eff,
            sanity_warnings=warnings,
            fingerprint=fp,
            best_params=best_params,
            optimizer_n_evals=optimizer_n_evals,
            optimizer_best_objective=optimizer_best_objective,
        )

    # ── Private helpers ───────────────────────────────────────────────

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
    ) -> Dict[str, float]:
        """
        Compute key metrics for a date window using PortfolioCalculations.

        Uses lightweight direct computation rather than the full report pipeline.
        """
        returns = self._get_returns_for_window(start, end, portfolio_data=portfolio_data)

        if returns.empty or len(returns) < 2:
            return {
                "sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0,
                "volatility": 0.0, "n_trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_holding_days": 0.0,
                "turnover_ann": 0.0,
            }

        n_days = len(returns)
        trading_days = 252
        years = n_days / trading_days

        # CAGR
        total_return = (1.0 + returns).prod() - 1.0
        cagr = (1.0 + total_return) ** (1.0 / max(years, 1e-9)) - 1.0

        # Volatility
        vol = returns.std() * np.sqrt(trading_days)

        # Sharpe
        rfr_daily = self._risk_free_rate / trading_days
        excess = returns.mean() - rfr_daily
        sharpe = (excess / returns.std() * np.sqrt(trading_days)) if returns.std() > 0 else 0.0

        # Max drawdown
        nav = (1.0 + returns).cumprod()
        running_max = nav.cummax()
        dd = (nav - running_max) / running_max
        max_dd = float(dd.min())

        # Trade analytics (if blotter available)
        n_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_holding_days = 0.0
        turnover_ann = 0.0

        if self._blotter is not None:
            try:
                trades_df = self._blotter.get_trades_dataframe()
                if trades_df is not None and not trades_df.empty:
                    # Filter trades within window
                    if "timestamp" in trades_df.columns:
                        ts_col = pd.to_datetime(trades_df["timestamp"])
                        mask = (ts_col >= start) & (ts_col <= end)
                        window_trades = trades_df[mask]
                        n_trades = len(window_trades)

                        if n_trades > 0:
                            # Win rate from positive PnL trades
                            if "pnl" in window_trades.columns:
                                wins = (window_trades["pnl"] > 0).sum()
                                win_rate = wins / n_trades if n_trades > 0 else 0.0

                                gross_profit = window_trades.loc[
                                    window_trades["pnl"] > 0, "pnl"
                                ].sum()
                                gross_loss = abs(
                                    window_trades.loc[
                                        window_trades["pnl"] < 0, "pnl"
                                    ].sum()
                                )
                                profit_factor = (
                                    gross_profit / gross_loss
                                    if gross_loss > 0
                                    else float("inf")
                                )

                            # Turnover estimate
                            if "dollar_amount" in window_trades.columns:
                                total_volume = window_trades["dollar_amount"].abs().sum()
                                turnover_ann = total_volume / self._initial_capital / max(years, 1e-9)

            except Exception:
                pass  # gracefully skip if blotter not compatible

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

    def _run_fold_refit(self) -> tuple[Any, Dict[str, Any]]:
        best_params: Dict[str, Any] = {}
        optimizer_n_evals = None
        optimizer_best_objective = None

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
            optimizer_n_evals = getattr(opt_result, "n_evaluated", None)
            optimizer_best_objective = getattr(opt_result, "best_objective", None)

        bt = self._build_fold_backtester(best_params)
        self._run_backtester(bt)
        pdata = getattr(bt, "portfolio_data", None)
        if pdata is None:
            raise ValueError("backtester_factory result must expose portfolio_data after run")
        return pdata, {
            "best_params": best_params or None,
            "optimizer_n_evals": optimizer_n_evals,
            "optimizer_best_objective": optimizer_best_objective,
        }

    def _build_fold_backtester(self, best_params: Dict[str, Any]) -> Any:
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

    def _run_backtester(self, bt: Any) -> None:
        runner = getattr(bt, "run_strategy", None) or getattr(bt, "run", None)
        if runner is None:
            raise ValueError("backtester_factory result must provide run_strategy() or run()")
        result = runner()
        if inspect.isawaitable(result):
            self._run_async(result)

    @staticmethod
    def _run_async(awaitable: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("WalkForward per-fold refit cannot run inside an active event loop")

    def _compute_fold_fingerprint(
        self, is_metrics: Dict, oos_metrics: Dict
    ) -> str:
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
