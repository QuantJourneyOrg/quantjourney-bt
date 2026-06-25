"""
ReportingMixin — performance analysis, archiving, blotter, summary
===================================================================

Extracted from core.py to keep the Backtester class focused on the
strategy pipeline (data → signals → weights → positions → performance).

All methods expect the host class to have the attributes set by
Backtester.__init__ (portfolio_data, instruments_data, blotter, etc.).

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

import logging
import os
import platform
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from backtester.version import __version__ as BACKTESTER_VERSION

try:
    from backtester.utils.logger import logger
except Exception:
    logger = logging.getLogger("backtester")


class ReportingMixin:
    """Performance analysis, archiving, blotter utilities, and print_summary."""

    # ─────────────────────────────────────────────────────────────────
    # Performance Analysis
    # ─────────────────────────────────────────────────────────────────

    async def _generate_strategy_analysis(self) -> None:
        """Generate performance reports using StrategyPerformanceAnalysis."""
        try:
            from backtester.engines import StrategyPerformanceAnalysis
            spa = StrategyPerformanceAnalysis(
                config={
                    "show_text_reports": self._show_text_reports,
                    "save_text_reports": self._save_text_reports,
                    "save_portfolio_plots": self._save_portfolio_plots,
                    "show_portfolio_plots": self._show_portfolio_plots,
                    "save_instrument_plots": self._save_instrument_plots,
                    "show_instrument_plots": self._show_instrument_plots,
                    "save_pdf_report": self._save_pdf_report,
                    "theme_plots": self._theme_plots,
                    "dpi": self._plot_dpi,
                    "reports_directory": self._reports_directory,
                    "benchmark": self._benchmark,
                    "reporting_frequency": getattr(self, "_reporting_frequency", "daily"),
                },
                portfolio_data=self.portfolio_data,
                instruments_data=self.instruments_data,
                data_connector=None,
                sdk_client=self._sdk_client,
                strategy_name=self.strategy_name,
                strategy_type=self.strategy_type,
                base_currency=self.base_currency,
                backtest_period=self.backtest_period,
                initial_capital=self.initial_capital,
            )
            # Build strategy parameters for the appendix
            strategy_params = self._build_strategy_params()

            # Get strategy source code
            strategy_code = ""
            try:
                import inspect
                strategy_code = inspect.getsource(self.__class__)
            except Exception:
                pass

            await spa.generate_strategy_performance_analysis(
                portfolio_data=self.portfolio_data,
                instruments_data=self.instruments_data,
                blotter=self.blotter,
                strategy_parameters=strategy_params,
                strategy_code=strategy_code,
                fill_engine=self.fill_engine,
            )
        except Exception as e:
            logger.warning(f"[Backtester] Performance analysis skipped: {e}")
            if getattr(self, "_strict_reporting", False):
                raise

    def _build_strategy_params(self) -> Dict[str, str]:
        """Build the strategy parameters dict for reports/appendix."""
        params = {
            "Backtester Version": BACKTESTER_VERSION,
            "Strategy Name": self.strategy_name,
            "Strategy Type": self.strategy_type,
            "Instruments": ", ".join(self.instruments),
            "Backtest Period": f"{self.backtest_period.start} to {self.backtest_period.end}",
            "Initial Capital": f"${self.initial_capital:,.0f}",
            "Base Currency": self.base_currency,
            "Data Source": self._source,
            "Granularity": self._granularity,
            "Max Position Size": f"{self.max_position_size:.0%}",
            "Rebalance Policy": str(self._rebalance_policy),
        }
        if getattr(self, "_risk_model", None) is not None:
            params["Risk Model"] = str(self._risk_model)
            params["Target Volatility"] = f"{self.target_volatility:.0%}"
        else:
            params["Risk Model"] = "not specified"
        # Add rebalance stats if available
        if hasattr(self, '_rebalance_stats'):
            rs = self._rebalance_stats
            params["Rebalance Count"] = str(rs.get('rebalance_count', '-'))
            params["Avg Days Between Rebal"] = f"{rs.get('avg_days_between', 0):.1f}"
            if rs.get('tracking_error_count', 0):
                params["TE Triggers"] = str(rs['tracking_error_count'])
            if rs.get('partial_positions_saved', 0):
                params["Partial Positions Saved"] = str(rs['partial_positions_saved'])
        if self.indicators_config:
            for ind in self.indicators_config:
                fn = ind.get("function", "?")
                ind_params = ind.get("params", {})
                params[f"Indicator: {fn}"] = str(ind_params)
        return params

    # ─────────────────────────────────────────────────────────────────
    # Archiving
    # ─────────────────────────────────────────────────────────────────

    async def _archive_strategy_data(self) -> None:
        """Archive strategy results."""
        try:
            from pathlib import Path
            from backtester.engines import StrategyArchive

            save_pickle_archive = os.environ.get(
                "QJ_SAVE_PICKLE_ARCHIVE",
                "0",
            ).strip().lower() in {"1", "true", "yes", "y", "on"}

            archive_folder = Path(self._reports_directory) / self.strategy_name
            if not save_pickle_archive:
                for filename in ("portfolio_data.pkl", "instruments_data.pkl", "blotter.pkl"):
                    stale_path = archive_folder / filename
                    if stale_path.exists():
                        stale_path.unlink()
                        logger.info(f"[Public] Removed stale pickle archive artifact: {stale_path}")

            sta = StrategyArchive(
                strategy_name=self.strategy_name,
                save_folder=self._reports_directory,
                save_blotter=(
                    save_pickle_archive
                    and self.blotter is not None
                    and len(self.blotter.trades) > 0
                ),
                save_portfolio_data=save_pickle_archive,
                save_instruments_data=save_pickle_archive,
            )
            await sta.archive_strategy_data(
                portfolio_data=self.portfolio_data,
                instruments_data=self.instruments_data,
                blotter=self.blotter,
                save_dir=self._reports_directory,
                metadata=self._build_run_metadata(),
            )
        except Exception as e:
            logger.warning(f"[Backtester] Archive skipped: {e}")
            if getattr(self, "_strict_reporting", False):
                raise

    def _build_run_metadata(self) -> Dict[str, Any]:
        """Build machine-readable metadata saved next to archived results."""
        benchmark = getattr(self, "_benchmark", {}) or {}
        period = getattr(self, "backtest_period", None)
        return {
            "backtester_version": BACKTESTER_VERSION,
            "package_name": "quantjourney-bt",
            "strategy_name": getattr(self, "strategy_name", None),
            "strategy_type": getattr(self, "strategy_type", None),
            "base_currency": getattr(self, "base_currency", None),
            "initial_capital": getattr(self, "initial_capital", None),
            "instruments": list(getattr(self, "instruments", []) or []),
            "backtest_period": {
                "start": getattr(period, "start", None),
                "end": getattr(period, "end", None),
            },
            "data_source": getattr(self, "_source", None),
            "granularity": getattr(self, "_granularity", None),
            "execution_mode": getattr(self, "execution_mode", None),
            "rebalance_policy": str(getattr(self, "_rebalance_policy", "")),
            "reporting_frequency": getattr(self, "_reporting_frequency", None),
            "theme_plots": getattr(self, "_theme_plots", None),
            "plot_dpi": getattr(self, "_plot_dpi", None),
            "benchmark": {
                "symbol": benchmark.get("symbol"),
                "name": benchmark.get("name"),
            },
            "session_id": getattr(self, "session_id", None),
            "dataset_id": getattr(self, "dataset_id", None),
            "strict_reporting": getattr(self, "_strict_reporting", False),
            "strict_data_fetch": getattr(self, "_strict_data_fetch", False),
            "quiet": getattr(self, "_quiet", False),
            "no_reports": getattr(self, "_no_reports", False),
            "python": {
                "executable": sys.executable,
                "version": sys.version.split()[0],
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
            "runtime_options": {
                "QJ_LOG_LEVEL": os.environ.get("QJ_LOG_LEVEL", "INFO"),
                "QJ_QUIET": os.environ.get("QJ_QUIET", "0"),
                "QJ_NO_REPORTS": os.environ.get("QJ_NO_REPORTS", "0"),
                "QJ_PLOT_DPI": os.environ.get("QJ_PLOT_DPI"),
                "QJ_OUTPUT_DIR": os.environ.get("QJ_OUTPUT_DIR"),
                "QJ_REPORTING_FREQUENCY": os.environ.get("QJ_REPORTING_FREQUENCY"),
                "QJ_SAVE_PICKLE_ARCHIVE": os.environ.get("QJ_SAVE_PICKLE_ARCHIVE", "0"),
            },
            "timings_seconds": dict(getattr(self, "_timings", {}) or {}),
            "run_started_at": getattr(self, "_run_started_at", None),
            "metadata_generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ─────────────────────────────────────────────────────────────────
    # Blotter Utilities
    # ─────────────────────────────────────────────────────────────────

    def save_blotter_csv(self, output_dir: Optional[str] = None) -> Optional[str]:
        """Save blotter trades to CSV. Returns the file path or None."""
        if self.blotter is None or not self.blotter.trades:
            logger.info("[Backtester] No blotter trades to save.")
            return None
        from pathlib import Path
        out = Path(output_dir) if output_dir else Path(self._reports_directory) / self.strategy_name
        out.mkdir(parents=True, exist_ok=True)
        csv_path = str(out / "blotter_trades.csv")
        self.blotter.save_trades_to_csv(csv_path)
        logger.info(f"[Backtester] Blotter trades saved to {csv_path}")
        return csv_path

    def generate_blotter_plots(self, output_dir: Optional[str] = None) -> None:
        """Blotter plot pack is a Pro reporting feature in the public/light repo."""
        logger.info("[Public] Blotter plot pack is available in QuantJourney Backtester Pro.")

    # ─────────────────────────────────────────────────────────────────
    # Quick Summary
    # ─────────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a quick summary of the backtest results."""
        if self.portfolio_data is None:
            print("No data yet — run run_strategy() first")
            return

        nav = self.portfolio_data.net_asset_value
        ret = self.portfolio_data.returns
        dd = self.portfolio_data.drawdown

        print(f"\n{'=' * 60}")
        print(f"  Strategy: {self.strategy_name}")
        print(f"  Backtester: v{BACKTESTER_VERSION}")
        print(f"  Period:   {nav.index[0].date()} → {nav.index[-1].date()}")
        print(f"  Assets:   {', '.join(self.instruments)}")
        print(f"{'─' * 60}")
        print(f"  Initial NAV:  ${self.initial_capital:>14,.2f}")
        print(f"  Final NAV:    ${nav.iloc[-1]:>14,.2f}")
        print(f"  Total Return: {((nav.iloc[-1] / nav.iloc[0]) - 1) * 100:>13.2f}%")
        print(f"  Ann. Volatility: {ret.std() * np.sqrt(252) * 100:>10.2f}%")
        if dd is not None:
            print(f"  Max Drawdown: {dd.min() * 100:>13.2f}%")
        if self.portfolio_data.sharpe_ratio is not None:
            print(f"  Sharpe Ratio: {float(self.portfolio_data.sharpe_ratio):>13.4f}")
        timings = getattr(self, "_timings", {}) or {}
        if timings:
            print(f"{'─' * 60}")
            print(f"  Runtime Total: {timings.get('total_seconds', 0.0):>12.2f}s")
            print(f"  Data Fetch:    {timings.get('data_fetch_seconds', 0.0):>12.2f}s")
            print(f"  Data Prep:     {timings.get('data_processing_seconds', 0.0):>12.2f}s")
            print(f"  Calculation:   {timings.get('calculation_seconds', 0.0):>12.2f}s")
            if "reporting_seconds" in timings:
                print(f"  Reporting:     {timings.get('reporting_seconds', 0.0):>12.2f}s")
        # Rebalance stats
        if hasattr(self, '_rebalance_stats'):
            rs = self._rebalance_stats
            print(f"{'─' * 60}")
            print(f"  Rebalance Policy: {self._rebalance_policy}")
            print(f"  Rebalances:   {rs['rebalance_count']:>13d}")
            print(f"  Avg Days Btw: {rs['avg_days_between']:>13.1f}")
            if rs.get('drift_count', 0):
                print(f"  Drift Trigr:  {rs['drift_count']:>13d}")
            if rs.get('tracking_error_count', 0):
                print(f"  TE Triggers:  {rs['tracking_error_count']:>13d}")
            if rs.get('circuit_breaker_count', 0):
                print(f"  CB Triggers:  {rs['circuit_breaker_count']:>13d}")
            if rs.get('partial_positions_saved', 0):
                print(f"  Partial Saved:{rs['partial_positions_saved']:>13d}")
        print(f"{'=' * 60}\n")
