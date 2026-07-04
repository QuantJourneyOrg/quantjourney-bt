"""
ReportingMixin — performance analysis, metadata, blotter, summary
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

import json
import logging
import math
import numbers
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from backtester.version import __version__ as BACKTESTER_VERSION

try:
    from backtester.utils.logger import logger
except Exception:
    logger = logging.getLogger("backtester")


class ReportingMixin:
    """Performance analysis, metadata, blotter utilities, and print_summary."""

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
    # Run Metadata
    # ─────────────────────────────────────────────────────────────────

    async def _archive_strategy_data(self) -> None:
        """Write public/light run metadata."""
        try:
            metadata_started = time.perf_counter()
            metadata_folder = Path(self._reports_directory) / self.strategy_name
            metadata_folder.mkdir(parents=True, exist_ok=True)

            metadata = self._build_run_metadata()
            metadata_seconds = time.perf_counter() - metadata_started
            timings = dict(metadata.get("timings_seconds") or {})
            timings["metadata_seconds"] = metadata_seconds
            if "total_before_metadata_seconds" in timings:
                timings["total_seconds"] = timings["total_before_metadata_seconds"] + metadata_seconds
            metadata["timings_seconds"] = timings

            metadata_path = metadata_folder / "run_metadata.json"
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self._json_safe(metadata), f, indent=2, sort_keys=True, default=str, allow_nan=False)
            logger.info(f"Run metadata saved to {metadata_path}")
        except Exception as e:
            logger.warning(f"[Backtester] Metadata save skipped: {e}")
            if getattr(self, "_strict_reporting", False):
                raise

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Convert run metadata to strict JSON-compatible values."""
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return None
            return number
        return value

    def _build_run_metadata(self) -> Dict[str, Any]:
        """Build machine-readable metadata saved next to report results."""
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
