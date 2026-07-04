# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Public/light Strategy Performance Analysis.

This module intentionally reuses the QuantJourney plotting and metric stack
(`PortfolioCalculations`, `PortfolioPlots`, `InstrumentPlots`, theme/compat
helpers) and only limits the scope of what is generated publicly. It does not
ship factsheet generation, narrative generation, walk-forward validation or
optimization.
"""

from __future__ import annotations

import html
import json
import math
import numbers
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtester.metrics import format_metric, generate_report_sections
from backtester.metrics.configs.portfolio_perf import PORTFOLIO_PERF_METRICS
from backtester.plots.plot_compat import C, ensure_style, reset_style
from backtester.plots.theme import PlotTheme, ThemeManager
from backtester.portfolio import (
    InstrumentCalculations,
    InstrumentPlots,
    PortfolioCalculations,
    PortfolioPlots,
)
from backtester.utils.logger import logger


RESULTS_PATH = Path("./reports")


@dataclass
class StrategyPerformanceConfig:
    dpi: int = field(default=300)
    risk_free_rate: float = field(default=0.02)
    confidence_level: float = field(default=0.95)
    lookback_window: int = field(default=252)
    show_text_reports: bool = field(default=True)
    save_text_reports: bool = field(default=True)
    save_portfolio_plots: bool = field(default=False)
    show_portfolio_plots: bool = field(default=False)
    save_instrument_plots: bool = field(default=False)
    show_instrument_plots: bool = field(default=False)
    theme_plots: PlotTheme = field(default=PlotTheme.QUANTJOURNEY)
    reports_directory: Path = field(default_factory=lambda: Path("./reports"))
    benchmark: Dict[str, str] = field(default_factory=lambda: {"symbol": "SPY", "name": "S&P 500 Index"})
    reporting_frequency: str = field(default="daily")

    def __post_init__(self) -> None:
        if isinstance(self.theme_plots, str):
            self.theme_plots = PlotTheme(self.theme_plots)
        self.reports_directory = Path(self.reports_directory)
        if self.dpi < 72:
            raise ValueError("DPI must be at least 72")


class StrategyPerformanceAnalysis:
    """Public/light report generator using QuantJourney's native plot/metric code."""

    def __init__(
        self,
        config: Dict[str, Any],
        portfolio_data: Optional[Any] = None,
        instruments_data: Optional[Any] = None,
        data_connector: Optional[Any] = None,
        strategy_name: str = "default_strategy",
        strategy_type: str = "Long-Short",
        base_currency: str = "USD",
        backtest_period: Optional[Any] = None,
        initial_capital: float = 100_000,
        sdk_client: Optional[Any] = None,
    ) -> None:
        self.config = StrategyPerformanceConfig(**(config or {}))
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.base_currency = base_currency
        self.backtest_period = backtest_period
        self.initial_capital = float(initial_capital)
        self._dc = data_connector
        self._sdk_client = sdk_client

        self.start_date = self._period_value("start")
        self.end_date = self._period_value("end")

        self.portfolio_data = portfolio_data
        self.instruments_data = instruments_data
        self.portfolio_calc: Optional[PortfolioCalculations] = None
        self.instrument_calc: Optional[InstrumentCalculations] = None
        self._performance_results: Dict[str, Any] = {}
        self._plot_paths: list[Path] = []

        self._setup_strategy_folder()
        self._initialize_components()

    def _period_value(self, key: str) -> str:
        if self.backtest_period is None:
            return ""
        if hasattr(self.backtest_period, "get"):
            return str(self.backtest_period.get(key, ""))
        return str(getattr(self.backtest_period, key, ""))

    def _setup_strategy_folder(self) -> None:
        reports_dir = Path(self.config.reports_directory)
        self.save_folder = (RESULTS_PATH if reports_dir == Path("reports") else reports_dir) / self.strategy_name
        self.save_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Strategy folder set to: {self.save_folder}")

    def _initialize_components(self) -> None:
        if self.portfolio_data is not None:
            self.portfolio_calc = PortfolioCalculations(self.portfolio_data)
        if self.instruments_data is not None:
            try:
                self.instrument_calc = InstrumentCalculations(self.instruments_data)
            except Exception as exc:
                logger.info(f"[Public] Instrument analytics skipped: {exc}")
                self.instrument_calc = None

    async def generate_strategy_performance_analysis(
        self,
        portfolio_data: Optional[Any] = None,
        instruments_data: Optional[Any] = None,
        blotter: Optional[Any] = None,
        strategy_parameters: Optional[Dict[str, Any]] = None,
        strategy_code: str = "",
        fill_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        self._blotter = blotter
        self._strategy_parameters = strategy_parameters or {}
        self._strategy_code = strategy_code
        self._fill_engine = fill_engine
        logger.info("Generating public/light strategy performance analysis.")

        if portfolio_data is not None:
            self.portfolio_data = portfolio_data
        if instruments_data is not None:
            self.instruments_data = instruments_data
        self._initialize_components()

        if self.portfolio_calc is None:
            raise ValueError("portfolio_data is required to generate public/light performance analysis")

        ThemeManager.set_theme(self.config.theme_plots)
        try:
            C.apply_theme(ThemeManager.get_current_theme())
            reset_style()
            ensure_style(ThemeManager.get_current_theme())
        except Exception:
            pass

        results = self._compute_public_metric_results()
        self._performance_results = results
        self._write_json("summary.json", results)
        self._write_metrics_csv(results)

        report_text = self._create_public_report_table(results)
        if self.config.save_text_reports:
            report_path = self.save_folder / "performance_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            # Compatibility with the first public-light implementation.
            (self.save_folder / "summary.txt").write_text(report_text, encoding="utf-8")
            logger.info(f"Performance report saved to: {report_path}")
        if self.config.show_text_reports:
            print(report_text)

        self._write_equity_curve_csv()

        if (
            self.config.show_portfolio_plots
            or self.config.save_portfolio_plots
            or self.config.show_instrument_plots
            or self.config.save_instrument_plots
        ):
            plots_folder = self.save_folder / "plots"
            plots_folder.mkdir(parents=True, exist_ok=True)
            self._plot_paths = self._generate_public_plot_pack(plots_folder)
            self._write_dashboard_html(results, self._plot_paths)
            logger.info(f"[Public] Dashboard saved to {self.save_folder / 'dashboard.html'}")
            logger.info(f"[Public] Plot pack saved to {plots_folder} ({len(self._plot_paths)} PNG files)")

        return results

    @property
    def returns(self) -> pd.Series:
        returns = getattr(self.portfolio_data, "returns", None)
        if isinstance(returns, pd.Series):
            return returns
        return self.portfolio_data.net_asset_value.pct_change().fillna(0.0)

    def _safe_metric(self, name: str, func: Callable[[], Any]) -> tuple[str, Any]:
        try:
            return name, func()
        except Exception as exc:
            logger.warning(f"[Public] Metric skipped: {name}: {exc}")
            return name, None

    def _compute_public_metric_results(self) -> Dict[str, Any]:
        pc = self.portfolio_calc
        assert pc is not None

        metric_definitions: Dict[str, Callable[[], Any]] = {
            "compute_annualized_return": lambda: pc.compute_annualized_return(),
            "cumulative_returns": lambda: pc.compute_cumulative_returns(),
            "compute_periodic_returns": lambda: pc.compute_periodic_returns(),
            "compute_period_stats": lambda: pc.compute_period_stats(),
            "compute_sharpe_ratio": lambda: pc.compute_sharpe_ratio(),
            "compute_sortino_ratio": lambda: pc.compute_sortino_ratio(),
            "compute_max_drawdown": lambda: pc.compute_max_drawdown(),
            "compute_recovery_factor": lambda: pc.compute_recovery_factor(),
            "compute_advanced_calmar_ratio": lambda: pc.compute_advanced_calmar_ratio(),
            "compute_advanced_annualized_volatility": lambda: pc.compute_advanced_annualized_volatility(
                short_window=30,
                long_window=252,
            ),
            "compute_advanced_turnover": lambda: pc.compute_advanced_turnover(
                trades_df=self._blotter.get_trades_dataframe() if self._blotter is not None else None,
            ),
        }

        results = {name: value for name, value in (self._safe_metric(name, func) for name, func in metric_definitions.items())}
        results.update(
            {
                "self.start_date": self.start_date,
                "self.end_date": self.end_date,
                "strategy_type": self.strategy_type,
                "base_currency": self.base_currency,
                "risk_free_rate": self.config.risk_free_rate,
                "returns_index": self.returns.index,
                "config": {"risk_free_rate": self.config.risk_free_rate, "reporting_frequency": "daily"},
                "edition": "public_light",
            }
        )
        return results

    def _create_public_report_table(self, results: Dict[str, Any]) -> str:
        try:
            sections = generate_report_sections(results)
        except Exception as exc:
            logger.warning(f"[Public] Structured report sections skipped: {exc}")
            sections = {}

        rows: list[tuple[str, str, str]] = []
        for section_name, metrics in sections.items():
            if not metrics or section_name == "Definitions":
                continue
            metric_paths = [
                PORTFOLIO_PERF_METRICS.get(section_name, {}).get(metric_name, ("", ""))[0].split(".")[0]
                for metric_name in metrics.keys()
            ]
            if metric_paths and not any(results.get(path) is not None for path in metric_paths):
                continue
            first = True
            for metric_name, value in metrics.items():
                if not str(value).strip():
                    continue
                rows.append((section_name if first else "", metric_name, str(value).strip()))
                first = False

        if not rows:
            rows = [
                ("Summary", "Annualized Return", str(results.get("compute_annualized_return", "n/a"))),
                ("", "Sharpe Ratio", str(results.get("compute_sharpe_ratio", "n/a"))),
                ("", "Max Drawdown", str(results.get("compute_max_drawdown", "n/a"))),
            ]

        header = ("Section", "Metric", "Value")
        widths = (
            max(len(header[0]), *(len(row[0]) for row in rows)),
            max(len(header[1]), *(len(row[1]) for row in rows)),
            max(len(header[2]), *(len(row[2]) for row in rows)),
        )
        sep = f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}"
        lines = [
            f"{header[0]:<{widths[0]}}  {header[1]:<{widths[1]}}  {header[2]:>{widths[2]}}",
            sep,
        ]
        lines.extend(f"{section:<{widths[0]}}  {metric:<{widths[1]}}  {value:>{widths[2]}}" for section, metric, value in rows)
        return "\n".join(lines) + "\n"

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        with open(self.save_folder / filename, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(data), f, indent=2, sort_keys=True, default=str, allow_nan=False)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            return str(value)
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return None
            return number
        return value

    def _flatten_metrics(self, data: Any, prefix: str = "") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(data, dict):
            for key, value in data.items():
                rows.extend(self._flatten_metrics(value, f"{prefix}.{key}" if prefix else str(key)))
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            rows.append({"metric": prefix, "value": str(data.tail(5).to_dict())})
        else:
            rows.append({"metric": prefix, "value": data})
        return rows

    def _write_metrics_csv(self, results: Dict[str, Any]) -> None:
        pd.DataFrame(self._public_metric_rows(results)).to_csv(self.save_folder / "metrics.csv", index=False)

    def _write_equity_curve_csv(self) -> None:
        nav = self.portfolio_data.net_asset_value.astype(float)
        returns = self.returns.reindex(nav.index).fillna(0.0)
        drawdown = self.portfolio_calc.drawdowns if self.portfolio_calc is not None else pd.Series(index=nav.index, dtype=float)
        pd.DataFrame(
            {
                "net_asset_value": nav,
                "return": returns,
                "drawdown": drawdown.reindex(nav.index).fillna(0.0),
            }
        ).to_csv(self.save_folder / "equity_curve.csv", index_label="date")
        weights = getattr(self.portfolio_data, "weights", None)
        if isinstance(weights, pd.DataFrame) and not weights.empty:
            weights.to_csv(self.save_folder / "weights.csv", index_label="date")

    def _public_plot_definitions(self) -> list[tuple[str, Callable[[], Any]]]:
        pc = self.portfolio_calc
        ic = self.instrument_calc
        assert pc is not None
        plots: list[tuple[str, Callable[[], Any]]] = [
            ("cumulative_returns", lambda: PortfolioPlots.plot_cumulative_returns(portfolio_calc=pc, instrument_calc=ic)),
            ("cumulative_log_returns", lambda: PortfolioPlots.plot_cumulative_log_returns(portfolio_calc=pc)),
            ("annual_returns", lambda: PortfolioPlots.plot_annual_returns(portfolio_calc=pc)),
            ("monthly_returns_heatmap", lambda: PortfolioPlots.plot_monthly_returns_heatmap(portfolio_calc=pc)),
            ("return_quantiles", lambda: PortfolioPlots.plot_return_quantiles(portfolio_calc=pc)),
            ("portfolio_drawdown", lambda: PortfolioPlots.plot_drawdown(portfolio_calc=pc)),
            ("drawdown_recovery", lambda: PortfolioPlots.plot_drawdown_recovery_analysis(portfolio_calc=pc)),
            ("time_underwater", lambda: PortfolioPlots.plot_time_underwater(portfolio_calc=pc)),
            ("monthly_returns_distribution", lambda: PortfolioPlots.plot_distribution_of_monthly_returns(portfolio_calc=pc)),
            ("rolling_sortino_ratio", lambda: PortfolioPlots.plot_rolling_sortino_ratio(portfolio_calc=pc, window=252)),
            ("rolling_var_cvar", lambda: PortfolioPlots.plot_rolling_var_cvar(portfolio_calc=pc, window=252, confidence=0.95)),
            ("portfolio_weights", lambda: PortfolioPlots.plot_portfolio_weights(portfolio_calc=pc, instrument_calc=ic)),
            ("percentage_weights", lambda: PortfolioPlots.plot_percentage_weights(portfolio_calc=pc, instrument_calc=ic)),
            ("latest_holdings", lambda: PortfolioPlots.plot_composition(portfolio_calc=pc)),
            ("rolling_weights", lambda: PortfolioPlots.plot_rolling_weights(portfolio_calc=pc, instrument_calc=ic)),
        ]
        if ic is not None:
            plots.extend(
                [
                    ("instrument_cumulative_returns", lambda: InstrumentPlots.plot_cumulative_returns(instrument_calc=ic)),
                    ("instrument_rolling_volatility", lambda: InstrumentPlots.plot_rolling_volatility(instrument_calc=ic, window=126)),
                    ("instrument_return_distribution", lambda: InstrumentPlots.plot_return_distribution(instrument_calc=ic)),
                    ("correlation_heatmap", lambda: PortfolioPlots.plot_correlation_heatmap(instrument_calc=ic)),
                    ("correlation_snapshot", lambda: PortfolioPlots.plot_correlation_snapshot(instrument_calc=ic, trailing_window=252)),
                    ("rolling_asset_correlations", lambda: PortfolioPlots.plot_rolling_asset_correlations(instrument_calc=ic, window=126)),
                ]
            )
        return plots

    def _generate_public_plot_pack(self, plots_folder: Path) -> list[Path]:
        reset_style()
        ensure_style(ThemeManager.get_current_theme())
        generated: list[Path] = []
        for name, plotter in self._public_plot_definitions():
            try:
                fig = plotter()
                path = plots_folder / f"{name}.png"
                fig.savefig(path, bbox_inches="tight", dpi=self.config.dpi, facecolor=C.FIG_BG)
                plt.close(fig)
                generated.append(path)
            except Exception as exc:
                logger.warning(f"[Public] Plot skipped: {name}: {exc}")

        cumulative = plots_folder / "cumulative_returns.png"
        if cumulative.exists():
            shutil.copyfile(cumulative, self.save_folder / "equity_curve.png")
        return generated

    def _dashboard_cards(self, results: Dict[str, Any]) -> list[tuple[str, str]]:
        return [
            ("Annualized Return", self._format_public_value(results.get("compute_annualized_return"), "percentage")),
            ("Sharpe", self._format_public_value(results.get("compute_sharpe_ratio"), "ratio")),
            ("Sortino", self._format_public_value(results.get("compute_sortino_ratio"), "ratio")),
            ("Max Drawdown", self._format_public_value(results.get("compute_max_drawdown"), "percentage")),
            ("Calmar", self._format_public_value(self._get_nested(results, "compute_advanced_calmar_ratio.base_calmar"), "ratio")),
            ("Volatility", self._format_public_value(self._get_nested(results, "compute_advanced_annualized_volatility.standard"), "percentage_raw")),
            ("Win Days", self._format_public_value(self._get_nested(results, "compute_period_stats.win_days"), "percentage_raw")),
            ("Avg Turnover", self._format_public_value(self._get_nested(results, "compute_advanced_turnover.average_turnover"), "percentage_raw")),
        ]

    def _public_metric_specs(self) -> list[tuple[str, str, str, str]]:
        return [
            ("Return", "compute_annualized_return", "Annualized return", "percentage"),
            ("Return", "cumulative_returns.total_return", "Total return", "percentage"),
            ("Return", "compute_periodic_returns.statistics.MTD", "MTD", "percentage"),
            ("Return", "compute_periodic_returns.statistics.QTD", "QTD", "percentage"),
            ("Return", "compute_periodic_returns.statistics.YTD", "YTD", "percentage"),
            ("Return", "compute_periodic_returns.statistics.1Y", "1Y", "percentage"),
            ("Return", "compute_periodic_returns.statistics.3Y", "3Y annualized", "percentage"),
            ("Return", "compute_periodic_returns.statistics.5Y", "5Y annualized", "percentage"),
            ("Risk", "compute_sharpe_ratio", "Sharpe ratio", "ratio"),
            ("Risk", "compute_sortino_ratio", "Sortino ratio", "ratio"),
            ("Risk", "compute_max_drawdown", "Max drawdown", "percentage"),
            ("Risk", "compute_advanced_calmar_ratio.base_calmar", "Calmar ratio", "ratio"),
            ("Risk", "compute_recovery_factor", "Recovery factor", "ratio"),
            ("Risk", "compute_advanced_annualized_volatility.standard", "Annualized volatility", "percentage_raw"),
            ("Consistency", "compute_period_stats.win_days", "Winning days", "percentage_raw"),
            ("Consistency", "compute_period_stats.win_month", "Winning months", "percentage_raw"),
            ("Trading", "compute_advanced_turnover.average_turnover", "Average daily turnover", "percentage_raw"),
            ("Trading", "compute_advanced_turnover.annualized_turnover", "Annualized turnover", "percentage_raw"),
            ("Trading", "compute_advanced_turnover.total_traded_notional", "Total traded notional", "currency0"),
        ]

    def _public_metric_rows(self, results: Dict[str, Any]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for section, path, label, formatter_type in self._public_metric_specs():
            value = self._get_nested(results, path) if "." in path else results.get(path)
            rows.append(
                {
                    "section": section,
                    "metric": label,
                    "value": self._format_public_value(value, formatter_type),
                }
            )
        return rows

    @staticmethod
    def _get_nested(data: Dict[str, Any], path: str) -> Any:
        value: Any = data
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    @staticmethod
    def _as_finite_float(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except Exception:
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    @classmethod
    def _format_public_value(cls, value: Any, formatter_type: str) -> str:
        try:
            formatted = format_metric(value, formatter_type)
        except Exception:
            formatted = ""
        if str(formatted).strip():
            return str(formatted).strip()

        number = cls._as_finite_float(value)
        if number is None:
            if value in (None, "") or isinstance(value, numbers.Real):
                return "n/a"
            return str(value)
        if formatter_type == "percentage":
            return f"{number * 100:.2f}%"
        if formatter_type == "percentage_raw":
            return f"{number:.2f}%"
        if formatter_type == "currency":
            return f"${number:,.2f}"
        if formatter_type == "currency0":
            return f"${number:,.0f}"
        if formatter_type == "integer":
            return f"{number:,.0f}"
        if formatter_type.startswith("ratio"):
            return f"{number:.2f}"
        return str(value)

    def _write_dashboard_html(self, results: Dict[str, Any], plot_paths: list[Path]) -> None:
        cards = "\n".join(
            f"<div class='card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
            for label, value in self._dashboard_cards(results)
        )
        metrics_rows = "\n".join(
            f"<tr><td>{html.escape(row['section'])}</td><td>{html.escape(row['metric'])}</td><td>{html.escape(row['value'])}</td></tr>"
            for row in self._public_metric_rows(results)
        )
        figures = "\n".join(
            f"<figure><img src='plots/{html.escape(path.name)}' alt='{html.escape(path.stem)}'><figcaption>{html.escape(path.stem.replace('_', ' ').title())}</figcaption></figure>"
            for path in plot_paths
        )
        content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(self.strategy_name)} Dashboard</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #101828; background: #f6f7f9; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    p {{ margin: 0; color: #475467; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ background: #fff; border: 1px solid #eaecf0; border-radius: 8px; padding: 14px 16px; }}
    .card span {{ display: block; color: #667085; font-size: 13px; margin-bottom: 6px; }}
    .card strong {{ display: block; color: #155EEF; font-size: 22px; }}
    .table-wrap {{ margin: 24px 0; background: #fff; border: 1px solid #eaecf0; border-radius: 8px; overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #eaecf0; text-align: left; vertical-align: top; }}
    th {{ background: #f2f4f7; color: #344054; }}
    td:last-child {{ color: #155EEF; font-weight: 650; }}
    .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #eaecf0; border-radius: 8px; padding: 12px; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ color: #475467; font-size: 13px; margin-top: 8px; }}
    @media (max-width: 520px) {{ .plots {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(self.strategy_name)}</h1>
      <p>{html.escape(self.strategy_type)} | {html.escape(self.start_date)} to {html.escape(self.end_date)} | QuantJourney Public Light Dashboard</p>
    </header>
    <section class="cards">{cards}</section>
    <section class="table-wrap">
      <table><thead><tr><th>Section</th><th>Metric</th><th>Value</th></tr></thead><tbody>{metrics_rows}</tbody></table>
    </section>
    <section class="plots">{figures}</section>
  </main>
</body>
</html>
"""
        (self.save_folder / "dashboard.html").write_text(content, encoding="utf-8")
