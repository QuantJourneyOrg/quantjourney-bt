"""
PortfolioPlots - Institutional-Quality Portfolio Visualisation Suite
====================================================================

Unified plotting module for portfolio analytics. Every chart follows the
QuantJourney institutional style system defined in ``plot_compat.py``.

All public methods are ``@staticmethod`` and return ``plt.Figure``.

Style contract
--------------
1. ``ensure_style()`` is called at the top of every function.
2. Figures use ``plt.subplots(figsize=(11.5, 6.2))``.
3. All colours come from ``C.*`` constants.
4. ``style_ax`` / ``smart_date_axis`` / ``add_watermark`` are applied.
5. ``endpoint_annotation`` marks the final value on time-series plots.
6. Legends use ``Line2D`` handles (no leaking artists).
7. ``fig.tight_layout()`` is called before ``add_watermark``.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import calendar
from typing import Optional, Tuple, List, Dict
from scipy import stats
from scipy.optimize import minimize
from matplotlib.lines import Line2D

from backtester.portfolio.portf_calc import PortfolioCalculations
from backtester.portfolio.instr_calc import InstrumentCalculations
from backtester.plots.plot_compat import (
    C, ensure_style, add_watermark, style_ax, smart_date_axis,
    endpoint_annotation, endpoint_annotations_pair,
    stats_box as _stats_box, fmt_pct, fmt_ratio, make_figure, diverging_cmap,
)
from backtester.utils.logger import logger


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _safe_float(v) -> float:
    """Convert a scalar / 0-d array / single-element Series to plain float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(np.nanmean(np.asarray(v)))


    # _stats_box is now imported from plot_compat as _stats_box


def _clean_label(label) -> str:
    """Strip instrument suffix (e.g. 'AAPL-equity' -> 'AAPL')."""
    s = label[0] if isinstance(label, tuple) else str(label)
    return s.split("-")[0]


_ALLOCATION_COLORS = [
    "#2563EB",  # vivid blue
    "#F2A900",  # gold
    "#00A6A6",  # teal
    "#E11D48",  # rose
    "#10B981",  # emerald
    "#0EA5E9",  # sky
    "#F97316",  # orange
    "#0F2F4A",  # ink navy
    "#4A9BD9",  # steel blue
    "#27864B",  # green
]
_ALLOCATION_COLOR_BY_LABEL = {
    "AAPL": "#2563EB",
    "AMZN": "#F2A900",
    "GOOGL": "#00A6A6",
    "GOOG": "#00A6A6",
    "MSFT": "#E11D48",
    "NVDA": "#10B981",
    "META": "#0EA5E9",
    "TSLA": "#F97316",
}
_CASH_COLOR = "#AFC3D8"


def _asset_color(label, *, asset_alpha: float = 0.92, cash_alpha: float = 0.38) -> tuple:
    """Stable colour per instrument label, independent of plotting order."""
    clean = _clean_label(label).upper()
    if clean == "CASH":
        return mcolors.to_rgba(_CASH_COLOR, cash_alpha)
    if clean in {"OTHER", "OTHERS"}:
        return mcolors.to_rgba(C.MUTED, 0.58)

    base = _ALLOCATION_COLOR_BY_LABEL.get(clean)
    if base is None:
        idx = sum(ord(ch) for ch in clean) % len(_ALLOCATION_COLORS)
        base = _ALLOCATION_COLORS[idx]
    return mcolors.to_rgba(base, asset_alpha)


def _allocation_colors(
    labels: List[str],
    *,
    asset_alpha: float = 0.88,
    cash_alpha: float = 0.62,
) -> List[tuple]:
    """Return high-contrast allocation colours with neutral cash."""
    return [_asset_color(label, asset_alpha=asset_alpha, cash_alpha=cash_alpha)
            for label in labels]


def _gradient_fill(ax, x, y, baseline, color, alpha_max=0.25):
    """Add a simple vertical gradient fill_between."""
    ax.fill_between(x, y, baseline, color=color, alpha=alpha_max, zorder=1)


def _true_runs(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Return contiguous true runs as (start, end, length)."""
    if mask is None or len(mask) == 0:
        return []
    m = pd.Series(mask).fillna(False).astype(bool)
    runs: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = None
    length = 0
    prev_idx = None
    for idx, val in m.items():
        if val:
            if start is None:
                start = idx
                length = 0
            length += 1
        elif start is not None:
            runs.append((start, prev_idx, length))
            start = None
            length = 0
        prev_idx = idx
    if start is not None:
        runs.append((start, prev_idx, length))
    return runs


_REGIME_COLORS = {
    "Bull": mcolors.to_rgba(C.GREEN, 0.22),
    "Sideways": mcolors.to_rgba(C.BENCHMARK, 0.16),
    "Bear": mcolors.to_rgba(C.RED, 0.24),
}


def _infer_periods_per_year(index: pd.Index) -> int:
    """Approximate observations per year from a DateTimeIndex."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252
    deltas = pd.Series(index).diff().dropna().dt.total_seconds() / 86400.0
    median_days = float(deltas.median())
    if median_days <= 3.5:
        return 252
    if median_days <= 10.5:
        return 52
    if median_days <= 45.0:
        return 12
    return 4


def _benchmark_regime_series(
    benchmark_returns: pd.Series,
    target_index: Optional[pd.Index] = None,
    smooth_window: int = 21,
) -> pd.Series:
    """Classify benchmark trend into Bull / Sideways / Bear regimes."""
    bench = benchmark_returns.dropna()
    bench_index = (1 + bench).cumprod()
    ppy = _infer_periods_per_year(bench_index.index)
    trend_window = max(4, min(200, int(round(ppy * 0.8))))
    trend_min = max(2, int(round(trend_window * 0.6)))
    momentum_window = max(2, min(63, int(round(ppy * 0.25))))
    smooth_window = max(1, min(smooth_window, max(2, ppy // 12))) if ppy < 252 else smooth_window
    sma200 = bench_index.rolling(trend_window, min_periods=trend_min).mean()
    ret63 = bench_index.pct_change(momentum_window)

    codes = pd.Series(0, index=bench_index.index, dtype=float)
    codes[(bench_index > sma200) & (ret63 > 0.03)] = 1
    codes[(bench_index < sma200) & (ret63 < -0.03)] = -1
    if smooth_window and smooth_window > 1:
        min_periods = min(smooth_window, max(1, smooth_window // 3))
        codes = codes.rolling(smooth_window, min_periods=min_periods).median().round().fillna(codes)
    regime = codes.astype(int).map({1: "Bull", 0: "Sideways", -1: "Bear"})
    if target_index is not None:
        regime = regime.reindex(target_index).ffill()
    return regime.dropna()


def _shade_regimes(
    ax: plt.Axes,
    regime: pd.Series,
    min_run: int = 21,
    *,
    full_height: bool = False,
) -> None:
    """Add benchmark-regime shading as full-height background or compact band."""
    if regime is None or regime.empty:
        return
    ymin, ymax = (0.0, 1.0) if full_height else (0.0, 0.045)
    for start, end, label in _regime_runs(regime):
        if len(regime.loc[start:end]) < min_run:
            continue
        color = _REGIME_COLORS.get(label, (0, 0, 0, 0))
        if full_height:
            r, g, b, a = color
            color = (r, g, b, min(a, 0.085))
        ax.axvspan(start, end, color=color,
                   ymin=ymin, ymax=ymax, lw=0, zorder=0)


def _regime_summary_text(regime: pd.Series, benchmark_name: str = "Benchmark") -> str:
    """Compact regime share summary for subtitles."""
    if regime is None or regime.empty:
        return f"Regime source: {benchmark_name}"
    counts = regime.value_counts(normalize=True) * 100
    return (
        f"{benchmark_name} regimes: "
        f"Bull {counts.get('Bull', 0):.0f}% | "
        f"Sideways {counts.get('Sideways', 0):.0f}% | "
        f"Bear {counts.get('Bear', 0):.0f}%"
    )


def _regime_runs(regime: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return contiguous regime runs as (start, end, label)."""
    clean = regime.dropna()
    if clean.empty:
        return []
    runs: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    start = clean.index[0]
    current = str(clean.iloc[0])
    prev = clean.index[0]
    for idx, value in clean.iloc[1:].items():
        label = str(value)
        if label != current:
            runs.append((start, prev, current))
            start = idx
            current = label
        prev = idx
    runs.append((start, prev, current))
    return runs


def _regime_legend_handles() -> List[mpatches.Patch]:
    """Legend handles for benchmark-regime background shading."""
    return [
        mpatches.Patch(facecolor=_REGIME_COLORS["Bull"], edgecolor="none", label="Bull Regime"),
        mpatches.Patch(facecolor=_REGIME_COLORS["Sideways"], edgecolor="none", label="Sideways"),
        mpatches.Patch(facecolor=_REGIME_COLORS["Bear"], edgecolor="none", label="Bear Regime"),
    ]


# ============================================================================
# PortfolioPlots
# ============================================================================

class PortfolioPlots:
    """Static methods producing institutional-quality portfolio charts."""

    # ------------------------------------------------------------------ 1
    @staticmethod
    def plot_cumulative_returns(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: Optional[InstrumentCalculations] = None,
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_name: str = "S&P 500",
    ) -> plt.Figure:
        """Plot portfolio cumulative returns with optional benchmark overlay.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
            Portfolio analytics facade.
        instrument_calc : InstrumentCalculations, optional
            If provided the first instrument is treated as a benchmark and
            overlaid with an orange dashed line.
        benchmark_returns : pd.Series, optional
            Daily benchmark returns. If provided, overlaid as a dashed line.
        benchmark_name : str
            Display name for the benchmark in the legend.

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        cum = (1 + returns).cumprod()

        ax.plot(cum.index, cum.values, color=C.BLUE, linewidth=C.LW_MAIN, zorder=4)
        ax.axhline(1.0, color=C.SPINE, lw=C.LW_HAIR, ls=":", alpha=0.65, zorder=1)

        # Subtitle stats
        total_ret = (cum.iloc[-1] / cum.iloc[0] - 1) * 100
        n_years = max((cum.index[-1] - cum.index[0]).days / 365.25, 1e-6)
        ann_ret = ((cum.iloc[-1] / cum.iloc[0]) ** (1 / n_years) - 1) * 100
        ann_vol = returns.std() * np.sqrt(252) * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = (cum / cum.cummax() - 1)
        max_dd = dd.min() * 100
        subtitle = (
            f"Total: {total_ret:+.1f}%  |  "
            f"Ann.: {ann_ret:+.1f}%  |  "
            f"Sharpe: {sharpe:.2f}  |  "
            f"Max DD: {max_dd:.1f}%"
        )

        handles = [Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Portfolio")]

        # Benchmark from returns series (preferred)
        if benchmark_returns is not None:
            try:
                bench_ret = benchmark_returns.reindex(cum.index).fillna(0)
                bench_cum = (1 + bench_ret).cumprod()
                ax.plot(bench_cum.index, bench_cum.values, color=C.BENCHMARK,
                        linewidth=C.LW_SECONDARY, linestyle=C.BENCHMARK_LS, alpha=0.90, zorder=3)
                endpoint_annotation(ax, bench_cum, benchmark_name, C.BENCHMARK,
                                    fmt="ratio", offset=(8, -14))
                handles.append(
                    Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                           label=benchmark_name)
                )
            except Exception:
                pass
        # Fallback: benchmark from instrument_calc
        elif instrument_calc is not None:
            try:
                bench_ret = instrument_calc.returns.iloc[:, 0].dropna()
                bench_cum = (1 + bench_ret).cumprod()
                ax.plot(bench_cum.index, bench_cum.values, color=C.BENCHMARK,
                        linewidth=C.LW_SECONDARY, linestyle=C.BENCHMARK_LS, alpha=0.90, zorder=3)
                endpoint_annotation(ax, bench_cum, "Benchmark", C.BENCHMARK,
                                    fmt="ratio", offset=(8, -14))
                handles.append(
                    Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                           label="Benchmark")
                )
            except Exception:
                pass

        endpoint_annotation(ax, cum, "Portfolio", C.BLUE, fmt="ratio")

        style_ax(ax, title="Cumulative Returns", ylabel="Growth of $1",
                 subtitle=subtitle)
        smart_date_axis(ax, cum)
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 1b
    @staticmethod
    def plot_cumulative_returns_with_regime(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        benchmark_name: str = "Benchmark",
    ) -> plt.Figure:
        """Cumulative returns with benchmark bull/sideways/bear shading."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        bench_ret = benchmark_returns.dropna()
        returns, bench_ret = returns.align(bench_ret, join="inner")
        if len(returns) < 252:
            raise ValueError("Insufficient data for regime cumulative returns.")

        regime = _benchmark_regime_series(bench_ret, returns.index)
        _shade_regimes(ax, regime, full_height=True)

        cum = (1 + returns).cumprod()
        bench_cum = (1 + bench_ret.reindex(cum.index).fillna(0)).cumprod()

        ax.plot(cum.index, cum.values, color=C.BLUE, linewidth=C.LW_MAIN,
                zorder=4)
        ax.plot(bench_cum.index, bench_cum.values, color=C.BENCHMARK,
                linewidth=C.LW_SECONDARY, linestyle=C.BENCHMARK_LS,
                alpha=0.92, zorder=3)
        ax.axhline(1.0, color=C.SPINE, lw=C.LW_HAIR, ls=":", alpha=0.65, zorder=1)

        endpoint_annotation(ax, cum, "Portfolio", C.BLUE, fmt="ratio")
        endpoint_annotation(ax, bench_cum, benchmark_name, C.BENCHMARK,
                            fmt="ratio", offset=(8, -14))

        subtitle = _regime_summary_text(regime, benchmark_name)
        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Portfolio"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN,
                   ls=C.BENCHMARK_LS, label=benchmark_name),
            *_regime_legend_handles(),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK, ncol=2)

        style_ax(ax, title="Cumulative Returns with Benchmark Regimes",
                 ylabel="Growth of $1", subtitle=subtitle)
        smart_date_axis(ax, cum)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 2
    @staticmethod
    def plot_cumulative_log_returns(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Plot cumulative log returns on a log-scale y-axis.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        log_ret = np.log1p(returns)
        cum_log = log_ret.cumsum()

        ax.plot(cum_log.index, cum_log.values, color=C.BLUE,
                linewidth=C.LW_MAIN, zorder=4)
        ax.fill_between(cum_log.index, cum_log.values, 0.0,
                        color=C.BLUE, alpha=0.055, zorder=1)
        ax.axhline(0.0, color=C.SPINE, lw=C.LW_HAIR, ls=":", alpha=0.65, zorder=1)

        total_log = cum_log.iloc[-1] * 100
        subtitle = f"Total Log Return: {total_log:+.1f}%"

        endpoint_annotation(ax, cum_log, "Log Return", C.BLUE, fmt="ratio")

        style_ax(ax, title="Cumulative Log Returns", ylabel="Log Return",
                 subtitle=subtitle)
        smart_date_axis(ax, cum_log)

        handles = [Line2D([], [], color=C.BLUE, lw=C.LW_MAIN,
                          label="Cumulative Log Returns")]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 3
    @staticmethod
    def plot_nav_composition(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: InstrumentCalculations,
    ) -> plt.Figure:
        """Stacked area chart of per-instrument NAV.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        instrument_calc : InstrumentCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        # Build per-instrument dollar value from positions × adjusted close.
        # Cash is added as the residual so the stacked total reconciles to
        # portfolio_data.net_asset_value.
        nav_df = pd.DataFrame()
        try:
            nav = portfolio_calc.portfolio_data.net_asset_value
            positions = portfolio_calc.portfolio_data.positions
            prices = instrument_calc.prices.xs("adj_close", axis=1, level=1)

            if isinstance(positions, pd.Series):
                positions = positions.to_frame().T

            if isinstance(nav, pd.Series) and isinstance(positions, pd.DataFrame):
                common_cols = [c for c in positions.columns if c in prices.columns]
                if common_cols:
                    pos_aligned = (
                        positions.reindex(index=nav.index, method="ffill")
                        .reindex(columns=common_cols)
                        .fillna(0)
                    )
                    prices_aligned = (
                        prices.reindex(index=nav.index, method="ffill")
                        .reindex(columns=common_cols)
                        .ffill()
                    )
                    nav_df = pos_aligned.multiply(prices_aligned).fillna(0)

                    cash = nav - nav_df.sum(axis=1)
                    if cash.abs().max() > max(float(nav.abs().max()) * 1e-6, 1e-6):
                        nav_df["Cash"] = cash
        except Exception:
            pass

        # Fallback: try compute_cumulative_pnl or compute_nav
        if nav_df is None or nav_df.empty:
            for method_name in ("compute_cumulative_pnl", "compute_nav", "compute_exposures"):
                try:
                    fn = getattr(instrument_calc, method_name, None)
                    if fn is None:
                        continue
                    if method_name == "compute_exposures":
                        nav_df = fn(add_total=False)
                    else:
                        nav_df = fn()
                    if nav_df is not None and not nav_df.empty:
                        # Check if values are meaningful (not all near zero)
                        total = nav_df.sum(axis=1)
                        if total.abs().max() > 1e-6:
                            break
                        nav_df = pd.DataFrame()
                except Exception:
                    continue

        if nav_df is None or nav_df.empty:
            ax.text(0.5, 0.5, "No NAV data available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color=C.MUTED)
            style_ax(ax, title="NAV Composition")
            fig.tight_layout()
            add_watermark(fig)
            return fig

        # Clean data
        nav_df = nav_df.fillna(0).replace([np.inf, -np.inf], 0)

        labels = [_clean_label(c) for c in nav_df.columns]
        colors = _allocation_colors(labels, asset_alpha=0.92, cash_alpha=0.64)

        ax.stackplot(nav_df.index, *[nav_df.iloc[:, i] for i in range(nav_df.shape[1])],
                     labels=labels, colors=colors, alpha=1.0, linewidth=0)

        # Total NAV line
        total_nav = nav_df.sum(axis=1)
        ax.plot(total_nav.index, total_nav.values, color=C.TITLE, lw=0.9,
                ls="--", alpha=0.48, zorder=5)

        # Stats box
        first_nonzero = total_nav[total_nav != 0]
        if len(first_nonzero) >= 2:
            total_ret = (first_nonzero.iloc[-1] / first_nonzero.iloc[0] - 1) * 100
            pct_changes = total_nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            avg_vol = pct_changes.std() * np.sqrt(252) * 100 if len(pct_changes) > 1 else 0
            dd = total_nav / total_nav.cummax() - 1
            max_dd = dd.min() * 100
            stats_text = (
                f"Total Return: {total_ret:+.1f}%\n"
                f"Avg Vol (ann.): {avg_vol:.1f}%\n"
                f"Max DD: {max_dd:.1f}%\n"
                f"Current: ${total_nav.iloc[-1]:,.2f}"
            )
            _stats_box(ax, stats_text, loc="upper left")

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        style_ax(ax, title="NAV Composition", ylabel="NAV ($)")
        smart_date_axis(ax, nav_df)

        handles = [mpatches.Patch(facecolor=colors[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95, fontsize=9.5, edgecolor=C.SPINE, borderpad=0.5)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 4
    @staticmethod
    def plot_drawdown(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Drawdown chart with severity zones, max-DD annotation and 30-day MA.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        dd = portfolio_calc.compute_drawdowns().dropna()
        dd_pct = dd * 100  # convert to %

        # Fill with severity colouring
        ax.fill_between(dd_pct.index, dd_pct.values, 0,
                        color=C.RED, alpha=0.14, zorder=2)
        ax.plot(dd_pct.index, dd_pct.values, color=C.RED, lw=C.LW_SECONDARY - 0.1, zorder=3)

        # 30-day MA
        ma30 = dd_pct.rolling(30, min_periods=1).mean()
        ax.plot(ma30.index, ma30.values, color=C.BENCHMARK, ls=C.BENCHMARK_LS, lw=C.LW_SECONDARY,
                zorder=4)

        # Max DD annotation
        max_dd_val = dd_pct.min()
        max_dd_date = dd_pct.idxmin()
        ax.plot(max_dd_date, max_dd_val, "o", color=C.RED, markersize=C.MARKER_SM,
                zorder=6)
        ax.annotate(
            f"Max DD: {max_dd_val:.1f}%\n{max_dd_date.strftime('%Y-%m-%d')}",
            xy=(max_dd_date, max_dd_val),
            xytext=(15, -10), textcoords="offset points",
            fontsize=C.FONT_ANNOT - 1, fontweight="bold", color=C.RED,
            arrowprops=dict(arrowstyle="->", color=C.RED, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc=C.FIG_BG, ec=C.RED,
                      alpha=0.9, lw=0.6),
        )

        # Severity zone lines
        for level, col in [(-5, C.DD_LIGHT), (-10, C.DD_MED),
                           (-20, C.DD_HEAVY)]:
            if max_dd_val < level:
                ax.axhline(level, color=col, ls=":", lw=C.LW_HAIR + 0.2, alpha=0.6, zorder=1)
                ax.text(dd_pct.index[-1], level, f" {level}%", fontsize=C.FONT_SMALL,
                        color=col, va="center", ha="left")

        # Stats box
        time_in_dd = (dd_pct < -1).mean() * 100  # % time in >1% DD
        avg_dd = dd_pct[dd_pct < 0].mean() if (dd_pct < 0).any() else 0
        stats_text = (
            f"Time in DD (>1%): {time_in_dd:.0f}%\n"
            f"Avg DD when negative: {avg_dd:.1f}%"
        )
        _stats_box(ax, stats_text, loc="lower right")

        handles = [
            Line2D([], [], color=C.RED, lw=C.LW_THIN, label="Drawdown"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS, label="30-day MA"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=False,
                  fontsize=C.FONT_TICK)

        subtitle = f"Max Drawdown: {max_dd_val:.1f}%  |  Current: {dd_pct.iloc[-1]:.1f}%"
        style_ax(ax, title="Portfolio Drawdown", ylabel="Drawdown (%)",
                 subtitle=subtitle)
        smart_date_axis(ax, dd_pct)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 4a
    @staticmethod
    def plot_drawdown_with_regime(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        benchmark_name: str = "Benchmark",
    ) -> plt.Figure:
        """Portfolio drawdown with benchmark bull/sideways/bear shading."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        dd = portfolio_calc.compute_drawdowns().dropna() * 100
        bench = benchmark_returns.dropna()
        dd, bench = dd.align(bench, join="inner")
        if len(dd) < 252:
            raise ValueError("Insufficient data for regime drawdown plot.")

        regime = _benchmark_regime_series(bench, dd.index)
        _shade_regimes(ax, regime, full_height=True)

        ax.fill_between(dd.index, dd.values, 0, color=C.RED, alpha=0.14, zorder=2)
        ax.plot(dd.index, dd.values, color=C.RED,
                lw=C.LW_SECONDARY - 0.1, zorder=3)
        ax.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)

        max_dd_val = float(dd.min())
        max_dd_date = dd.idxmin()
        ax.plot(max_dd_date, max_dd_val, "o", color=C.RED,
                markersize=C.MARKER_SM, zorder=6)
        ax.annotate(
            f"Max DD: {max_dd_val:.1f}%\n{max_dd_date.strftime('%Y-%m-%d')}",
            xy=(max_dd_date, max_dd_val),
            xytext=(15, -10), textcoords="offset points",
            fontsize=C.FONT_ANNOT - 1, fontweight="bold", color=C.RED,
            arrowprops=dict(arrowstyle="->", color=C.RED, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc=C.FIG_BG, ec=C.RED,
                      alpha=0.9, lw=0.6),
        )

        bear_days = int((regime == "Bear").sum())
        bear_avg_dd = dd[regime == "Bear"].mean() if bear_days else np.nan
        stats_text = (
            f"Max DD: {max_dd_val:.1f}%\n"
            f"Current DD: {dd.iloc[-1]:.1f}%\n"
            f"Bear Days: {bear_days}\n"
            f"Avg DD in Bear: {bear_avg_dd:.1f}%"
            if np.isfinite(bear_avg_dd)
            else
            f"Max DD: {max_dd_val:.1f}%\n"
            f"Current DD: {dd.iloc[-1]:.1f}%\n"
            f"Bear Days: {bear_days}\n"
            "Avg DD in Bear: n/a"
        )
        _stats_box(ax, stats_text, loc="lower right")

        handles = [
            Line2D([], [], color=C.RED, lw=C.LW_MAIN, label="Drawdown"),
            *_regime_legend_handles(),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=False,
                  fontsize=C.FONT_TICK, ncol=2)

        style_ax(ax, title="Portfolio Drawdown with Benchmark Regimes",
                 ylabel="Drawdown (%)",
                 subtitle=_regime_summary_text(regime, benchmark_name))
        smart_date_axis(ax, dd)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 4b
    @staticmethod
    def plot_drawdown_recovery_analysis(
        portfolio_calc: PortfolioCalculations,
        max_events: int = 12,
    ) -> plt.Figure:
        """Time-to-recovery by drawdown event."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(12.2, 6.6))

        returns = portfolio_calc.returns.dropna()
        wealth = (1 + returns).cumprod()
        dd = wealth / wealth.cummax() - 1

        events: List[Dict[str, object]] = []
        in_dd = False
        start = trough = None
        trough_dd = 0.0
        peak_date = None
        current_peak_date = wealth.index[0] if len(wealth) else None

        for dt, val in dd.items():
            if val >= -1e-10:
                current_peak_date = dt
                if in_dd:
                    recovery = dt
                    window = dd.loc[start:recovery]
                    events.append({
                        "start": start,
                        "peak": peak_date,
                        "trough": trough,
                        "recovery": recovery,
                        "max_dd": trough_dd,
                        "days": len(window),
                        "open": False,
                    })
                    in_dd = False
                    start = trough = None
                    trough_dd = 0.0
                continue

            if not in_dd:
                in_dd = True
                start = dt
                peak_date = current_peak_date
                trough = dt
                trough_dd = float(val)
            elif val < trough_dd:
                trough = dt
                trough_dd = float(val)

        if in_dd and start is not None:
            window = dd.loc[start:]
            events.append({
                "start": start,
                "peak": peak_date,
                "trough": trough,
                "recovery": None,
                "max_dd": trough_dd,
                "days": len(window),
                "open": True,
            })

        # Remove tiny one-day noise; keep open events even if shallow.
        filtered = [
            e for e in events
            if e["open"] or abs(float(e["max_dd"])) >= 0.005 or int(e["days"]) >= 5
        ]
        if not filtered:
            ax.text(0.5, 0.5, "No meaningful drawdown events",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color=C.MUTED)
            style_ax(ax, title="Drawdown Recovery Analysis")
            fig.tight_layout()
            add_watermark(fig)
            return fig

        selected = sorted(
            filtered,
            key=lambda e: (int(e["days"]), abs(float(e["max_dd"]))),
            reverse=True,
        )[:max_events]
        selected = sorted(selected, key=lambda e: e["start"])

        labels = [
            f"{pd.Timestamp(e['start']):%Y-%m} -> "
            f"{'Open' if e['open'] else pd.Timestamp(e['recovery']):%Y-%m}"
            if not e["open"] else f"{pd.Timestamp(e['start']):%Y-%m} -> Open"
            for e in selected
        ]
        days = [int(e["days"]) for e in selected]
        depths = [float(e["max_dd"]) * 100 for e in selected]
        y = np.arange(len(selected))

        colors = [
            mcolors.to_rgba(C.RED, min(0.88, 0.38 + abs(d) / 45))
            for d in depths
        ]
        bars = ax.barh(y, days, color=colors, edgecolor=C.EDGE,
                       linewidth=C.LW_EDGE, height=0.62, zorder=3)

        for bar, e, depth in zip(bars, selected, depths):
            label = f"{int(e['days'])}d | {depth:.1f}%"
            if e["open"]:
                label += " open"
                bar.set_hatch("//")
            ax.text(bar.get_width() + max(days) * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left",
                    fontsize=C.FONT_ANNOT, color=C.LABEL)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=C.FONT_TICK)
        ax.invert_yaxis()

        completed = [e for e in filtered if not e["open"]]
        med_days = np.median([int(e["days"]) for e in completed]) if completed else np.nan
        max_days = max(days)
        open_count = sum(1 for e in filtered if e["open"])
        subtitle = (
            f"Events: {len(filtered)} | Median completed recovery: "
            f"{med_days:.0f}d | Longest shown: {max_days}d | Open: {open_count}"
        )

        style_ax(ax, title="Drawdown Recovery Analysis",
                 xlabel="Trading Days Underwater", ylabel="",
                 subtitle=subtitle)
        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 4c
    @staticmethod
    def plot_time_underwater(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Running number of trading days since the last equity high."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        wealth = (1 + returns).cumprod()
        if wealth.empty:
            ax.text(0.5, 0.5, "No portfolio data available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color=C.MUTED)
            style_ax(ax, title="Time Under Water")
            fig.tight_layout()
            add_watermark(fig)
            return fig

        hwm = wealth.cummax()
        underwater = wealth < hwm * (1 - 1e-10)
        tuw_values = []
        count = 0
        for is_underwater in underwater:
            count = count + 1 if is_underwater else 0
            tuw_values.append(count)
        tuw = pd.Series(tuw_values, index=wealth.index, dtype=float)

        ax.fill_between(tuw.index, tuw.values, 0, color=C.BLUE,
                        alpha=0.10, zorder=1)
        ax.plot(tuw.index, tuw.values, color=C.BLUE,
                lw=C.LW_MAIN, zorder=3, label="Time Under Water")

        max_tuw = int(tuw.max())
        current_tuw = int(tuw.iloc[-1])
        underwater_share = float((tuw > 0).mean() * 100)
        run_lengths = [length for _start, _end, length in _true_runs(tuw > 0)]
        median_run = np.median(run_lengths) if run_lengths else np.nan

        max_date = tuw.idxmax()
        ax.plot(max_date, max_tuw, "o", color=C.BLUE, markersize=C.MARKER_SM,
                zorder=5)
        ax.annotate(
            f"Longest: {max_tuw}d",
            xy=(max_date, max_tuw),
            xytext=(12, 10), textcoords="offset points",
            fontsize=C.FONT_ANNOT, fontweight="bold", color=C.BLUE,
            bbox=dict(boxstyle="round,pad=0.22", fc=C.FIG_BG,
                      ec="none", alpha=0.76, lw=0),
        )

        if np.isfinite(median_run):
            stats_text = (
                f"Longest: {max_tuw}d\n"
                f"Current: {current_tuw}d\n"
                f"Time underwater: {underwater_share:.0f}%\n"
                f"Median run: {median_run:.0f}d"
            )
        else:
            stats_text = (
                f"Longest: {max_tuw}d\n"
                f"Current: {current_tuw}d\n"
                f"Time underwater: {underwater_share:.0f}%\n"
                "Median run: n/a"
            )
        _stats_box(ax, stats_text, loc="upper left")

        ax.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}d"))
        style_ax(ax, title="Running Time Under Water",
                 ylabel="Trading Days Since Last High",
                 subtitle="Continuous recovery-duration clock, resets at each new equity high")
        smart_date_axis(ax, tuw)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 4d
    @staticmethod
    def plot_rolling_var_cvar(
        portfolio_calc: PortfolioCalculations,
        window: int = 252,
        confidence: float = 0.95,
        reporting_label: str = "daily",
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_name: str = "Benchmark",
        show_regime: bool = False,
    ) -> plt.Figure:
        """Rolling historical VaR and CVaR/Expected Shortfall."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        active = returns[returns.abs() > 1e-12]
        if not active.empty:
            returns = returns.loc[active.index[0]:]
        tail_q = 1.0 - confidence
        var = returns.rolling(window, min_periods=window).quantile(tail_q)

        def _cvar(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return np.nan
            q = np.nanquantile(x, tail_q)
            tail = x[x <= q]
            return float(np.nanmean(tail)) if len(tail) else np.nan

        cvar = returns.rolling(window, min_periods=window).apply(_cvar, raw=True)
        risk = pd.DataFrame({"VaR": var, "CVaR": cvar}).dropna() * 100

        if risk.empty:
            raise ValueError("No rolling VaR/CVaR values calculated.")

        regime = None
        if show_regime and benchmark_returns is not None:
            regime = _benchmark_regime_series(benchmark_returns.dropna(), risk.index)
            _shade_regimes(ax, regime, full_height=True)

        ax.fill_between(risk.index, risk["CVaR"].values, 0,
                        color=C.RED, alpha=0.10, zorder=1)
        ax.plot(risk.index, risk["VaR"].values, color=C.RED,
                lw=C.LW_MAIN, zorder=4, label=f"VaR {confidence:.0%}")
        ax.plot(risk.index, risk["CVaR"].values, color=C.DARK,
                lw=C.LW_SECONDARY, ls="--", zorder=5,
                label=f"CVaR {confidence:.0%}")
        ax.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)

        endpoint_annotation(ax, risk["VaR"], "VaR", C.RED, fmt="pct")
        endpoint_annotation(ax, risk["CVaR"], "CVaR", C.DARK, fmt="pct", offset=(8, -18))

        handles = [
            Line2D([], [], color=C.RED, lw=C.LW_MAIN, label=f"VaR {confidence:.0%}"),
            Line2D([], [], color=C.DARK, lw=C.LW_SECONDARY, ls="--",
                   label=f"CVaR {confidence:.0%}"),
        ]
        if regime is not None and not regime.empty:
            handles.extend(_regime_legend_handles())
        ax.legend(handles=handles, loc="lower left", frameon=False,
                  fontsize=C.FONT_TICK, ncol=2 if regime is not None else 1)

        subtitle = (
            f"Window: {window} {reporting_label} obs | Current VaR: {risk['VaR'].iloc[-1]:.2f}% | "
            f"Current CVaR: {risk['CVaR'].iloc[-1]:.2f}%"
        )
        if regime is not None and not regime.empty:
            subtitle += f" | {_regime_summary_text(regime, benchmark_name)}"
        style_ax(ax, title="Rolling VaR / CVaR",
                 ylabel=f"{reporting_label.title()} Return Tail Risk (%)", subtitle=subtitle)
        smart_date_axis(ax, risk["VaR"])

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 5
    @staticmethod
    def plot_returns_distribution(
        portfolio_calc: PortfolioCalculations,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Histogram of daily returns with KDE, normal overlay and stats box.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        returns = returns[returns != 0]  # drop zero-return days
        ret_pct = returns * 100

        bins = np.linspace(ret_pct.min(), ret_pct.max(),
                           min(80, max(20, len(ret_pct) // 10)))
        n, bin_edges, patches = ax.hist(
            ret_pct, bins=bins, density=True, alpha=C.FILL_HIST,
            color=C.BLUE, edgecolor=C.EDGE, linewidth=C.LW_EDGE, zorder=2,
        )

        # KDE
        try:
            kde_x = np.linspace(ret_pct.min() - 1, ret_pct.max() + 1, 300)
            kde = stats.gaussian_kde(ret_pct)
            ax.plot(kde_x, kde(kde_x), color=C.TEAL, lw=C.LW_MAIN + 0.2, zorder=4,
                    label="KDE")
        except Exception:
            pass

        # Normal overlay
        mu, sigma = ret_pct.mean(), ret_pct.std()
        norm_x = np.linspace(ret_pct.min() - 1, ret_pct.max() + 1, 300)
        norm_y = stats.norm.pdf(norm_x, mu, sigma)
        ax.plot(norm_x, norm_y, color=C.MUTED, lw=C.LW_THIN, ls="--", zorder=3,
                label="Normal")

        # Mean & zero lines
        ax.axvline(mu, color=C.BENCHMARK, ls=C.BENCHMARK_LS, lw=C.LW_THIN, zorder=5)
        ax.axvline(0, color=C.SPINE, ls="-", lw=0.8, zorder=5)

        # Stats box
        skew_val = ret_pct.skew()
        kurt_val = ret_pct.kurtosis()
        var95 = np.percentile(ret_pct, 5)
        best = ret_pct.max()
        worst = ret_pct.min()
        stats_text = (
            f"Mean: {mu:.3f}%\n"
            f"Std: {sigma:.3f}%\n"
            f"Skew: {skew_val:.2f}\n"
            f"Kurtosis: {kurt_val:.2f}\n"
            f"VaR(95%): {var95:.2f}%\n"
            f"Best Day: {best:+.2f}%\n"
            f"Worst Day: {worst:+.2f}%"
        )
        _stats_box(ax, stats_text, loc="upper right")

        handles = [
            Line2D([], [], color=C.TEAL, lw=C.LW_MAIN, label="KDE"),
            Line2D([], [], color=C.MUTED, lw=C.LW_THIN, ls="--", label="Normal"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS, label=f"Mean ({mu:.3f}%)"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        subtitle = (
            f"Frequency: {reporting_label} | Observations: {len(ret_pct):,}  |  "
            f"Skew: {skew_val:.2f}  |  Kurtosis: {kurt_val:.2f}"
        )
        style_ax(ax, title="Returns Distribution", xlabel=f"{reporting_label.title()} Return (%)",
                 ylabel="Density", subtitle=subtitle)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 6
    @staticmethod
    def plot_correlation_heatmap(
        instrument_calc: InstrumentCalculations,
    ) -> plt.Figure:
        """Lower-triangle correlation heatmap with diverging colourmap.

        Parameters
        ----------
        instrument_calc : InstrumentCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11, 9))

        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()

        clean_cols = [_clean_label(c) for c in returns.columns]
        returns.columns = clean_cols

        corr = returns.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        cmap = diverging_cmap()
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            vmin=-1, vmax=1, center=0, ax=ax, square=True,
            annot_kws={"size": C.FONT_TICK, "fontweight": "bold"},
            cbar_kws={"label": "Correlation", "shrink": 0.8},
            linewidths=0.45, linecolor=C.FIG_BG,
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        # Summary stats text
        no_diag = corr.values[~np.eye(len(corr), dtype=bool)]
        if len(no_diag) > 0:
            avg_c = np.nanmean(no_diag)
            max_c = np.nanmax(no_diag)
            min_c = np.nanmin(no_diag)
            stats_text = (
                f"Avg Corr: {avg_c:.2f}\n"
                f"Max Corr: {max_c:.2f}\n"
                f"Min Corr: {min_c:.2f}"
            )
            _stats_box(ax, stats_text, loc="lower left")

        style_ax(ax, title="Correlation Matrix")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 6b
    @staticmethod
    def plot_correlation_snapshot(
        instrument_calc: InstrumentCalculations,
        trailing_window: int = 252,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Full-history vs trailing-window correlation matrices."""
        ensure_style()
        fig = plt.figure(figsize=(13.4, 5.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.28], wspace=0.26)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis("off")

        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
        returns = returns.dropna(how="all").copy()
        returns.columns = [_clean_label(c) for c in returns.columns]
        if returns.shape[1] < 2:
            raise ValueError("At least two assets are required for correlation snapshot.")

        trailing_obs = min(max(trailing_window, 12), len(returns))
        full_corr = returns.corr()
        trailing_corr = returns.tail(trailing_obs).corr()
        cmap = diverging_cmap()

        for ax, corr, title in [
            (axes[0], full_corr, "Full History"),
            (axes[1], trailing_corr, f"Trailing {trailing_obs}d"),
        ]:
            sns.heatmap(
                corr, annot=returns.shape[1] <= 8, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0, ax=ax, square=True,
                annot_kws={"size": C.FONT_TICK - 0.5, "fontweight": "bold"},
                cbar=False, linewidths=0.45, linecolor=C.FIG_BG,
            )
            ax.set_title(title, color=C.TITLE, fontsize=C.FONT_LABEL,
                         fontweight="bold", pad=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)

        def _avg_offdiag(corr: pd.DataFrame) -> float:
            values = corr.values[~np.eye(len(corr), dtype=bool)]
            return float(np.nanmean(values)) if len(values) else np.nan

        full_avg = _avg_offdiag(full_corr)
        trailing_avg = _avg_offdiag(trailing_corr)
        delta = trailing_avg - full_avg

        stats_text = (
            f"Full Avg: {full_avg:.2f}\n"
            f"Trailing Avg: {trailing_avg:.2f}\n"
            f"Delta: {delta:+.2f}\n"
            f"Assets: {returns.shape[1]}"
        )
        ax_stats.text(0.0, 0.86, "Summary",
                      transform=ax_stats.transAxes, ha="left", va="top",
                      fontsize=C.FONT_LABEL, fontweight="bold", color=C.TITLE)
        ax_stats.text(
            0.0, 0.76, stats_text,
            transform=ax_stats.transAxes, ha="left", va="top",
            fontsize=C.FONT_ANNOT, color=C.LABEL, linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.38", fc=C.FIG_BG, ec=C.GRID,
                      alpha=0.92, lw=0.6),
        )

        fig.suptitle("Correlation Snapshot", x=0.06, y=0.98,
                     ha="left", fontsize=C.FONT_TITLE, fontweight="bold",
                     color=C.TITLE)
        fig.text(0.94, 0.965,
                 f"Full sample vs recent diversification | Window: {trailing_obs} {reporting_label} obs",
                 ha="right", va="top", fontsize=C.FONT_ANNOT, color=C.SUBTITLE,
                 bbox=dict(boxstyle="round,pad=0.22", fc=C.FIG_BG,
                           ec=C.GRID, alpha=0.78, lw=0.45))

        fig.subplots_adjust(left=0.06, right=0.97, top=0.84,
                            bottom=0.16, wspace=0.34)
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 7
    @staticmethod
    def plot_portfolio_weights(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: Optional[InstrumentCalculations] = None,
    ) -> plt.Figure:
        """Stacked area of raw (un-normalised) portfolio weights over time.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        instrument_calc : InstrumentCalculations, optional

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        weights = portfolio_calc.portfolio_data.weights
        if weights is None or (hasattr(weights, "empty") and weights.empty):
            raise ValueError("No weights data available in the portfolio.")

        if isinstance(weights, pd.Series):
            weights = weights.to_frame()

        w = weights.clip(lower=0).copy()
        labels = [_clean_label(c) for c in w.columns]
        colors = _allocation_colors(labels, asset_alpha=0.92, cash_alpha=0.64)

        # Step-interpolation for clean discrete transitions
        idx = w.index
        if len(idx) > 1:
            new_rows, new_idx = [], []
            for i in range(len(idx) - 1):
                new_rows.append(w.iloc[i].values)
                new_idx.append(idx[i])
                new_rows.append(w.iloc[i].values)
                new_idx.append(idx[i + 1] - pd.Timedelta(microseconds=1))
            new_rows.append(w.iloc[-1].values)
            new_idx.append(idx[-1])
            w_step = pd.DataFrame(new_rows, index=new_idx, columns=w.columns)
        else:
            w_step = w

        ax.stackplot(w_step.index,
                     *[w_step.iloc[:, i] for i in range(w_step.shape[1])],
                     colors=colors, alpha=1.0, linewidth=0,
                     edgecolor="none")

        style_ax(ax, title="Portfolio Weights Over Time", ylabel="Weight")
        smart_date_axis(ax, w)

        handles = [mpatches.Patch(facecolor=colors[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95, fontsize=9.5, edgecolor=C.SPINE, borderpad=0.5)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 8
    @staticmethod
    def plot_percentage_weights(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: Optional[InstrumentCalculations] = None,
    ) -> plt.Figure:
        """Stacked area of portfolio allocation over time.

        Shows each instrument's share of the invested capital plus a
        "Cash" band for any unallocated portion (weights summing to
        less than 1).  Uses step interpolation so discrete rebalance
        events are rendered cleanly without diagonal transitions.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        instrument_calc : InstrumentCalculations, optional

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        weights = portfolio_calc.portfolio_data.weights
        if weights is None or (hasattr(weights, "empty") and weights.empty):
            raise ValueError("No weights data available in the portfolio.")
        if isinstance(weights, pd.Series):
            weights = weights.to_frame()

        # Clip negative weights to 0 for the stacked area display
        w = weights.clip(lower=0).copy()

        # Add a Cash band = 1 − Σ(weights), clipped to [0, 1]
        cash = (1.0 - w.sum(axis=1)).clip(0, 1)
        w["Cash"] = cash

        # Build labels & colours — Cash gets a neutral grey-blue.
        asset_labels = [_clean_label(c) for c in weights.columns]
        labels = asset_labels + ["Cash"]
        colors = _allocation_colors(labels, asset_alpha=0.92, cash_alpha=0.64)

        # Step-interpolation: duplicate each row at the next timestamp
        # so transitions are vertical, not diagonal
        idx = w.index
        if len(idx) > 1:
            new_rows = []
            new_idx = []
            for i in range(len(idx) - 1):
                new_rows.append(w.iloc[i].values)
                new_idx.append(idx[i])
                # Insert a copy of the old weights just before the next date
                new_rows.append(w.iloc[i].values)
                new_idx.append(idx[i + 1] - pd.Timedelta(microseconds=1))
            new_rows.append(w.iloc[-1].values)
            new_idx.append(idx[-1])
            w_step = pd.DataFrame(new_rows, index=new_idx, columns=w.columns)
        else:
            w_step = w

        ax.stackplot(w_step.index,
                     *[w_step.iloc[:, i] for i in range(w_step.shape[1])],
                     colors=colors, alpha=1.0, linewidth=0,
                     edgecolor="none")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

        # Subtitle with allocation stats
        invested = (1.0 - cash)
        avg_invested = invested.mean()
        cash_only_runs = _true_runs(cash >= 0.995)
        cash_only_days = int((cash >= 0.995).sum())
        longest_cash = max((r[2] for r in cash_only_runs), default=0)
        first_invested = invested[invested > 0.005]
        first_invested_ts = first_invested.index[0] if not first_invested.empty else None
        runtime_cash_runs = [
            run for run in cash_only_runs
            if first_invested_ts is not None and run[0] > first_invested_ts
        ]
        material_runtime_runs = [run for run in runtime_cash_runs if run[2] >= 20]
        runtime_cash_days = sum(run[2] for run in runtime_cash_runs)
        runtime_longest_cash = max((run[2] for run in runtime_cash_runs), default=0)

        if material_runtime_runs:
            start = material_runtime_runs[0][0]
            end = material_runtime_runs[-1][1]
            midpoint = start + (end - start) / 2
            lines = ["Risk-off cash"]
            for run_start, run_end, length in material_runtime_runs[:3]:
                lines.append(f"{run_start:%Y-%m-%d} to {run_end:%Y-%m-%d} ({length} obs)")
            ax.annotate(
                "\n".join(lines),
                xy=(midpoint, 0.97),
                xytext=(0, -38),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=C.FONT_ANNOT,
                color=C.TITLE,
                bbox=dict(boxstyle="round,pad=0.34", fc=C.FIG_BG,
                          ec=_CASH_COLOR, alpha=0.92, lw=0.8),
                arrowprops=dict(arrowstyle="-", color=_CASH_COLOR, lw=0.9),
                zorder=7,
            )
        subtitle = (f"Avg invested {avg_invested:.0%}  ·  "
                    f"Avg cash {1 - avg_invested:.0%}  ·  "
                    f"Risk-off cash {runtime_cash_days}d / longest {runtime_longest_cash}d  ·  "
                    f"{len(weights.columns)} instruments")

        style_ax(ax, title="Portfolio Allocation Over Time",
                 subtitle=subtitle, ylabel="Allocation (%)")
        smart_date_axis(ax, w)

        handles = [mpatches.Patch(facecolor=colors[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95, fontsize=9.5, edgecolor=C.SPINE, borderpad=0.5)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 9
    @staticmethod
    def plot_monthly_returns_heatmap(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Monthly returns heatmap (months as rows, years as columns) with
        yearly totals column on the right.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(15, 7.5))

        returns = portfolio_calc.returns.dropna()
        # Keep only trading days
        returns = returns[returns.index.dayofweek < 5]

        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        active_returns = returns[returns.abs() > 1e-12]
        first_active_month = None
        if not active_returns.empty:
            first_active_ts = pd.Timestamp(active_returns.index[0])
            if first_active_ts.tzinfo is not None:
                first_active_ts = first_active_ts.tz_localize(None)
            first_active_month = first_active_ts + pd.offsets.MonthEnd(0)
            if getattr(monthly.index, "tz", None) is not None:
                first_active_month = first_active_month.tz_localize(monthly.index.tz)
            monthly.loc[monthly.index < first_active_month] = np.nan

        def _compound_valid(x: pd.Series) -> float:
            x = x.dropna()
            return (1 + x).prod() - 1 if len(x) else np.nan

        yearly = monthly.groupby(monthly.index.year).apply(_compound_valid)

        # Build matrix: rows = years, columns = months
        matrix = (
            monthly
            .groupby([monthly.index.year, monthly.index.month])
            .first()
            .unstack()
            .reindex(columns=range(1, 13))
        )
        matrix.index.name = None     # remove auto "date" y-label
        matrix.columns.name = None   # remove auto "date" x-label

        cmap = diverging_cmap()

        sns.heatmap(
            matrix, annot=True, fmt=".1%", cmap=cmap, center=0, ax=ax,
            mask=matrix.isna(),
            cbar=False, linewidths=0.45, linecolor=C.FIG_BG,
            annot_kws={"size": C.FONT_TICK, "fontweight": "bold"},
        )

        # Yearly returns column
        for i, year in enumerate(matrix.index):
            yr_ret = yearly.get(year, np.nan)
            if pd.isna(yr_ret):
                continue
            color = C.BLUE if yr_ret >= 0 else C.RED
            ax.text(
                13, i + 0.5, f"{yr_ret:+.1%}",
                va="center", ha="left", fontweight="bold",
                fontsize=9.5, color=color,
            )

        ax.set_xlim(0, 14)
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        ax.set_xticks([i + 0.5 for i in range(12)])
        ax.set_xticklabels(month_names, fontsize=9, color=C.TICK)
        ax.tick_params(axis="y", labelsize=9, colors=C.TICK)

        # Win rate subtitle
        total_m = int(monthly.notna().sum())
        pos_m = int((monthly > 0).sum())
        win_rate = pos_m / total_m * 100 if total_m > 0 else 0

        subtitle = f"Win Rate: {win_rate:.0f}% ({pos_m}/{total_m} active months)"
        if first_active_month is not None:
            subtitle += f"  |  Active from {first_active_month:%b %Y}"
        style_ax(ax, title="Monthly Returns Heatmap",
                 subtitle=subtitle)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 10
    @staticmethod
    def plot_return_quantiles(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Box plots of returns by period (daily, weekly, monthly, quarterly,
        yearly). Boxes coloured green/red by median sign.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.replace([np.inf, -np.inf], np.nan).dropna()

        rules = {"D": "Daily", "W": "Weekly", "ME": "Monthly",
                 "QE": "Quarterly", "YE": "Yearly"}
        data_list: List[np.ndarray] = []
        tick_labels: List[str] = []
        for rule, label in rules.items():
            r = returns.resample(rule).last().dropna()
            r = r.replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) > 0:
                data_list.append((r * 100).values)
                tick_labels.append(label)

        if not data_list:
            raise ValueError("All resampled datasets are empty.")

        bp = ax.boxplot(
            data_list, tick_labels=tick_labels, patch_artist=True, notch=True,
            medianprops=dict(color=C.TITLE, lw=1.2),
            whiskerprops=dict(color=C.NAVY, lw=0.8),
            capprops=dict(color=C.NAVY, lw=0.8),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=C.MUTED,
                            markeredgecolor=C.MUTED, alpha=0.5),
        )

        box_colors = [C.ICE, C.PALE, C.LIGHT, C.STEEL, C.BLUE][:len(data_list)]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)
            patch.set_edgecolor(C.NAVY)
            patch.set_linewidth(1.0)

        ax.axhline(0, color=C.SPINE, ls="-", lw=0.7)

        daily_median = np.median(data_list[0]) if data_list else 0
        style_ax(ax, title="Return Quantiles by Period", ylabel="Return (%)",
                 subtitle=f"Daily Median: {daily_median:+.3f}%  |  Periods: {len(tick_labels)}")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 11
    @staticmethod
    def plot_annual_returns(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Vertical bar chart of annual returns coloured by sign, with average
        line and value labels.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        returns = returns[returns.index.dayofweek < 5]

        annual = (1 + returns).groupby(returns.index.year).prod() - 1

        bar_colors = [C.BLUE if v >= 0 else C.RED for v in annual.values]
        bars = ax.bar(annual.index.astype(str), annual.values, color=bar_colors,
                      edgecolor=C.EDGE, linewidth=C.LW_EDGE + 0.3, width=0.75, zorder=3,
                      alpha=C.FILL_HEAVY + 0.05)

        # Value labels
        for bar, val in zip(bars, annual.values):
            y_offset = 0.005 if val >= 0 else -0.005
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_offset,
                    f"{val:.1%}", ha="center", va=va, fontsize=C.FONT_ANNOT - 1,
                    fontweight="bold", color=C.LABEL)

        # Average line
        avg = annual.mean()
        ax.axhline(avg, color=C.BENCHMARK, ls=C.BENCHMARK_LS, lw=C.LW_THIN, zorder=4)
        ax.text(len(annual) - 0.5, avg, f"  Avg: {avg:.1%}",
                fontsize=8.5, color=C.BENCHMARK, va="center", fontweight="bold")

        ax.axhline(0, color=C.SPINE, lw=0.7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

        pos_years = (annual > 0).sum()
        total_years = len(annual)
        subtitle = f"Positive Years: {pos_years}/{total_years}  |  Average: {avg:.1%}"

        handles = [
            Line2D([], [], color=C.BLUE, lw=6, label="Positive"),
            Line2D([], [], color=C.RED, lw=6, label="Negative"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS, label="Average"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  fontsize=C.FONT_TICK)

        style_ax(ax, title="Annual Returns", ylabel="Return (%)",
                 subtitle=subtitle)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 12
    @staticmethod
    def plot_distribution_of_monthly_returns(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Histogram of monthly returns with normal overlay and stats.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()
        m_pct = monthly * 100

        # Histogram
        n_bins = min(50, max(15, len(m_pct) // 3))
        bins_arr = np.linspace(m_pct.min(), m_pct.max(), n_bins)
        n_vals, bin_edges, patches = ax.hist(
            m_pct, bins=bins_arr, density=True, alpha=0.82,
            edgecolor=C.EDGE, linewidth=C.LW_EDGE, color=C.BLUE, zorder=2,
        )

        # Normal overlay
        mu, sigma = m_pct.mean(), m_pct.std()
        x_norm = np.linspace(m_pct.min() - 2, m_pct.max() + 2, 300)
        ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
                color=C.MUTED, ls="--", lw=C.LW_SECONDARY - 0.2, zorder=3, label="Normal")

        # Mean / zero lines
        ax.axvline(mu, color=C.BENCHMARK, ls=C.BENCHMARK_LS, lw=C.LW_THIN, zorder=5)
        ax.axvline(0, color=C.SPINE, ls="-", lw=0.7, zorder=5)

        # Stats box
        pos = (monthly > 0).sum()
        total = len(monthly)
        win_rate = pos / total * 100 if total > 0 else 0
        stats_text = (
            f"Win Rate: {win_rate:.0f}% ({pos}/{total})\n"
            f"Best Month: {m_pct.max():+.1f}%\n"
            f"Worst Month: {m_pct.min():+.1f}%\n"
            f"Mean: {mu:.2f}%\n"
            f"Std: {sigma:.2f}%"
        )
        _stats_box(ax, stats_text, loc="upper right")

        handles = [
            Line2D([], [], color=C.BLUE, lw=6, alpha=0.75, label="Monthly Returns"),
            Line2D([], [], color=C.MUTED, lw=1.0, ls="--", label="Normal"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                   label=f"Mean ({mu:.2f}%)"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        style_ax(ax, title="Distribution of Monthly Returns",
                 xlabel="Monthly Return (%)", ylabel="Density")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 13
    @staticmethod
    def plot_composition(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Donut chart showing current portfolio allocation.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(9.8, 6.4))

        weights = portfolio_calc.portfolio_data.weights
        if weights is None or (hasattr(weights, "empty") and weights.empty):
            raise ValueError("Weights data is empty.")
        if isinstance(weights, pd.DataFrame):
            weights = weights.iloc[-1]
        if not isinstance(weights, pd.Series):
            raise TypeError("Weights should be a pandas Series.")

        sorted_w = weights.sort_values(ascending=False)
        invested = float(sorted_w.sum())
        cash_weight = 1.0 - invested
        if abs(cash_weight) > 1e-6:
            sorted_w.loc["Cash"] = cash_weight
            sorted_w = sorted_w.sort_values(ascending=False)

        # Group small positions
        threshold = 0.025
        others = sorted_w[sorted_w.abs() < threshold].sum()
        main_w = sorted_w[sorted_w.abs() >= threshold].copy()
        if others != 0:
            main_w["Others"] = others

        labels = [_clean_label(l) for l in main_w.index]
        colors = _allocation_colors(labels, asset_alpha=0.96, cash_alpha=0.68)

        pie_values = main_w.abs()
        pie_total = float(pie_values.sum())
        wedges, texts, autotexts = ax.pie(
            pie_values.values, labels=None,
            autopct=lambda p: f"{p:.1f}%" if p >= 6 else "",
            startangle=90, colors=colors,
            wedgeprops=dict(width=0.42, edgecolor=C.FIG_BG, linewidth=1.4),
            pctdistance=0.75,
        )
        for wedge, at in zip(wedges, autotexts):
            at.set_fontsize(C.FONT_TICK)
            if at.get_text():
                r, g, b, _ = wedge.get_facecolor()
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                at.set_color(C.TITLE if luminance > 0.72 else C.FIG_BG)
            at.set_fontweight("bold")

        ax.axis("equal")

        # Centre text
        strategy_name = getattr(portfolio_calc.portfolio_data, "name", "Portfolio")
        ax.text(0, 0, strategy_name, ha="center", va="center",
                fontsize=C.FONT_LABEL, fontweight="bold", color=C.TITLE)

        # Legend
        leg_labels = [
            f"{labels[i]}  {pie_values.iloc[i] / pie_total:.1%}"
            for i in range(len(labels))
        ]
        handles = [mpatches.Patch(facecolor=colors[i], label=leg_labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.95, 0.5),
                  framealpha=0.88, fontsize=C.FONT_TICK, edgecolor=C.GRID, borderpad=0.5,
                  labelspacing=0.6)

        style_ax(ax, title="Portfolio Composition")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 14
    @staticmethod
    def plot_efficient_frontier(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: InstrumentCalculations,
        risk_free_rate: float = 0.0,
    ) -> plt.Figure:
        """Efficient frontier with current portfolio, individual assets,
        and capital allocation line.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        instrument_calc : InstrumentCalculations
        risk_free_rate : float

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        ret_df = instrument_calc.returns
        if isinstance(ret_df, pd.Series):
            ret_df = ret_df.to_frame()

        mean_ret = ret_df.mean() * 252
        cov_mat = ret_df.cov() * 252

        weights = portfolio_calc.portfolio_data.weights
        if weights is None or (hasattr(weights, "empty") and weights.empty):
            raise ValueError("Weights data is empty.")
        if isinstance(weights, pd.DataFrame):
            weights = weights.iloc[-1]
        w = weights.values.astype(float)

        # Current portfolio
        port_ret = float(np.sum(mean_ret.values * w))
        port_std = float(np.sqrt(np.dot(w, np.dot(cov_mat.values, w))))

        # Random portfolios (scatter cloud)
        n_sims = 3000
        n_assets = len(mean_ret)
        sim_ret, sim_std = [], []
        for _ in range(n_sims):
            rw = np.random.dirichlet(np.ones(n_assets))
            sim_ret.append(float(np.sum(rw * mean_ret.values)))
            sim_std.append(float(np.sqrt(np.dot(rw, np.dot(cov_mat.values, rw)))))

        ax.scatter(sim_std, sim_ret, c=C.TEAL, s=6, alpha=0.22, zorder=1,
                   edgecolors="none")

        # Efficient frontier
        n_pts = 80
        targets = np.linspace(min(mean_ret), max(mean_ret), n_pts)
        ef_std = []
        for tgt in targets:
            cons = (
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x: float(np.sum(x * mean_ret.values)) - tgt},
            )
            x0 = np.ones(n_assets) / n_assets
            res = minimize(
                lambda x: float(np.sqrt(np.dot(x.T, np.dot(cov_mat.values, x)))),
                x0, method="SLSQP", constraints=cons,
                bounds=tuple((0, 1) for _ in range(n_assets)),
                options={"maxiter": 200},
            )
            ef_std.append(res.fun if res.success else np.nan)

        valid = ~np.isnan(ef_std)
        ax.plot(np.array(ef_std)[valid], targets[valid], color=C.BLUE,
                lw=C.LW_MAIN + 0.5, zorder=5, label="Efficient Frontier")

        # Current portfolio
        ax.scatter(port_std, port_ret, marker="*", s=300, color=C.BENCHMARK,
                   edgecolors=C.EDGE, linewidths=C.LW_THIN, zorder=7,
                   label="Current Portfolio")

        # Individual assets
        asset_labels = [_clean_label(c) for c in ret_df.columns]
        asset_colors = C.get(n_assets)
        for i in range(n_assets):
            a_std = float(np.sqrt(cov_mat.values[i, i]))
            a_ret = float(mean_ret.values[i])
            ax.scatter(a_std, a_ret, marker="o", s=C.MARKER_LG, color=asset_colors[i],
                       edgecolors=C.EDGE, linewidths=0.6, zorder=6)
            ax.annotate(
                asset_labels[i], (a_std, a_ret), fontsize=C.FONT_SMALL + 0.5,
                color=asset_colors[i], xytext=(6, 4),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc=C.FIG_BG,
                          ec=C.GRID, alpha=0.85, lw=0.4),
            )

        # Capital allocation line
        if port_std > 0:
            sharpe = (port_ret - risk_free_rate) / port_std
            cal_x = np.linspace(0, max(sim_std) * 0.9, 100)
            cal_y = risk_free_rate + sharpe * cal_x
            ax.plot(cal_x, cal_y, color=C.TEAL, ls="--", lw=0.9, alpha=0.6,
                    zorder=2, label="CAL")

        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        # Auto-center xlim: tight left margin, 10% padding right
        all_stds = sim_std + [port_std] + [float(np.sqrt(cov_mat.values[i, i])) for i in range(n_assets)]
        x_min = min(all_stds) * 0.85
        x_max = max(all_stds) * 1.10
        ax.set_xlim(x_min, x_max)

        sharpe_val = (port_ret - risk_free_rate) / port_std if port_std > 0 else 0
        subtitle = (f"Portfolio: Return {port_ret:.1%} | Risk {port_std:.1%} | "
                    f"Sharpe {sharpe_val:.2f}")

        handles_leg = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Efficient Frontier"),
            Line2D([], [], marker="*", color=C.BENCHMARK, lw=0, markersize=12,
                   label="Current Portfolio"),
            Line2D([], [], color=C.TEAL, lw=0.9, ls="--", label="CAL"),
        ]
        ax.legend(handles=handles_leg, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        style_ax(ax, title="Efficient Frontier",
                 xlabel="Annualised Volatility",
                 ylabel="Annualised Return", subtitle=subtitle)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 15
    @staticmethod
    def plot_rolling_beta(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        window: int = 252,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Rolling beta with green/red fill zones around 1.0.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        benchmark_returns : pd.Series
        window : int

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        port_ret = portfolio_calc.returns.dropna()
        bench = benchmark_returns.dropna()
        port_ret, bench = port_ret.align(bench, join="inner")

        rolling_cov = port_ret.rolling(window).cov(bench)
        rolling_var = bench.rolling(window).var()
        beta = (rolling_cov / rolling_var).dropna()

        if beta.empty:
            raise ValueError("No valid beta values calculated.")

        ax.plot(beta.index, beta.values, color=C.BLUE, lw=C.LW_MAIN, zorder=4)

        # Thresholds
        for lev in [0.5, 1.0, 1.5]:
            ax.axhline(lev, color=C.SPINE, ls=":", lw=0.55, alpha=0.55)

        avg_beta = _safe_float(beta.mean())
        ax.axhline(avg_beta, color=C.BENCHMARK, ls=C.BENCHMARK_LS,
                   lw=C.LW_THIN, alpha=0.9, zorder=3)

        endpoint_annotation(ax, beta, "Current", C.BLUE, fmt="ratio")
        endpoint_annotation(ax, pd.Series(avg_beta, index=[beta.index[-1]]),
                            "Average", C.BENCHMARK, fmt="ratio", offset=(8, -16))

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Rolling Beta"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                   label=f"Average ({avg_beta:.2f})"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        subtitle = (f"Window: {window} {reporting_label} obs  |  Current: {_safe_float(beta.iloc[-1]):.2f}  |  "
                    f"Average: {avg_beta:.2f}")
        style_ax(ax, title="Rolling Beta", ylabel="Beta", subtitle=subtitle)
        smart_date_axis(ax, beta)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 15b
    @staticmethod
    def plot_rolling_beta_with_regime(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        benchmark_name: str = "Benchmark",
        window: int = 252,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Rolling benchmark beta with benchmark regime shading."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        port_ret = portfolio_calc.returns.dropna()
        bench = benchmark_returns.dropna()
        port_ret, bench = port_ret.align(bench, join="inner")

        rolling_cov = port_ret.rolling(window).cov(bench)
        rolling_var = bench.rolling(window).var()
        beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan).dropna()
        if beta.empty:
            raise ValueError("No valid beta values calculated.")

        regime = _benchmark_regime_series(bench, beta.index)
        _shade_regimes(ax, regime, full_height=True)

        ax.plot(beta.index, beta.values, color=C.BLUE, lw=C.LW_MAIN, zorder=4)
        for lev in [0.5, 1.0, 1.5]:
            ax.axhline(lev, color=C.SPINE, ls=":", lw=0.55, alpha=0.55)

        avg_beta = _safe_float(beta.mean())
        ax.axhline(avg_beta, color=C.BENCHMARK, ls=C.BENCHMARK_LS,
                   lw=C.LW_THIN, alpha=0.9, zorder=3)

        endpoint_annotation(ax, beta, "Current", C.BLUE, fmt="ratio")
        endpoint_annotation(ax, pd.Series(avg_beta, index=[beta.index[-1]]),
                            "Average", C.BENCHMARK, fmt="ratio", offset=(8, -16))

        stats_lines = []
        aligned_regime = regime.reindex(beta.index)
        for name in ["Bull", "Sideways", "Bear"]:
            vals = beta[aligned_regime == name].dropna()
            if len(vals):
                stats_lines.append(f"{name}: {vals.mean():.2f} avg beta")
        if stats_lines:
            _stats_box(ax, "\n".join(stats_lines), loc="upper right")

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Rolling Beta"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN,
                   ls=C.BENCHMARK_LS, label=f"Average ({avg_beta:.2f})"),
            *_regime_legend_handles(),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK, ncol=2)

        subtitle = (
            f"Window: {window} {reporting_label} obs | Current: {_safe_float(beta.iloc[-1]):.2f} | "
            f"{_regime_summary_text(regime, benchmark_name)}"
        )
        style_ax(ax, title="Rolling Beta with Benchmark Regimes",
                 ylabel="Beta", subtitle=subtitle)
        smart_date_axis(ax, beta)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 16
    @staticmethod
    def plot_rolling_alpha(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        window: int = 252,
        periods_per_year: Optional[int] = None,
        reporting_label: str = "daily",
        benchmark_name: str = "Benchmark",
        show_regime: bool = False,
    ) -> plt.Figure:
        """Rolling alpha with fill zones above/below zero.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        benchmark_returns : pd.Series
        window : int

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        periods_per_year = int(periods_per_year or getattr(portfolio_calc, "trading_days", 252))

        port_ret = portfolio_calc.returns.dropna()
        bench = benchmark_returns.dropna()
        port_ret, bench = port_ret.align(bench, join="inner")

        # Rolling beta
        rolling_cov = port_ret.rolling(window).cov(bench)
        rolling_var = bench.rolling(window).var()
        r_beta = rolling_cov / rolling_var

        # Rolling alpha (annualised)
        r_alpha = ((port_ret.rolling(window).mean()
                    - r_beta * bench.rolling(window).mean()) * periods_per_year).dropna()

        regime = None
        if show_regime:
            regime = _benchmark_regime_series(bench, r_alpha.index)
            _shade_regimes(ax, regime, full_height=True)

        ax.plot(r_alpha.index, r_alpha.values, color=C.BLUE, lw=C.LW_MAIN, zorder=4)

        ax.axhline(0, color=C.SPINE, ls="-", lw=0.65, alpha=0.75)

        avg_alpha = _safe_float(r_alpha.mean())
        ax.axhline(avg_alpha, color=C.BENCHMARK, ls=C.BENCHMARK_LS,
                   lw=C.LW_THIN, alpha=0.9, zorder=3)

        no_alpha_runs = _true_runs(r_alpha <= 0)
        if not show_regime:
            for start, end, _length in no_alpha_runs:
                ax.axvspan(start, end, color=C.RED, alpha=0.055, zorder=1)
        no_alpha_time = (r_alpha <= 0).mean() * 100
        longest_no_alpha = max((r[2] for r in no_alpha_runs), default=0)
        _stats_box(
            ax,
            f"Alpha <= 0: {no_alpha_time:.0f}%\n"
            f"Longest <= 0: {longest_no_alpha} obs\n"
            f"{'Background: regimes' if show_regime else 'Shaded: no-alpha periods'}",
            loc="upper right",
        )

        endpoint_annotation(ax, r_alpha, "Current", C.BLUE, fmt="ratio")
        endpoint_annotation(ax, pd.Series(avg_alpha, index=[r_alpha.index[-1]]),
                            "Average", C.BENCHMARK, fmt="ratio", offset=(8, -16))

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Rolling Alpha"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                   label=f"Average ({avg_alpha:.3f})"),
        ]
        if show_regime and regime is not None and not regime.empty:
            handles.extend(_regime_legend_handles())
        else:
            handles.append(mpatches.Patch(facecolor=mcolors.to_rgba(C.RED, 0.12),
                                          edgecolor="none", label="Alpha <= 0"))
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK, ncol=2 if show_regime else 1)

        subtitle = (f"Window: {window} {reporting_label} obs  |  Current: {_safe_float(r_alpha.iloc[-1]):.3f}  |  "
                    f"Average: {avg_alpha:.3f}")
        if show_regime and regime is not None and not regime.empty:
            subtitle += f"  |  {_regime_summary_text(regime, benchmark_name)}"
        style_ax(ax, title="Rolling Alpha (Annualised)", ylabel="Alpha",
                 subtitle=subtitle)
        smart_date_axis(ax, r_alpha)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 17
    @staticmethod
    def plot_rolling_information_ratio(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        window: int = 252,
        periods_per_year: Optional[int] = None,
        reporting_label: str = "daily",
        benchmark_name: str = "Benchmark",
        show_regime: bool = False,
    ) -> plt.Figure:
        """Rolling information ratio with fill zones and thresholds.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        benchmark_returns : pd.Series
        window : int

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        periods_per_year = int(periods_per_year or getattr(portfolio_calc, "trading_days", 252))

        port_ret = portfolio_calc.returns.dropna()
        bench = benchmark_returns.dropna()
        port_ret, bench = port_ret.align(bench, join="inner")

        excess = port_ret - bench
        rolling_excess = excess.rolling(window).mean() * periods_per_year
        rolling_te = excess.rolling(window).std() * np.sqrt(periods_per_year)
        ir = (rolling_excess / rolling_te).replace([np.inf, -np.inf], np.nan).dropna()

        regime = None
        if show_regime:
            regime = _benchmark_regime_series(bench, ir.index)
            _shade_regimes(ax, regime, full_height=True)

        ax.plot(ir.index, ir.values, color=C.BLUE, lw=C.LW_MAIN, zorder=4)

        for lev in [0, 0.5]:
            ax.axhline(lev, color=C.SPINE, ls=":", lw=0.55, alpha=0.55)

        avg_ir = _safe_float(ir.mean())
        ax.fill_between(ir.index, ir.values, 0,
                        where=ir.values >= 0, color=C.BLUE, alpha=0.055,
                        interpolate=True, zorder=1)
        ax.fill_between(ir.index, ir.values, 0,
                        where=ir.values < 0, color=C.RED, alpha=0.055,
                        interpolate=True, zorder=1)

        ax.axhline(avg_ir, color=C.BENCHMARK, ls=C.BENCHMARK_LS,
                   lw=C.LW_THIN, alpha=0.9, zorder=3)

        endpoint_annotation(ax, ir, "Current", C.BLUE, fmt="ratio")
        endpoint_annotation(ax, pd.Series(avg_ir, index=[ir.index[-1]]),
                            "Average", C.BENCHMARK, fmt="ratio", offset=(8, -16))

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Information Ratio"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                   label=f"Average ({avg_ir:.2f})"),
        ]
        if show_regime and regime is not None and not regime.empty:
            handles.extend(_regime_legend_handles())
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK, ncol=2 if show_regime else 1)

        subtitle = (f"Window: {window} {reporting_label} obs  |  Current: {_safe_float(ir.iloc[-1]):.2f}  |  "
                    f"Average: {avg_ir:.2f}")
        if show_regime and regime is not None and not regime.empty:
            subtitle += f"  |  {_regime_summary_text(regime, benchmark_name)}"
        style_ax(ax, title="Rolling Information Ratio",
                 ylabel="Information Ratio", subtitle=subtitle)
        smart_date_axis(ax, ir)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 17b
    @staticmethod
    def plot_relative_performance(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        benchmark_name: str = "Benchmark",
        window: int = 252,
        periods_per_year: Optional[int] = None,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Cumulative relative performance and rolling active return."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        periods_per_year = int(periods_per_year or getattr(portfolio_calc, "trading_days", 252))

        port_ret = portfolio_calc.returns.dropna()
        bench = benchmark_returns.dropna()
        port_ret, bench = port_ret.align(bench, join="inner")

        required_obs = max(2, min(max(20, window // 3), window))
        if len(port_ret) < required_obs:
            raise ValueError("Insufficient data for relative performance plot.")

        port_growth = (1 + port_ret).cumprod()
        bench_growth = (1 + bench).cumprod()
        rel = (port_growth / bench_growth - 1).dropna() * 100
        active = (port_ret - bench).rolling(window, min_periods=required_obs).mean() * periods_per_year * 100
        active = active.dropna()

        ax.fill_between(rel.index, rel.values, 0,
                        where=rel.values >= 0, color=C.BLUE, alpha=0.08,
                        interpolate=True, zorder=1)
        ax.fill_between(rel.index, rel.values, 0,
                        where=rel.values < 0, color=C.RED, alpha=0.10,
                        interpolate=True, zorder=1)
        ax.plot(rel.index, rel.values, color=C.BLUE, lw=C.LW_MAIN,
                zorder=4, label="Cumulative Relative Return")
        ax.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)

        if not active.empty:
            ax.plot(active.index, active.values, color=C.BENCHMARK,
                    lw=C.LW_SECONDARY, ls=C.BENCHMARK_LS,
                    zorder=5, label=f"{window} {reporting_label} obs Active Return")

        endpoint_annotation(ax, rel, "Relative", C.BLUE, fmt="pct")

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN,
                   label="Cumulative Relative Return"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_SECONDARY,
                   ls=C.BENCHMARK_LS, label=f"{window} {reporting_label} obs Active Return"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        subtitle = (
            f"Vs {benchmark_name} | Current relative: {rel.iloc[-1]:+.1f}%"
        )
        if not active.empty:
            subtitle += f" | Current active: {active.iloc[-1]:+.1f}% ann."
        style_ax(ax, title="Relative Performance vs Benchmark",
                 ylabel="Relative Return / Active Return (%)", subtitle=subtitle)
        smart_date_axis(ax, rel)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 17c
    @staticmethod
    def plot_factor_exposure_proxy(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: InstrumentCalculations,
        benchmark_returns: pd.Series,
        periods_per_year: Optional[int] = None,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """OLS exposure to local proxy factors: market, momentum and liquidity-size."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        periods_per_year = int(periods_per_year or getattr(portfolio_calc, "trading_days", 252))

        port_ret = portfolio_calc.returns.dropna().rename("Portfolio")
        inst_ret = instrument_calc.returns.copy()
        if isinstance(inst_ret, pd.Series):
            inst_ret = inst_ret.to_frame()
        inst_ret.columns = [_clean_label(c) for c in inst_ret.columns]

        bench = benchmark_returns.dropna().rename("Market")
        factors = pd.DataFrame(index=inst_ret.index)
        factors["Market"] = bench.reindex(factors.index)

        # Cross-sectional momentum proxy: top trailing 63d performers minus bottom.
        mom_score = (1 + inst_ret).rolling(63, min_periods=40).apply(np.prod, raw=True) - 1
        mom_score = mom_score.shift(1)
        mom_vals = []
        for dt, scores in mom_score.iterrows():
            r = inst_ret.loc[dt].dropna()
            s = scores.reindex(r.index).dropna()
            common = r.index.intersection(s.index)
            if len(common) < 3:
                mom_vals.append(np.nan)
                continue
            ranked = s.loc[common].sort_values()
            n = max(1, len(ranked) // 3)
            low = ranked.index[:n]
            high = ranked.index[-n:]
            mom_vals.append(float(r.loc[high].mean() - r.loc[low].mean()))
        factors["Momentum"] = pd.Series(mom_vals, index=inst_ret.index)

        # Size proxy from rolling dollar volume: low ADV minus high ADV.
        size_proxy = pd.Series(np.nan, index=inst_ret.index, dtype=float)
        try:
            prices = instrument_calc.prices
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices.xs("adj_close", axis=1, level=1)
                volume = prices.xs("volume", axis=1, level=1)
            else:
                close = prices
                volume = pd.DataFrame(index=prices.index)
            close.columns = [_clean_label(c) for c in close.columns]
            volume.columns = [_clean_label(c) for c in volume.columns]
            adv = close.reindex(columns=inst_ret.columns).multiply(
                volume.reindex(columns=inst_ret.columns)
            ).rolling(63, min_periods=40).mean().shift(1)
            vals = []
            for dt, score in adv.iterrows():
                r = inst_ret.loc[dt].dropna()
                s = score.reindex(r.index).dropna()
                common = r.index.intersection(s.index)
                if len(common) < 3:
                    vals.append(np.nan)
                    continue
                ranked = s.loc[common].sort_values()
                n = max(1, len(ranked) // 3)
                low_liq = ranked.index[:n]
                high_liq = ranked.index[-n:]
                vals.append(float(r.loc[low_liq].mean() - r.loc[high_liq].mean()))
            size_proxy = pd.Series(vals, index=inst_ret.index)
        except Exception:
            pass
        factors["Size Proxy"] = size_proxy

        data = pd.concat([port_ret, factors], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        min_obs = max(8, min(120, periods_per_year * 2))
        if len(data) < min_obs:
            raise ValueError("Insufficient data for factor exposure proxy.")

        y = data["Portfolio"].values
        X_df = data[["Market", "Momentum", "Size Proxy"]]
        X = np.column_stack([np.ones(len(X_df)), X_df.values])
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ coefs
        resid = y - fitted
        r2 = 1 - (np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
        alpha_ann = coefs[0] * periods_per_year * 100

        exposures = pd.Series(coefs[1:], index=X_df.columns)
        colors = [C.BLUE, C.BENCHMARK, C.TEAL]
        bars = ax.bar(exposures.index, exposures.values, color=colors,
                      alpha=0.88, edgecolor=C.EDGE, linewidth=C.LW_EDGE,
                      zorder=3)
        ax.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)

        for bar, val in zip(bars, exposures.values):
            va = "bottom" if val >= 0 else "top"
            offset = 0.03 if val >= 0 else -0.03
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:.2f}", ha="center", va=va,
                    fontsize=C.FONT_ANNOT, fontweight="bold", color=C.LABEL)

        stats_text = (
            f"Annual Alpha: {alpha_ann:+.2f}%\n"
            f"Model R2: {r2:.2f}\n"
            f"Obs: {len(data):,}\n"
            "Size proxy: low ADV - high ADV"
        )
        _stats_box(ax, stats_text, loc="upper right")

        style_ax(ax, title="Proxy Factor Exposure",
                 ylabel="OLS Beta",
                 subtitle=f"Frequency: {reporting_label} | Local factors: market, momentum, liquidity-size proxy")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 17d
    @staticmethod
    def plot_turnover_vs_performance(
        portfolio_calc: PortfolioCalculations,
        window: int = 63,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Scatter of rolling turnover against same-window portfolio return."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        turnover = portfolio_calc.compute_turnover()
        if isinstance(turnover, pd.DataFrame):
            turnover = turnover.sum(axis=1)
        turnover = turnover.reindex(returns.index).fillna(0)

        min_periods = max(2, min(max(20, window // 3), window))
        roll_turnover = turnover.rolling(window, min_periods=min_periods).sum() * 100
        roll_return = returns.rolling(window, min_periods=min_periods).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        ) * 100
        data = pd.DataFrame({"Turnover": roll_turnover, "Return": roll_return}).dropna()
        if data.empty:
            raise ValueError("No turnover/performance windows calculated.")

        ax.scatter(data["Turnover"], data["Return"], s=22, color=C.BLUE,
                   alpha=0.42, edgecolors="none", zorder=3)
        ax.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)

        corr = data["Turnover"].corr(data["Return"])
        if len(data) >= 3 and data["Turnover"].std() > 0:
            slope, intercept = np.polyfit(data["Turnover"], data["Return"], 1)
            x_line = np.linspace(data["Turnover"].min(), data["Turnover"].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=C.BENCHMARK,
                    lw=C.LW_SECONDARY, ls=C.BENCHMARK_LS, zorder=4,
                    label=f"Fit (corr {corr:.2f})")
            ax.legend(loc="upper left", frameon=False, fontsize=C.FONT_TICK)

        _stats_box(
            ax,
            f"Corr: {corr:.2f}\n"
            f"Median TO: {data['Turnover'].median():.1f}%\n"
            f"Median Ret: {data['Return'].median():+.1f}%\n"
            f"Windows: {len(data):,}",
            loc="upper right",
        )

        style_ax(ax, title="Turnover vs Performance",
                 xlabel=f"{window} {reporting_label} obs Turnover (%)",
                 ylabel=f"{window} {reporting_label} obs Return (%)",
                 subtitle="Checks whether higher trading activity is associated with better realised performance")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 17e
    @staticmethod
    def plot_rolling_tail_risk(
        portfolio_calc: PortfolioCalculations,
        window: int = 252,
        reporting_label: str = "daily",
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_name: str = "Benchmark",
        show_regime: bool = False,
    ) -> plt.Figure:
        """Rolling skewness and excess kurtosis."""
        ensure_style()
        fig, ax1 = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        active = returns[returns.abs() > 1e-12]
        if not active.empty:
            returns = returns.loc[active.index[0]:]
        skew = returns.rolling(window, min_periods=window).skew().dropna()
        kurt = returns.rolling(window, min_periods=window).kurt().dropna()
        skew, kurt = skew.align(kurt, join="inner")
        if skew.empty:
            raise ValueError("No rolling tail-risk values calculated.")

        regime = None
        if show_regime and benchmark_returns is not None:
            regime = _benchmark_regime_series(benchmark_returns.dropna(), skew.index)
            _shade_regimes(ax1, regime, full_height=True)

        ax1.plot(skew.index, skew.values, color=C.BLUE,
                 lw=C.LW_MAIN, zorder=4, label="Skewness")
        ax1.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)
        ax1.set_ylabel("Skewness", color=C.BLUE)
        ax1.tick_params(axis="y", colors=C.BLUE)

        ax2 = ax1.twinx()
        ax2.patch.set_alpha(0)
        ax2.plot(kurt.index, kurt.values, color=C.BENCHMARK,
                 lw=C.LW_SECONDARY, ls=C.BENCHMARK_LS, zorder=3,
                 label="Excess Kurtosis")
        ax2.axhline(0, color=C.BENCHMARK, lw=C.LW_HAIR, ls=":", alpha=0.55)
        ax2.set_ylabel("Excess Kurtosis", color=C.BENCHMARK)
        ax2.tick_params(axis="y", colors=C.BENCHMARK)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(C.BENCHMARK)
        ax2.spines["top"].set_visible(False)

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Skewness"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_SECONDARY,
                   ls=C.BENCHMARK_LS, label="Excess Kurtosis"),
        ]
        if regime is not None and not regime.empty:
            handles.extend(_regime_legend_handles())
        ax1.legend(handles=handles, loc="upper left", frameon=False,
                   fontsize=C.FONT_TICK, ncol=2 if regime is not None else 1)

        subtitle = (
            f"Window: {window} {reporting_label} obs | Current skew: {skew.iloc[-1]:+.2f} | "
            f"Current excess kurtosis: {kurt.iloc[-1]:+.2f}"
        )
        if regime is not None and not regime.empty:
            subtitle += f" | {_regime_summary_text(regime, benchmark_name)}"
        style_ax(ax1, title="Rolling Tail Risk",
                 subtitle=subtitle)
        smart_date_axis(ax1, skew)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 17f
    @staticmethod
    def plot_benchmark_regime_analysis(
        portfolio_calc: PortfolioCalculations,
        benchmark_returns: pd.Series,
        benchmark_name: str = "Benchmark",
        periods_per_year: Optional[int] = None,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Strategy performance by benchmark trend regime."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        periods_per_year = int(periods_per_year or getattr(portfolio_calc, "trading_days", 252))

        port_ret = portfolio_calc.returns.dropna()
        bench = benchmark_returns.dropna()
        port_ret, bench = port_ret.align(bench, join="inner")
        min_required = max(8, periods_per_year)
        if len(port_ret) < min_required:
            raise ValueError("Insufficient data for benchmark regime analysis.")

        regime = _benchmark_regime_series(bench, port_ret.index)
        aligned_ret = port_ret.reindex(regime.index)

        rows = []
        for name in ["Bull", "Sideways", "Bear"]:
            r = aligned_ret[regime == name].dropna()
            if len(r) < max(3, periods_per_year // 4):
                continue
            total = (1 + r).prod() - 1
            ann = (1 + total) ** (periods_per_year / len(r)) - 1
            vol = r.std() * np.sqrt(periods_per_year)
            sharpe = ann / vol if vol > 0 else np.nan
            rows.append({
                "Regime": name,
                "CAGR": ann * 100,
                "Vol": vol * 100,
                "Sharpe": sharpe,
                "Days": len(r),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No regime buckets with enough observations.")

        colors = [C.BLUE if r == "Bull" else C.BENCHMARK if r == "Sideways" else C.RED
                  for r in df["Regime"]]
        bars = ax.bar(df["Regime"], df["CAGR"], color=colors, alpha=0.86,
                      edgecolor=C.EDGE, linewidth=C.LW_EDGE, zorder=3)
        ax.axhline(0, color=C.SPINE, lw=C.LW_HAIR, zorder=2)

        for bar, _, row in zip(bars, df["Regime"], df.itertuples()):
            if row.CAGR >= 0:
                y_text = row.CAGR + 0.8
                va = "bottom"
                text_color = C.LABEL
            else:
                y_text = row.CAGR / 2
                va = "center"
                text_color = C.FIG_BG
            ax.text(bar.get_x() + bar.get_width() / 2, y_text,
                    f"{row.CAGR:+.1f}%\nS {row.Sharpe:.2f}\n{row.Days}d",
                    ha="center", va=va, fontsize=C.FONT_ANNOT,
                    color=text_color, fontweight="bold")

        y_min = min(0.0, float(df["CAGR"].min()))
        y_max = max(0.0, float(df["CAGR"].max()))
        pad = max((y_max - y_min) * 0.12, 5.0)
        ax.set_ylim(y_min - pad, y_max + pad)

        style_ax(ax, title="Benchmark Regime Analysis",
                 ylabel="Strategy CAGR by Regime (%)",
                 subtitle=f"Frequency: {reporting_label} | Regimes from {benchmark_name}: adaptive trend SMA + momentum")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 18
    @staticmethod
    def plot_rolling_sortino_ratio(
        portfolio_calc: PortfolioCalculations,
        window: int = 252,
        periods_per_year: Optional[int] = None,
        reporting_label: str = "daily",
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_name: str = "Benchmark",
        show_regime: bool = False,
    ) -> plt.Figure:
        """Rolling Sortino ratio with green/red fill zones.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        window : int

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        periods_per_year = int(periods_per_year or getattr(portfolio_calc, "trading_days", 252))

        try:
            sortino = portfolio_calc.compute_rolling_sortino(window=window)
        except Exception:
            # Manual calculation fallback — filter to negative returns only
            ret = portfolio_calc.returns.dropna()
            roll_mean = ret.rolling(window).mean() * periods_per_year
            neg = ret.copy()
            neg[neg >= 0] = np.nan  # NaN out positives (don't zero them)
            roll_down_std = neg.rolling(window, min_periods=max(2, window // 10)).std() * np.sqrt(periods_per_year)
            sortino = (roll_mean / roll_down_std).replace(
                [np.inf, -np.inf], np.nan)

        sortino = sortino.dropna()

        regime = None
        if show_regime and benchmark_returns is not None:
            regime = _benchmark_regime_series(benchmark_returns.dropna(), sortino.index)
            _shade_regimes(ax, regime, full_height=True)

        ax.plot(sortino.index, sortino.values, color=C.BLUE, lw=C.LW_MAIN, zorder=4)

        ax.fill_between(sortino.index, sortino.values, 0,
                        where=sortino.values >= 0, color=C.BLUE, alpha=0.045,
                        interpolate=True, zorder=1)
        ax.fill_between(sortino.index, sortino.values, 0,
                        where=sortino.values < 0, color=C.RED, alpha=0.055,
                        interpolate=True, zorder=1)

        for lev in [0, 1.0]:
            ax.axhline(lev, color=C.SPINE, ls=":", lw=0.55, alpha=0.55)

        avg_sort = _safe_float(sortino.mean())
        ax.axhline(avg_sort, color=C.BENCHMARK, ls=C.BENCHMARK_LS,
                   lw=C.LW_THIN, alpha=0.9, zorder=3)

        endpoint_annotation(ax, sortino, "Current", C.BLUE, fmt="ratio")
        endpoint_annotation(ax, pd.Series(avg_sort, index=[sortino.index[-1]]),
                            "Average", C.BENCHMARK, fmt="ratio", offset=(8, -16))

        time_above_1 = (sortino > 1).mean() * 100
        time_above_0 = (sortino > 0).mean() * 100
        stats_text = (f"Time > 1.0: {time_above_1:.0f}%\n"
                      f"Time > 0.0: {time_above_0:.0f}%")
        _stats_box(ax, stats_text, loc="upper right")

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Sortino Ratio"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                   label=f"Average ({avg_sort:.2f})"),
        ]
        if regime is not None and not regime.empty:
            handles.extend(_regime_legend_handles())
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK, ncol=2 if regime is not None else 1)

        subtitle = (f"Window: {window} {reporting_label} obs  |  Current: {_safe_float(sortino.iloc[-1]):.2f}  |  "
                    f"Average: {avg_sort:.2f}")
        if regime is not None and not regime.empty:
            subtitle += f"  |  {_regime_summary_text(regime, benchmark_name)}"
        style_ax(ax, title="Rolling Sortino Ratio", ylabel="Sortino Ratio",
                 subtitle=subtitle)
        smart_date_axis(ax, sortino)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 19
    @staticmethod
    def plot_rolling_max_drawdown(
        portfolio_calc: PortfolioCalculations,
        window: int = 252,
    ) -> plt.Figure:
        """Rolling maximum drawdown with severity zones.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        window : int

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()

        # Compute full DD series first, then rolling min (avoids staircase)
        cum = (1 + returns).cumprod()
        dd_series = cum / cum.cummax() - 1
        roll_mdd = dd_series.rolling(window, min_periods=window // 2).min()
        roll_mdd = roll_mdd.dropna() * 100  # percent

        ax.fill_between(roll_mdd.index, roll_mdd.values, 0,
                        color=C.RED, alpha=0.14, zorder=1)
        ax.plot(roll_mdd.index, roll_mdd.values, color=C.RED, lw=2.0,
                zorder=4)

        # Severity zones
        for lev, col in [(-10, C.DD_LIGHT), (-20, C.DD_MED),
                         (-30, C.DD_HEAVY)]:
            if roll_mdd.min() < lev:
                ax.axhline(lev, color=col, ls=":", lw=C.LW_HAIR + 0.2, alpha=0.6)
                ax.text(roll_mdd.index[-1], lev, f"  {lev}%", fontsize=C.FONT_SMALL,
                        color=col, va="center")

        endpoint_annotation(ax, roll_mdd, "Current", C.NAVY, fmt="pct")

        avg_mdd = _safe_float(roll_mdd.mean())
        ax.axhline(avg_mdd, color=C.BENCHMARK, ls=C.BENCHMARK_LS,
                   lw=C.LW_SECONDARY - 0.2, zorder=3)

        handles = [
            Line2D([], [], color=C.RED, lw=C.LW_MAIN, label="Rolling Max DD"),
            Line2D([], [], color=C.BENCHMARK, lw=C.LW_THIN, ls=C.BENCHMARK_LS,
                   label=f"Average ({avg_mdd:.1f}%)"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=False,
                  fontsize=C.FONT_TICK)

        subtitle = (f"Window: {window}d  |  Current: {_safe_float(roll_mdd.iloc[-1]):.1f}%  |  "
                    f"Average: {avg_mdd:.1f}%")
        style_ax(ax, title="Rolling Maximum Drawdown",
                 ylabel="Max Drawdown (%)", subtitle=subtitle)
        smart_date_axis(ax, roll_mdd)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 20
    @staticmethod
    def plot_asset_contribution_to_risk(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: InstrumentCalculations,
    ) -> plt.Figure:
        """Horizontal bar chart of each asset's contribution to total risk.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        instrument_calc : InstrumentCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        weights = portfolio_calc.portfolio_data.weights
        if isinstance(weights, pd.DataFrame):
            weights = weights.iloc[-1]
        if not isinstance(weights, pd.Series) or weights.empty:
            raise ValueError("Weights must be a non-empty pandas Series.")

        returns = instrument_calc.returns
        if returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        cov_mat = returns.cov()
        port_var = np.dot(weights.values, np.dot(cov_mat.values, weights.values))
        if port_var <= 0:
            raise ValueError("Portfolio variance is non-positive.")
        marginal = np.dot(cov_mat.values, weights.values) / np.sqrt(port_var)
        risk_contrib = weights.values * marginal
        total_risk = risk_contrib.sum()
        risk_pct = risk_contrib / total_risk if total_risk != 0 else risk_contrib

        labels = [_clean_label(c) for c in weights.index]
        rc_series = pd.Series(risk_pct, index=labels).sort_values()

        colors = [
            mcolors.to_rgba(C.BLUE, 0.88) if v >= 0
            else mcolors.to_rgba(C.RED, 0.84)
            for v in rc_series.values
        ]
        bars = ax.barh(rc_series.index, rc_series.values * 100,
                       color=colors, edgecolor=C.EDGE, linewidth=C.LW_EDGE,
                       height=0.6, zorder=3)

        # Value labels
        for bar, val in zip(bars, rc_series.values * 100):
            x_pos = val + (1.0 if val >= 0 else -1.0)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", ha=ha, va="center", fontsize=8,
                    color=C.LABEL)

        ax.axvline(0, color=C.SPINE, lw=0.7)

        style_ax(ax, title="Asset Contribution to Portfolio Risk",
                 xlabel="Risk Contribution (%)", ylabel="")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 21
    @staticmethod
    def plot_rolling_asset_correlations(
        instrument_calc: InstrumentCalculations,
        window: int = 252,
        reporting_label: str = "daily",
    ) -> plt.Figure:
        """Rolling pairwise correlations for top-5 most volatile pairs,
        with aggregate band (mean +/- 1 sigma).

        Parameters
        ----------
        instrument_calc : InstrumentCalculations
        window : int

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()

        cols = returns.columns.tolist()
        n_assets = len(cols)

        # Build all pairwise rolling correlations
        pairs: List[Tuple[str, str]] = []
        pair_corrs: List[pd.Series] = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                rc = returns.iloc[:, i].rolling(window).corr(returns.iloc[:, j])
                pairs.append((_clean_label(cols[i]), _clean_label(cols[j])))
                pair_corrs.append(rc)

        if not pair_corrs:
            ax.text(0.5, 0.5, "Insufficient assets for correlations",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color=C.MUTED)
            style_ax(ax, title="Rolling Asset Correlations")
            fig.tight_layout()
            add_watermark(fig)
            return fig

        # Select top-5 most volatile pairs
        volatilities = [pc.std() for pc in pair_corrs]
        top_idx = np.argsort(volatilities)[-min(5, len(volatilities)):]

        for idx_i, pidx in enumerate(top_idx):
            pc = pair_corrs[pidx].dropna()
            label = f"{pairs[pidx][0]} / {pairs[pidx][1]}"
            color = mcolors.to_rgba(_ALLOCATION_COLORS[idx_i % len(_ALLOCATION_COLORS)], 0.96)
            ax.plot(pc.index, pc.values, color=color, lw=C.LW_SECONDARY,
                    label=label, zorder=3)

        # Aggregate band
        all_corrs = pd.concat(pair_corrs, axis=1).dropna(how="all")
        mean_corr = all_corrs.mean(axis=1)
        std_corr = all_corrs.std(axis=1)
        ax.fill_between(mean_corr.index,
                        (mean_corr - std_corr).values,
                        (mean_corr + std_corr).values,
                        color=C.BLUE, alpha=0.09, zorder=1,
                        label="Mean +/- 1 Sigma")
        ax.plot(mean_corr.index, mean_corr.values, color=C.BLUE, lw=C.LW_HAIR + 0.4,
                ls="--", zorder=2)

        # Reference lines
        for lev in [0.5, 0.7]:
            ax.axhline(lev, color=C.SPINE, ls=":", lw=0.6, alpha=0.4)
        ax.axhline(0, color=C.SPINE, ls="-", lw=0.7)

        ax.legend(loc="upper left", framealpha=0.92, fontsize=8,
                  edgecolor=C.GRID, ncol=2)

        style_ax(ax, title="Rolling Asset Correlations",
                 ylabel="Correlation",
                 subtitle=f"Window: {window} {reporting_label} obs  |  Top-5 most volatile pairs")
        smart_date_axis(ax, mean_corr)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 22
    @staticmethod
    def plot_omega_curve(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Omega ratio across return thresholds with fill regions.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()

        thresholds = np.linspace(-0.02, 0.02, 200)

        def _omega(ret, thr):
            gains = np.maximum(ret - thr, 0).mean()
            losses = np.maximum(thr - ret, 0).mean()
            return gains / losses if losses > 0 else np.nan

        omegas = np.array([_omega(returns.values, t) for t in thresholds])

        ax.plot(thresholds * 100, omegas, color=C.BLUE, lw=C.LW_MAIN, zorder=4)

        # Fill zones
        ax.fill_between(thresholds * 100, omegas, 1.0,
                        where=omegas > 1.0, color=C.GREEN, alpha=0.12,
                        interpolate=True, zorder=1)
        ax.fill_between(thresholds * 100, omegas, 1.0,
                        where=omegas <= 1.0, color=C.RED, alpha=0.10,
                        interpolate=True, zorder=1)

        ax.axhline(1.0, color=C.SPINE, ls=":", lw=0.7)
        ax.axvline(0, color=C.SPINE, ls="-", lw=0.7)

        # Current omega at threshold=0
        omega_at_zero = _omega(returns.values, 0.0)
        if np.isfinite(omega_at_zero):
            ax.plot(0, omega_at_zero, "o", color=C.BENCHMARK, markersize=C.MARKER_SM, zorder=6)
            ax.annotate(
                f"Omega(0): {omega_at_zero:.2f}",
                xy=(0, omega_at_zero), xytext=(12, 8),
                textcoords="offset points", fontsize=8.5,
                fontweight="bold", color=C.BENCHMARK,
                bbox=dict(boxstyle="round,pad=0.3", fc=C.FIG_BG,
                          ec=C.BENCHMARK, alpha=0.9, lw=0.6),
            )

        # Clip y for readability
        finite_omegas = omegas[np.isfinite(omegas)]
        if len(finite_omegas) > 0:
            y_max = min(np.percentile(finite_omegas, 95) * 1.3, 10)
            ax.set_ylim(0, max(y_max, 2.5))

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Omega Curve"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  fontsize=C.FONT_TICK)

        subtitle = f"Omega at 0%: {omega_at_zero:.2f}" if np.isfinite(omega_at_zero) else ""
        style_ax(ax, title="Omega Curve",
                 xlabel="Return Threshold (%)", ylabel="Omega Ratio",
                 subtitle=subtitle)

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 23
    @staticmethod
    def plot_decile_performance(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Bar chart of average returns by decile of daily returns.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()

        # Split into 10 deciles
        sorted_ret = returns.sort_values()
        decile_means = []
        n = len(sorted_ret)
        chunk_size = n // 10
        for i in range(10):
            start = i * chunk_size
            end = start + chunk_size if i < 9 else n
            decile_means.append(sorted_ret.iloc[start:end].mean() * 100)

        decile_labels = [f"D{i+1}" for i in range(10)]

        # Colour gradient: red (worst decile) -> blue (best decile)
        cmap_gradient = mcolors.LinearSegmentedColormap.from_list(
            "decile_grad", [C.RED, C.ORANGE, C.TEAL, C.BLUE], N=10)
        bar_colors = [cmap_gradient(i / 9) for i in range(10)]

        bars = ax.bar(decile_labels, decile_means, color=bar_colors,
                      edgecolor=C.EDGE, linewidth=C.LW_EDGE + 0.3, width=0.75, zorder=3,
                      alpha=C.FILL_HEAVY + 0.05)

        # Value labels
        for bar, val in zip(bars, decile_means):
            y_off = 0.02 if val >= 0 else -0.02
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                    f"{val:.2f}%", ha="center", va=va, fontsize=8,
                    color=C.LABEL)

        ax.axhline(0, color=C.SPINE, lw=0.7)

        style_ax(ax, title="Decile Performance",
                 xlabel="Return Decile (worst to best)",
                 ylabel="Average Return (%)",
                 subtitle="Average daily return per decile bucket")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 24
    @staticmethod
    def plot_expected_return_profile(
        portfolio_calc: PortfolioCalculations,
    ) -> plt.Figure:
        """Distribution of expected returns across different holding periods
        with confidence intervals.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = portfolio_calc.returns.dropna()
        cum = (1 + returns).cumprod()

        # Various holding windows
        max_window = min(len(cum) // 3, 504)
        windows = list(range(5, max_window, max(1, max_window // 30)))

        means = []
        p5s = []
        p95s = []
        for w in windows:
            rolling = cum.rolling(w).apply(
                lambda p: (p.iloc[-1] / p.iloc[0]) - 1, raw=False
            ).dropna()
            if len(rolling) == 0:
                means.append(np.nan)
                p5s.append(np.nan)
                p95s.append(np.nan)
            else:
                means.append(rolling.mean())
                p5s.append(rolling.quantile(0.05))
                p95s.append(rolling.quantile(0.95))

        means = np.array(means) * 100
        p5s = np.array(p5s) * 100
        p95s = np.array(p95s) * 100

        ax.plot(windows, means, color=C.BLUE, lw=C.LW_MAIN, zorder=4,
                label="Expected Return")
        ax.fill_between(windows, p5s, p95s, color=C.TEAL, alpha=0.10,
                        zorder=1, label="5th-95th Percentile")

        ax.axhline(0, color=C.SPINE, ls="-", lw=0.7)

        # Reference markers for common holding periods
        ref_periods = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
        for label, d in ref_periods.items():
            if d <= max(windows):
                idx = min(range(len(windows)), key=lambda i: abs(windows[i] - d))
                ax.plot(windows[idx], means[idx], "o", color=C.BENCHMARK,
                        markersize=C.MARKER_SM - 1, zorder=6)
                ax.annotate(label, (windows[idx], means[idx]),
                            xytext=(0, 8), textcoords="offset points",
                            fontsize=C.FONT_SMALL, ha="center", color=C.BENCHMARK,
                            fontweight="bold")

        # Stats box with key holding period stats
        stats_lines = []
        for label, d in ref_periods.items():
            if d <= max(windows):
                idx = min(range(len(windows)), key=lambda i: abs(windows[i] - d))
                stats_lines.append(
                    f"{label}: {means[idx]:+.1f}% [{p5s[idx]:+.1f}%, {p95s[idx]:+.1f}%]"
                )
        if stats_lines:
            _stats_box(ax, "\n".join(stats_lines), loc="lower right")

        # Breakeven day (where 5th percentile crosses 0)
        for i in range(len(p5s)):
            if np.isfinite(p5s[i]) and p5s[i] >= 0:
                breakeven = windows[i]
                ax.axvline(breakeven, color=C.TEAL, ls=":", lw=0.8, alpha=0.6)
                ax.text(breakeven, ax.get_ylim()[1] * 0.95,
                        f"  Breakeven: {breakeven}d",
                        fontsize=C.FONT_SMALL, color=C.TEAL, va="top")
                break

        handles = [
            Line2D([], [], color=C.BLUE, lw=C.LW_MAIN, label="Expected Return"),
            mpatches.Patch(facecolor=C.TEAL, alpha=0.10,
                           label="5th-95th Percentile"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=C.FONT_TICK)

        style_ax(ax, title="Expected Return Profile",
                 xlabel="Holding Period (Days)", ylabel="Expected Return (%)",
                 subtitle="Mean return and confidence band by holding period")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ------------------------------------------------------------------ 25
    @staticmethod
    def plot_rolling_weights(
        portfolio_calc: PortfolioCalculations,
        instrument_calc: Optional[InstrumentCalculations] = None,
        window: int = 30,
    ) -> plt.Figure:
        """Stacked area of smoothed rolling weights over time.

        Parameters
        ----------
        portfolio_calc : PortfolioCalculations
        instrument_calc : InstrumentCalculations, optional
        window : int
            Rolling window for smoothing.

        Returns
        -------
        plt.Figure
        """
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        weights = portfolio_calc.portfolio_data.weights
        if weights is None or (hasattr(weights, "empty") and weights.empty):
            raise ValueError("Weights data is empty.")
        if isinstance(weights, pd.Series):
            weights = weights.to_frame()

        smooth = weights.rolling(window, min_periods=1).mean()
        labels = [_clean_label(c) for c in smooth.columns]
        colors = _allocation_colors(labels, asset_alpha=0.92, cash_alpha=0.64)

        ax.stackplot(smooth.index,
                     *[smooth.iloc[:, i] for i in range(smooth.shape[1])],
                     labels=labels, colors=colors, alpha=1.0, linewidth=0)

        style_ax(ax, title=f"Rolling Weights ({window}-day smoothed)",
                 ylabel="Weight")
        smart_date_axis(ax, smooth)

        handles = [mpatches.Patch(facecolor=colors[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95, fontsize=9.5, edgecolor=C.SPINE, borderpad=0.5)

        fig.tight_layout()
        add_watermark(fig)
        return fig
