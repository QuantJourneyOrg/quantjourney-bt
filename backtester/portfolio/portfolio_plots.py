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

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import calendar
from typing import Optional, Tuple, List, Dict
from scipy import stats
from matplotlib.lines import Line2D

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    sns = None

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


def _heatmap(data, *, ax, mask=None, annot=False, fmt=".2f", cmap=None,
             vmin=None, vmax=None, center=None, square=False, annot_kws=None,
             cbar=True, cbar_kws=None, linewidths=0.0, linecolor=None, **kwargs):
    """Draw a seaborn heatmap when available, otherwise use Matplotlib."""
    if sns is not None:
        return sns.heatmap(
            data,
            ax=ax,
            mask=mask,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            square=square,
            annot_kws=annot_kws,
            cbar=cbar,
            cbar_kws=cbar_kws,
            linewidths=linewidths,
            linecolor=linecolor,
            **kwargs,
        )

    frame = pd.DataFrame(data)
    values = frame.to_numpy(dtype=float)
    if mask is not None:
        values = np.ma.array(values, mask=np.asarray(mask, dtype=bool))

    im = ax.imshow(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal" if square else "auto",
    )
    ax.set_xticks(np.arange(frame.shape[1]))
    ax.set_yticks(np.arange(frame.shape[0]))
    ax.set_xticklabels([str(c) for c in frame.columns])
    ax.set_yticklabels([str(i) for i in frame.index])

    if linewidths:
        ax.set_xticks(np.arange(-0.5, frame.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, frame.shape[0], 1), minor=True)
        ax.grid(which="minor", color=linecolor or C.FIG_BG, linewidth=linewidths)
        ax.tick_params(which="minor", bottom=False, left=False)

    if annot:
        text_style = annot_kws or {}
        mask_values = np.ma.getmaskarray(values)
        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                value = frame.iat[row, col]
                if mask_values[row, col] or pd.isna(value):
                    continue
                ax.text(col, row, format(float(value), fmt), ha="center",
                        va="center", **text_style)

    if cbar:
        cbar_kws = cbar_kws or {}
        label = cbar_kws.get("label")
        shrink = cbar_kws.get("shrink", 1.0)
        cb = ax.figure.colorbar(im, ax=ax, shrink=shrink)
        if label:
            cb.set_label(label)
    return im


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
        _heatmap(
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
            _heatmap(
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

        _heatmap(
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
    # ------------------------------------------------------------------ 15
    # ------------------------------------------------------------------ 15b
    # ------------------------------------------------------------------ 16
    # ------------------------------------------------------------------ 17
    # ------------------------------------------------------------------ 17b
    # ------------------------------------------------------------------ 17c
    # ------------------------------------------------------------------ 17d
    # ------------------------------------------------------------------ 17e
    # ------------------------------------------------------------------ 17f
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
    # ------------------------------------------------------------------ 20
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
    # ------------------------------------------------------------------ 23
    # ------------------------------------------------------------------ 24
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
