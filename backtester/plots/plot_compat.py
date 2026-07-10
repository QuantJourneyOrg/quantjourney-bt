"""
Unified QuantJourney public plotting style helpers.
==================================================

Single source of truth for all plotting styles, colours, helpers.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

__all__ = [
    "C",
    "apply_institutional_style",
    "ensure_style",
    "reset_style",
    "add_watermark",
    "style_ax",
    "smart_date_axis",
    "endpoint_annotation",
    "endpoint_annotations_pair",
    "stats_box",
    "fmt_pct",
    "fmt_ratio",
    "make_figure",
    "diverging_cmap",
    "LegendStats",
    "ColorMap",
    "ColorUtilities",
    "LegendUtilities",
    "PlotUtilities",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtester.plots.theme.types import ThemeConfig


# ═══════════════════════════════════════════════════════════════════════════
# FONT RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

_FONT_CANDIDATES = [
    "Inter",
    "Helvetica Neue",
    "SF Pro Display",
    "Avenir Next",
    "Segoe UI",
    "Source Sans Pro",
    "Roboto",
    "Verdana",
    "DejaVu Sans",
]


@lru_cache(maxsize=1)
def _resolve_font() -> str:
    """Pick the first available font from the candidate list."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in _FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return "sans-serif"


# ═══════════════════════════════════════════════════════════════════════════
# COLOUR SYSTEM — single source of truth
# ═══════════════════════════════════════════════════════════════════════════


class C:
    """Blue monochromatic institutional colour system.

    Every chart uses shades of blue + one muted-gold accent.
    This creates the cohesive, Citadel-grade look across all plots.
    """

    # ── Blue scale (dark → light) ──
    NAVY = "#0A1F38"  # Deepest navy — primary lines, titles
    DARK = "#14395A"  # Dark blue
    MID = "#1A5276"  # Medium blue
    BLUE = "#2471A3"  # Standard blue — rich and vibrant
    STEEL = "#4A9BD9"  # Steel blue — brighter
    LIGHT = "#7CB9E8"  # Light blue — more saturated
    PALE = "#A7D2F0"  # Pale blue
    ICE = "#D4E9F7"  # Ice blue — lightest fills

    # ── Warm accents (used for key annotations and highlights) ──
    GOLD = "#C49A3C"  # Rich gold — averages, highlights
    AMBER = "#E8A838"  # Bright amber — strong callouts

    # ── Semantic aliases ──
    TEAL = "#177E89"  # True teal — distinct hue for contrast
    ORANGE = "#D4842A"  # Warm orange — distinct from gold
    RED = "#C0392B"  # Vibrant red — negative elements (more visible)
    GREEN = "#27864B"  # Rich green — positive elements (more visible)
    PURPLE = "#6C3483"  # True purple — additional accent

    # ── Extended palette (15 colours with hue variation for multi-series) ──
    PALETTE = [
        "#0A1F38",  # Deep navy
        "#2471A3",  # Rich blue
        "#177E89",  # Teal
        "#14395A",  # Dark blue
        "#4A9BD9",  # Steel blue
        "#C49A3C",  # Gold
        "#1A5276",  # Mid blue
        "#27864B",  # Green
        "#7CB9E8",  # Light blue
        "#5B677A",  # Slate
        "#3498DB",  # Bright blue
        "#D4842A",  # Orange
        "#1F618D",  # Royal blue
        "#2C8C6F",  # Emerald
        "#5DADE2",  # Sky blue
    ]

    # ── Chrome / UI ──
    BG = "#FAFBFC"  # Subtle warm-grey tint (not clinical white)
    FIG_BG = "#FAFBFC"
    SPINE = "#CBD2DA"  # Slightly stronger spine
    GRID = "#E2E7ED"  # More visible grid
    TICK = "#5A6577"  # Darker ticks for readability
    LABEL = "#2D3748"  # Darker labels
    TITLE = "#0F1724"  # Near-black titles
    SUBTITLE = "#4A5568"  # Clear subtitle
    WATERMARK = "#B0B8C4"
    MUTED = "#8896A7"  # Slightly darker muted

    # ── Heatmap / diverging (red → white → navy) ──
    HM_NEG = "#C0392B"  # Brighter red for heatmaps
    HM_ZERO = "#FAFBFC"
    HM_POS = "#0A1F38"  # Deep navy for positive

    # ── Up / Down ──
    UP = "#27864B"  # Green — brighter than before
    DOWN = "#C0392B"  # Red — brighter than before

    # ── Drawdown severity gradient (light → dark blue) ──
    DD_LIGHT = "#A7D2F0"
    DD_MED = "#4A9BD9"
    DD_HEAVY = "#1A5276"
    DD_SEVERE = "#0A1F38"

    # ── Line width hierarchy ──
    LW_MAIN = 2.0  # Primary data lines
    LW_SECONDARY = 1.6  # Benchmark, MA lines
    LW_THIN = 1.0  # Reference/average lines
    LW_HAIR = 0.5  # Zone lines, subtle refs
    LW_EDGE = 0.5  # Bar/histogram edges

    # ── Fill opacity hierarchy ──
    FILL_MAIN = 0.28  # Gradient under main line
    FILL_LIGHT = 0.18  # Rolling metric fills
    FILL_HEAVY = 0.80  # Stacked area, bars
    FILL_HIST = 0.65  # Histograms

    # ── Benchmark ──
    BENCHMARK = "#C49A3C"  # Same as GOLD default
    BENCHMARK_LS = "--"

    # ── Edge color (bars/histograms) ──
    EDGE = "white"

    # ── Marker sizes ──
    MARKER_SM = 6
    MARKER_LG = 50

    # ── Font sizes ──
    FONT_TITLE = 15
    FONT_LABEL = 11
    FONT_TICK = 10
    FONT_ANNOT = 9
    FONT_SMALL = 7

    @staticmethod
    def get(n: int) -> list[str]:
        """Return *n* colours, cycling the palette if needed."""
        return [C.PALETTE[i % len(C.PALETTE)] for i in range(n)]

    @classmethod
    def apply_theme(cls, config: ThemeConfig) -> None:
        """Bridge: update C class attributes from a ThemeConfig.

        This is the key mechanism that makes themes work without
        touching any plot function code.
        """
        cs = config.colors
        # Blue scale
        cls.NAVY = cs.PRIMARY
        cls.DARK = cs.DARK_BLUE
        cls.MID = cs.MID_BLUE
        cls.BLUE = cs.LIGHT_BLUE
        cls.STEEL = cs.SOFT_BLUE
        cls.LIGHT = cs.LIGHT_BLUE
        cls.PALE = cs.FILL
        cls.ICE = cs.FILL
        # Accents
        cls.GOLD = cs.GOLD
        cls.AMBER = cs.AMBER
        cls.TEAL = cs.TEAL
        cls.ORANGE = cs.ORANGE
        cls.RED = cs.NEGATIVE
        cls.GREEN = cs.POSITIVE
        cls.PURPLE = cs.PURPLE
        # Chrome / UI
        cls.BG = cs.BACKGROUND
        cls.FIG_BG = cs.BACKGROUND2
        cls.SPINE = cs.SPINES
        cls.GRID = cs.GRID
        cls.TICK = cs.TICK
        cls.LABEL = cs.LABELS
        cls.TITLE = cs.TITLE
        cls.SUBTITLE = cs.SUBTITLE
        cls.WATERMARK = cs.WATERMARK
        cls.MUTED = cs.MUTED
        # Heatmap diverging
        cls.HM_NEG = cs.HM_NEG
        cls.HM_ZERO = cs.HM_ZERO
        cls.HM_POS = cs.HM_POS
        # Up / Down
        cls.UP = cs.UP
        cls.DOWN = cs.DOWN
        # Drawdown gradient
        cls.DD_LIGHT = cs.DD_LIGHT
        cls.DD_MED = cs.DD_MED
        cls.DD_HEAVY = cs.DD_HEAVY
        cls.DD_SEVERE = cs.DD_SEVERE
        # Multi-series palette
        cls.PALETTE = list(cs.VIZ_CATEGORICAL)
        # Line widths
        cls.LW_MAIN = config.LW_MAIN
        cls.LW_SECONDARY = config.LW_SECONDARY
        cls.LW_THIN = config.LW_THIN
        cls.LW_HAIR = config.LW_HAIR
        cls.LW_EDGE = config.LW_EDGE
        # Fills
        cls.FILL_MAIN = config.FILL_MAIN
        cls.FILL_LIGHT = config.FILL_LIGHT
        cls.FILL_HEAVY = config.FILL_HEAVY
        cls.FILL_HIST = config.FILL_HIST
        # Benchmark
        cls.BENCHMARK = config.BENCHMARK_COLOR or cs.GOLD
        cls.BENCHMARK_LS = config.BENCHMARK_LS
        # Edge
        cls.EDGE = config.EDGE_COLOR
        # Markers
        cls.MARKER_SM = config.MARKER_SM
        cls.MARKER_LG = config.MARKER_LG
        # Fonts
        cls.FONT_TITLE = config.TITLE_FONTSIZE
        cls.FONT_LABEL = config.LABEL_FONTSIZE
        cls.FONT_TICK = config.TICK_FONTSIZE
        cls.FONT_ANNOT = config.FONT_ANNOT
        cls.FONT_SMALL = config.FONT_SMALL

    @classmethod
    def reset(cls) -> None:
        """Reset all attributes to original institutional defaults."""
        for attr, val in cls._DEFAULTS.items():
            setattr(cls, attr, val)


# Capture original C defaults for reset
C._DEFAULTS = {
    k: (list(v) if isinstance(v, list) else v)
    for k, v in vars(C).items()
    if isinstance(v, (str, list, int, float)) and not k.startswith("_")
}


# Backward-compatible alias
class ColorMap:
    """Legacy colour registry — delegates to C for consistency."""

    PLOT_PRIMARY = C.NAVY
    PLOT_SECONDARY = C.BENCHMARK
    PLOT_BACKGROUND = C.BG
    FIGURE_BACKGROUND = C.FIG_BG
    SPINE = C.SPINE
    GRID = C.GRID
    TITLE = C.TITLE
    LABEL = C.LABEL
    TICK = C.TICK
    WATERMARK = C.WATERMARK


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════


def apply_institutional_style(config: ThemeConfig | None = None) -> None:
    """Apply style to matplotlib rcParams.

    Reads colours from C (which may have been themed) and optionally
    reads non-colour settings (figsize, dpi, font, grid) from *config*.
    When called with no arguments, behaves identically to the original.
    """
    font = _resolve_font()

    if config is not None:
        font_family = config.FONTFAMILY
        font_size = config.FONTSIZE
        title_size = config.TITLE_FONTSIZE
        label_size = config.LABEL_FONTSIZE
        tick_size = config.TICK_FONTSIZE
        legend_size = config.LEGEND_FONTSIZE
        figsize = config.FIGSIZE
        dpi = config.DPI
        linewidth = config.LINEWIDTH
        grid_alpha = config.GRID_ALPHA
        grid_ls = config.GRID_LINESTYLE
        grid_lw = config.GRID_LINEWIDTH
        spine_lw = config.SPINE_LINEWIDTH
        title_pad = config.TITLE_PAD
        label_pad = config.LABEL_PAD
        tick_pad = config.TICK_PAD
        legend_framealpha = config.LEGEND_FRAMEALPHA
        legend_edgecolor = config.LEGEND_EDGECOLOR
        hide_top_right = not config.SPINE_VISIBLE
    else:
        font_family = "sans-serif"
        font_size = 11
        title_size = 15
        label_size = 11
        tick_size = 10
        legend_size = 10
        figsize = (14, 7)
        dpi = 200
        linewidth = 1.8
        grid_alpha = 0.85
        grid_ls = "-"
        grid_lw = 0.5
        spine_lw = 0.7
        title_pad = 18
        label_pad = 10
        tick_pad = 5
        legend_framealpha = 0.95
        legend_edgecolor = C.SPINE
        hide_top_right = True

    plt.rcParams.update(
        {
            # Typography
            "font.family": font_family,
            "font.sans-serif": [font, "Helvetica Neue", "Arial", "sans-serif"],
            "font.size": font_size,
            # Figure
            "figure.facecolor": C.FIG_BG,
            "figure.edgecolor": "none",
            "figure.dpi": dpi,
            "figure.figsize": figsize,
            # Axes
            "axes.facecolor": C.BG,
            "axes.edgecolor": C.SPINE,
            "axes.linewidth": spine_lw,
            "axes.labelsize": label_size,
            "axes.labelcolor": C.LABEL,
            "axes.titlesize": title_size,
            "axes.titleweight": "bold",
            "axes.titlepad": title_pad,
            "axes.labelpad": label_pad,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": not hide_top_right,
            "axes.spines.right": not hide_top_right,
            # Grid
            "grid.color": C.GRID,
            "grid.linewidth": grid_lw,
            "grid.alpha": grid_alpha,
            "grid.linestyle": grid_ls,
            # Ticks
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "xtick.color": C.TICK,
            "ytick.color": C.TICK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": spine_lw,
            "ytick.major.width": spine_lw,
            "xtick.major.pad": tick_pad,
            "ytick.major.pad": tick_pad,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": legend_framealpha,
            "legend.edgecolor": legend_edgecolor,
            "legend.fontsize": legend_size,
            "legend.borderpad": 0.7,
            "legend.labelspacing": 0.45,
            "legend.handlelength": 2.0,
            # Lines
            "lines.linewidth": linewidth,
            "lines.antialiased": True,
            # Savefig
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "savefig.facecolor": C.FIG_BG,
        }
    )


_style_applied = False
_current_theme_id: int | None = None


def ensure_style(config: ThemeConfig | None = None) -> None:
    """Apply style once, or re-apply if the theme changed."""
    global _style_applied, _current_theme_id
    theme_id = id(config) if config is not None else None
    if not _style_applied or theme_id != _current_theme_id:
        apply_institutional_style(config)
        _style_applied = True
        _current_theme_id = theme_id


def reset_style() -> None:
    """Force re-application of style on the next ensure_style() call."""
    global _style_applied, _current_theme_id
    _style_applied = False
    _current_theme_id = None


# ═══════════════════════════════════════════════════════════════════════════
# WATERMARK
# ═══════════════════════════════════════════════════════════════════════════


def add_watermark(fig: plt.Figure, text: str = "Source: QuantJourney Backtester") -> None:
    """Add subtle source attribution and generation timestamp."""
    from datetime import datetime

    fig.text(
        0.01,
        0.01,
        text,
        fontsize=7,
        color=C.MUTED,
        ha="left",
        va="bottom",
        style="italic",
        alpha=0.6,
        transform=fig.transFigure,
    )
    generated = datetime.now().astimezone().strftime("Generated: %Y-%m-%d %H:%M %Z")
    fig.text(
        0.99,
        0.01,
        generated,
        fontsize=7,
        color=C.MUTED,
        ha="right",
        va="bottom",
        alpha=0.6,
        transform=fig.transFigure,
    )


# ═══════════════════════════════════════════════════════════════════════════
# AXIS STYLING HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def style_ax(
    ax: plt.Axes,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    subtitle: str = "",
) -> None:
    """Apply institutional styling to an axes object."""
    if title:
        ax.set_title(
            title,
            color=C.TITLE,
            fontsize=C.FONT_TITLE,
            fontweight="bold",
            pad=14,
            loc="left",
        )
    if subtitle:
        ax.text(
            1.0,
            1.03,
            subtitle,
            transform=ax.transAxes,
            fontsize=C.FONT_ANNOT,
            color=C.SUBTITLE,
            ha="right",
            va="bottom",
            clip_on=False,
            zorder=20,
        )
    if ylabel:
        ax.set_ylabel(ylabel, color=C.LABEL, fontsize=C.FONT_LABEL, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=C.LABEL, fontsize=C.FONT_LABEL, labelpad=10)
    ax.tick_params(
        axis="both",
        which="major",
        length=3,
        width=0.55,
        colors=C.TICK,
        labelsize=C.FONT_TICK,
        pad=5,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C.SPINE)
        ax.spines[spine].set_linewidth(0.55)


def smart_date_axis(ax: plt.Axes, data: Any) -> None:
    """Auto-configure date axis based on data time span."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        idx = data.index
    else:
        return

    if not isinstance(idx, pd.DatetimeIndex):
        return

    if len(idx) < 2:
        return

    span = idx[-1] - idx[0]
    span_days = span.days
    span_seconds = max(span.total_seconds(), 0)

    has_intraday_time = not ((idx.hour == 0) & (idx.minute == 0) & (idx.second == 0)).all()
    median_delta = pd.Series(idx).diff().dropna().median()
    is_intraday = has_intraday_time or (
        pd.notna(median_delta) and median_delta < pd.Timedelta(days=1)
    )

    if is_intraday:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        if span_seconds <= 2 * 86400:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        else:
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel("Date / time")
        ax.tick_params(axis="x", rotation=0)
        ax.set_xlim(idx[0], idx[-1])
        return

    if span_days > 3650:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        fmt = "'%y"
    elif span_days > 1825:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        fmt = "'%y"
    elif span_days > 365:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        fmt = "%b '%y"
    elif span_days > 90:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        fmt = "%b '%y"
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fmt = "%d %b"

    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.tick_params(axis="x", rotation=0)

    # Tight x-axis: data touches left/right edges (institutional style)
    ax.set_xlim(idx[0], idx[-1])


def endpoint_annotation(
    ax: plt.Axes,
    series: pd.Series,
    label: str,
    color: str,
    fmt: str = "ratio",
    offset: tuple[int, int] = (8, 0),
) -> None:
    """Add a labelled dot + text at the end of a time series."""
    clean = series.dropna()
    if len(clean) == 0:
        return

    last_val = clean.iloc[-1]
    last_date = clean.index[-1]

    if fmt == "pct":
        txt = fmt_pct(last_val)
    elif fmt == "currency":
        txt = f"${last_val:,.0f}"
    else:
        txt = fmt_ratio(last_val)

    ax.plot(
        last_date,
        last_val,
        "o",
        color=color,
        markersize=C.MARKER_SM,
        zorder=10,
        markeredgecolor=C.EDGE,
        markeredgewidth=0.6,
    )
    ax.annotate(
        f"{label}: {txt}",
        xy=(last_date, last_val),
        xytext=(offset[0], offset[1]),
        textcoords="offset points",
        fontsize=C.FONT_ANNOT,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.12",
            fc=C.FIG_BG,
            ec="none",
            alpha=0.62,
            lw=0.0,
        ),
    )


def endpoint_annotations_pair(
    ax: plt.Axes,
    series: pd.Series,
    avg_value: float,
    series_color: str | None = None,
    avg_color: str | None = None,
    series_label: str = "Current",
    avg_label: str = "Average",
    fmt: str = "ratio",
    min_gap_pts: int = 22,
) -> None:
    """Place two endpoint annotations with guaranteed vertical separation."""
    series_color = series_color or C.NAVY
    avg_color = avg_color or C.BENCHMARK
    clean = series.dropna()
    if len(clean) == 0:
        return

    last_val = clean.iloc[-1]
    last_date = clean.index[-1]
    data_range = clean.max() - clean.min()
    if data_range == 0:
        data_range = abs(last_val) or 1.0

    val_gap_normalized = abs(last_val - avg_value) / data_range

    if val_gap_normalized < 0.08:
        off_current = (8, min_gap_pts // 2 + 6)
        off_avg = (8, -(min_gap_pts // 2 + 6))
    else:
        off_current = (8, 0)
        off_avg = (8, 0)

    endpoint_annotation(ax, series, series_label, series_color, fmt=fmt, offset=off_current)
    avg_series = pd.Series(avg_value, index=[last_date])
    endpoint_annotation(ax, avg_series, avg_label, avg_color, fmt=fmt, offset=off_avg)


def stats_box(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    """Place a semi-transparent stats box on *ax*."""
    x, y, ha, va = {
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
    }.get(loc, (0.02, 0.97, "left", "top"))
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=C.FONT_ANNOT,
        color=C.LABEL,
        ha=ha,
        va=va,
        linespacing=1.35,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc=C.FIG_BG, ec=C.SPINE, alpha=0.88, lw=0.55),
    )


def fmt_pct(v: float, decimals: int = 1) -> str:
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.{decimals}f}%"


def fmt_ratio(v: float, decimals: int = 2) -> str:
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def make_figure(
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure + axes with institutional styling pre-applied."""
    ensure_style()
    figsize = figsize or plt.rcParams.get("figure.figsize", (14, 7))
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def diverging_cmap(
    neg: str | None = None,
    zero: str | None = None,
    pos: str | None = None,
) -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap for heatmaps."""
    neg = neg or C.HM_NEG
    zero = zero or C.HM_ZERO
    pos = pos or C.HM_POS
    return mcolors.LinearSegmentedColormap.from_list(
        "qj_diverging", [neg, "#F2B8B5", zero, "#60A5FA", pos]
    )


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY COMPAT: LegendStats / ColorUtilities / LegendUtilities / PlotUtilities
# ═══════════════════════════════════════════════════════════════════════════


class LegendStats(Enum):
    TOTAL_CHANGE = "total_change"
    AVG_STD_SKEW_KURT = "avg_std_skew_kurt"
    AVG = "avg"
    LAST = "last"


class ColorUtilities:
    _PALETTE = C.PALETTE

    @staticmethod
    def get_n_colors(n: int = 5) -> list[str]:
        return C.get(n)


class LegendUtilities:
    @staticmethod
    def get_legend_lines(
        data: Any,
        legend_stats: LegendStats = LegendStats.LAST,
    ) -> list[str]:
        if isinstance(data, pd.Series):
            df = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return [str(data)]

        lines: list[str] = []
        for col in df.columns:
            s = df[col].dropna()
            name = str(col)
            if legend_stats == LegendStats.TOTAL_CHANGE:
                total = ((1 + s).prod() - 1) * 100 if len(s) else 0.0
                lines.append(f"{name}: {total:+.2f}%")
            elif legend_stats == LegendStats.AVG_STD_SKEW_KURT:
                lines.append(
                    f"{name} (\u03bc={s.mean():.4f}, \u03c3={s.std():.4f}, "
                    f"skew={s.skew():.2f}, kurt={s.kurtosis():.2f})"
                )
            elif legend_stats == LegendStats.AVG:
                lines.append(f"{name} (avg={s.mean():.4f})")
            elif legend_stats == LegendStats.LAST:
                last = s.iloc[-1] if len(s) else 0.0
                lines.append(f"{name}: {last:.4f}")
            else:
                lines.append(name)
        return lines

    @staticmethod
    def set_legend(
        ax: plt.Axes,
        labels: list[Any] | None = None,
        legend_loc: str | None = None,
        lines: Any = None,
        colors: Any = None,
        **kwargs,
    ) -> None:
        loc = legend_loc or "best"
        _kw = dict(
            loc=loc,
            framealpha=0.92,
            fontsize=9,
            edgecolor=C.GRID,
            borderpad=0.6,
        )
        if lines is not None and labels is not None:
            ax.legend(lines, [str(label) for label in labels], **_kw)
        elif labels is not None:
            str_labels = [str(label) for label in labels]
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, str_labels, **_kw)
            else:
                ax.legend(str_labels, **_kw)
        else:
            ax.legend(**_kw)

    @staticmethod
    def set_legend_with_rectangles(
        ax: plt.Axes,
        labels: list[str],
        colors: list[str],
        legend_color: str | None = None,
        legend_loc: str | None = None,
        bbox_to_anchor: tuple | None = None,
        framealpha: float | None = None,
        **kwargs,
    ) -> None:
        patches = [
            mpatches.Patch(facecolor=color, label=label)
            for color, label in zip(colors, labels, strict=True)
        ]
        loc = legend_loc or "best"
        kw: dict[str, Any] = {
            "loc": loc,
            "framealpha": framealpha or 0.92,
            "fontsize": 9,
            "edgecolor": C.GRID,
            "borderpad": 0.6,
            "labelspacing": 0.4,
        }
        if bbox_to_anchor is not None:
            kw["bbox_to_anchor"] = bbox_to_anchor
        ax.legend(handles=patches, **kw)


class PlotUtilities:
    @staticmethod
    def create_figure(
        figsize: tuple[int, int] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        return make_figure(figsize)

    @staticmethod
    def plot_line(ax, x_data=None, y_data=None, color=None, label=None, **kwargs):
        kw: dict[str, Any] = {"linewidth": C.LW_MAIN, "antialiased": True}
        if color:
            kw["color"] = color
        if label:
            kw["label"] = label
        kw.update(kwargs)
        if x_data is not None:
            (line,) = ax.plot(x_data, y_data, **kw)
        else:
            (line,) = ax.plot(y_data, **kw)
        return line

    @staticmethod
    def set_title(ax, title="", **kwargs):
        ax.set_title(
            title,
            fontsize=C.FONT_TITLE,
            fontweight="bold",
            color=C.TITLE,
            pad=18,
            loc="left",
        )

    @staticmethod
    def set_ax_xy_labels(ax, xlabel=None, ylabel=None, **kwargs):
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=C.FONT_LABEL, color=C.LABEL, labelpad=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=C.FONT_LABEL, color=C.LABEL, labelpad=10)

    @staticmethod
    def style_axis(ax, **kwargs):
        ensure_style()
        ax.grid(True, alpha=0.85, linestyle="-", linewidth=C.LW_HAIR, color=C.GRID)
        ax.tick_params(labelsize=C.FONT_TICK, colors=C.TICK, length=4, width=0.7, pad=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C.SPINE)
            ax.spines[side].set_linewidth(0.7)

    @staticmethod
    def set_date_on_axis(
        ax, data=None, x_date_freq=None, x_date_format=None, x_date_rotation=None, **kwargs
    ):
        if data is not None:
            smart_date_axis(ax, data)
        else:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    @staticmethod
    def plot_moving_average(ax, data=None, window=30, color=None, **kwargs):
        sma = data.rolling(window=window, min_periods=1).mean()
        kw = {"linestyle": "--", "linewidth": C.LW_THIN, "alpha": 0.7, "label": f"SMA({window})"}
        if color:
            kw["color"] = color
        (line,) = ax.plot(sma.index, sma.values, **kw)
        return line

    @staticmethod
    def add_secondary_y_axis(ax, data=None, color=None, label=None, **kwargs):
        ax2 = ax.twinx()
        kw = {}
        if color:
            kw["color"] = color
        if label:
            kw["label"] = label
        if data is not None:
            ax2.plot(data.index, data.values, **kw)
        return ax2

    @staticmethod
    def add_scatter_points(ax, label_x_y=None, fontsize=9, color=None, marker="o", **kwargs):
        if label_x_y is None:
            return
        c = color or C.NAVY
        for lbl, (x, y) in label_x_y.items():
            ax.scatter(
                x,
                y,
                color=c,
                marker=marker,
                zorder=5,
                s=C.MARKER_LG // 2,
                edgecolors=C.EDGE,
                linewidths=0.8,
            )
            if lbl:
                ax.annotate(
                    lbl,
                    (x, y),
                    fontsize=fontsize,
                    color=c,
                    xytext=(6, 6),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.2", facecolor="white", edgecolor=C.GRID, alpha=0.9
                    ),
                )

    @staticmethod
    def format_y_axis_as_percentage(ax, decimals=0, **kwargs):
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%"))

    @staticmethod
    def set_y_limits(ax, y_limits=None, **kwargs):
        if y_limits:
            ax.set_ylim(y_limits)
