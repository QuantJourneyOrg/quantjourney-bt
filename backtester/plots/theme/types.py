"""
        Types for theme configuration in QuantJourney plots
        ------------------------------------------------------------

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


# Plot Themes ------------------------------------------------------------
class PlotTheme(Enum):
    QUANTJOURNEY = "quantjourney"


# Plot Configurations ------------------------------------------------------------
@dataclass
class PlotLineConfig:
    MEAN: dict[str, Any]
    MEAN_WITH_SHADOWS: dict[str, Any]
    AVERAGE: dict[str, Any]
    AVERAGE_WITH_SHADOWS: dict[str, Any]
    ZERO: dict[str, Any]
    ZERO_WITH_SHADOWS: dict[str, Any]
    TREND_LINE: dict[str, Any]
    TREND_LINE_WITH_SHADOWS: dict[str, Any]
    ABOVE_ZERO: dict[str, Any]
    ABOVE_ZERO_WITH_SHADOWS: dict[str, Any]
    REGRESSION: dict[str, Any]
    REGRESSION_WITH_SHADOW: dict[str, Any]


# Plot Label Configurations ------------------------------------------------------------
@dataclass
class PlotLabelConfig:
    NONE: dict[str, Any]
    LAST_VALUE: dict[str, Any]
    LAST_VALUE_SORTED: dict[str, Any]
    AVERAGE_VALUE: dict[str, Any]
    AVERAGE_VALUE_SORTED: dict[str, Any]
    MAX_VALUE: dict[str, Any]
    MAX_VALUE_SORTED: dict[str, Any]
    MIN_VALUE: dict[str, Any]
    MIN_VALUE_SORTED: dict[str, Any]
    PERCENTILE: dict[str, Any]
    PERCENTILE_SORTED: dict[str, Any]
    PERCENTILE_LABEL: dict[str, Any]
    PERCENTILE_LABEL_SORTED: dict[str, Any]


# Legend Configurations ------------------------------------------------------------
@dataclass
class LegendConfig:
    NONE: dict[str, Any]
    LAST: dict[str, Any]
    AVG: dict[str, Any]
    AVG_LAST: dict[str, Any]
    AVG_STD: dict[str, Any]
    AVG_STD_SKEW_KURT: dict[str, Any]
    AVG_STD_LAST: dict[str, Any]
    AVG_NONNAN_LAST: dict[str, Any]
    MEDIAN_NONNAN_LAST: dict[str, Any]
    AVG_MEDIAN_STD_NONNAN_LAST: dict[str, Any]
    TOTAL: dict[str, Any]
    PERCENTILES: dict[str, Any]


# Theme Configuration ------------------------------------------------------------
@dataclass
class ColorScheme:
    PRIMARY: str
    SECONDARY: str
    TERTIARY: str
    DARK_BLUE: str
    MID_BLUE: str
    LIGHT_BLUE: str
    SOFT_BLUE: str
    BACKGROUND: str
    BACKGROUND2: str
    BACKGROUND3: str
    TITLE: str
    TEXT: str
    AXIS: str
    LABELS: str
    GRID: str
    SPINES: str
    FILL: str
    FILL_UP: str
    FILL_DOWN: str
    POSITIVE: str
    NEGATIVE: str
    NEUTRAL: str
    WARNING: str
    HIGHLIGHT: str
    VIZ_SEQUENTIAL: tuple[str, ...]
    VIZ_DIVERGING: tuple[str, ...]
    VIZ_CATEGORICAL: tuple[str, ...]

    # ── Extended fields for full C.* coverage (defaults = institutional blue) ──
    GOLD: str = "#C49A3C"
    AMBER: str = "#E8A838"
    TEAL: str = "#177E89"
    ORANGE: str = "#D4842A"
    PURPLE: str = "#6C3483"
    MUTED: str = "#8896A7"
    WATERMARK: str = "#B0B8C4"
    SUBTITLE: str = "#4A5568"
    TICK: str = "#5A6577"
    HM_NEG: str = "#C0392B"
    HM_ZERO: str = "#FAFBFC"
    HM_POS: str = "#0A1F38"
    UP: str = ""
    DOWN: str = ""
    DD_LIGHT: str = "#A7D2F0"
    DD_MED: str = "#4A9BD9"
    DD_HEAVY: str = "#1A5276"
    DD_SEVERE: str = "#0A1F38"

    def __post_init__(self):
        if not self.UP:
            self.UP = self.POSITIVE
        if not self.DOWN:
            self.DOWN = self.NEGATIVE


# Theme Configuration ------------------------------------------------------------
@dataclass
class ThemeConfig:
    """
    Complete theme configuration including colors and plot parameters for QuantJourney plots
    """

    colors: ColorScheme
    plot_lines: PlotLineConfig
    plot_labels: PlotLabelConfig
    legend: LegendConfig

    # Figure settings
    FIGSIZE: tuple[int, int] = (10, 6)
    DPI: int = 100

    # Line settings
    LINEWIDTH: float = 1.0
    LINESTYLE: str = "--"
    MARKER: str | None = None
    MARKERSIZE: int | None = 6
    ALPHA: float = 1.0

    # Text settings
    FONTSIZE: int = 12
    TITLE_FONTSIZE: int = 14
    LABEL_FONTSIZE: int = 12
    TICK_FONTSIZE: int = 10
    LEGEND_FONTSIZE: int = 10
    FONTWEIGHT: str = "normal"
    FONTFAMILY: str = "sans-serif"

    # Grid settings
    GRID_ALPHA: float = 0.15
    GRID_LINESTYLE: str = "--"
    GRID_LINEWIDTH: float = 0.5

    # Legend settings
    LEGEND_LOC: str = "best"
    LEGEND_FRAMEALPHA: float = 0.8
    LEGEND_EDGECOLOR: str = "none"
    BBOX_TO_ANCHOR: tuple[float, float] | None = None

    # Date axis settings
    DATE_FORMAT: str = "%Y-%m-%d"
    DATE_FREQ: str = "auto"
    DATE_ROTATION: int = 45

    # Spine settings
    SPINE_LINEWIDTH: float = 0.8
    SPINE_VISIBLE: bool = True

    # Padding/Margins
    TITLE_PAD: int = 10
    LABEL_PAD: int = 5
    TICK_PAD: int = 3

    # ── Line width hierarchy ──
    LW_MAIN: float = 2.0  # Primary data lines (portfolio, rolling metrics)
    LW_SECONDARY: float = 1.6  # Benchmark, MA, secondary lines
    LW_THIN: float = 1.0  # Reference/average lines
    LW_HAIR: float = 0.5  # Zone lines, subtle references
    LW_EDGE: float = 0.5  # Bar/histogram edges

    # ── Fill opacity hierarchy ──
    FILL_MAIN: float = 0.28  # Gradient under main line
    FILL_LIGHT: float = 0.18  # Rolling metric positive/negative fills
    FILL_HEAVY: float = 0.80  # Stacked area charts, bar charts
    FILL_HIST: float = 0.65  # Histograms

    # ── Benchmark styling ──
    BENCHMARK_COLOR: str = ""  # Empty = use colors.GOLD
    BENCHMARK_LS: str = "--"

    # ── Edge color for bars/histograms ──
    EDGE_COLOR: str = "white"  # "white" for light themes, dark bg for dark

    # ── Marker sizes ──
    MARKER_SM: int = 6  # Standard markers
    MARKER_LG: int = 50  # Large scatter markers

    # ── Annotation font sizes ──
    FONT_ANNOT: int = 9  # Annotation/stats text
    FONT_SMALL: int = 7  # Zone labels, small references
