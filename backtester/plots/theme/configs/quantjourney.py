"""
    QuantJourney Framework - QuantJourney Theme Configuration
        ------------------------------------------------------------

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from backtester.plots.theme.types import (
    ColorScheme,
    LegendConfig,
    PlotLabelConfig,
    PlotLineConfig,
    ThemeConfig,
)

QUANTJOURNEY_THEME = ThemeConfig(
    colors=ColorScheme(
        # Primary Colors
        PRIMARY="#123047",
        SECONDARY="#F2A900",
        TERTIARY="#D63F31",
        # Blues
        DARK_BLUE="#123047",
        MID_BLUE="#1F77B4",
        LIGHT_BLUE="#2563EB",
        SOFT_BLUE="#3FA7D6",
        # Background
        BACKGROUND="#FBFCFE",
        BACKGROUND2="#FBFCFE",
        BACKGROUND3="#F4F7FA",
        # Text
        TITLE="#101828",
        TEXT="#344054",
        AXIS="#344054",
        LABELS="#344054",
        # Grid and Spines
        GRID="#E8EDF3",
        SPINES="#D7DEE7",
        # Fill Colors
        FILL="#E7F0FF",
        FILL_UP="#27AE60",
        FILL_DOWN="#E74C3C",
        # Special Purpose
        POSITIVE="#27AE60",
        NEGATIVE="#C0392B",
        NEUTRAL="#667085",
        WARNING="#F39C12",
        HIGHLIGHT="#16A085",
        # Extended (matching institutional C defaults)
        GOLD="#F2A900",
        AMBER="#FFB000",
        TEAL="#00A6A6",
        ORANGE="#F97316",
        PURPLE="#7B61FF",
        MUTED="#8B97A8",
        WATERMARK="#BAC3CF",
        SUBTITLE="#4A5568",
        TICK="#5A6577",
        HM_NEG="#C0392B",
        HM_ZERO="#FAFBFC",
        HM_POS="#2563EB",
        DD_LIGHT="#D7E8F4",
        DD_MED="#8DB9D8",
        DD_HEAVY="#3D7197",
        DD_SEVERE="#123047",
        # Data Visualization Colors
        VIZ_SEQUENTIAL=(
            "#f7fbff",
            "#deebf7",
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#4292c6",
            "#2171b5",
            "#084594",
        ),
        VIZ_DIVERGING=(
            "#d73027",
            "#f46d43",
            "#fdae61",
            "#fee090",
            "#e0f3f8",
            "#abd9e9",
            "#74add1",
            "#4575b4",
        ),
        VIZ_CATEGORICAL=(
            "#2563EB",
            "#F2A900",
            "#00A6A6",
            "#E11D48",
            "#10B981",
            "#0EA5E9",
            "#F97316",
            "#0F2F4A",
            "#4A9BD9",
            "#27864B",
        ),
    ),
    plot_lines=PlotLineConfig(
        MEAN={"color": "#002D62", "linestyle": "-", "linewidth": 1.2},
        MEAN_WITH_SHADOWS={"color": "#002D62", "shadow_alpha": 0.3},
        AVERAGE={"color": "#005B96", "linestyle": "-", "linewidth": 1.2},
        AVERAGE_WITH_SHADOWS={"color": "#005B96", "shadow_alpha": 0.3},
        ZERO={"color": "#C62828", "linestyle": ":", "linewidth": 1.0},
        ZERO_WITH_SHADOWS={"color": "#C62828", "shadow_alpha": 0.2},
        TREND_LINE={"color": "#E67E22", "linestyle": "-", "linewidth": 1.5},
        TREND_LINE_WITH_SHADOWS={"color": "#E67E22", "shadow_alpha": 0.3},
        ABOVE_ZERO={"color": "#2E7D32", "fill_alpha": 0.15},
        ABOVE_ZERO_WITH_SHADOWS={"color": "#2E7D32", "shadow_alpha": 0.2},
        REGRESSION={"color": "#6497B1", "linestyle": "--", "linewidth": 1.3},
        REGRESSION_WITH_SHADOW={"color": "#6497B1", "shadow_alpha": 0.25},
    ),
    plot_labels=PlotLabelConfig(
        NONE={},
        LAST_VALUE={"fontsize": 11, "ha": "right", "va": "center", "color": "#002D62"},
        LAST_VALUE_SORTED={"fontsize": 11, "ha": "left", "va": "center", "color": "#002D62"},
        AVERAGE_VALUE={"fontsize": 11, "ha": "center", "va": "bottom", "color": "#005B96"},
        AVERAGE_VALUE_SORTED={"fontsize": 11, "ha": "left", "va": "center", "color": "#005B96"},
        MAX_VALUE={"fontsize": 11, "ha": "center", "va": "bottom", "color": "#2E7D32"},
        MAX_VALUE_SORTED={"fontsize": 11, "ha": "left", "va": "center", "color": "#2E7D32"},
        MIN_VALUE={"fontsize": 11, "ha": "center", "va": "top", "color": "#C62828"},
        MIN_VALUE_SORTED={"fontsize": 11, "ha": "left", "va": "center", "color": "#C62828"},
        PERCENTILE={"fontsize": 10, "ha": "right", "va": "center", "color": "#2C3E50"},
        PERCENTILE_SORTED={"fontsize": 10, "ha": "left", "va": "center", "color": "#2C3E50"},
        PERCENTILE_LABEL={"fontsize": 10, "ha": "right", "va": "center", "color": "#2C3E50"},
        PERCENTILE_LABEL_SORTED={"fontsize": 10, "ha": "left", "va": "center", "color": "#2C3E50"},
    ),
    legend=LegendConfig(
        NONE={},
        LAST={"location": "best", "fontsize": 9, "framealpha": 0.92},
        AVG={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
        AVG_LAST={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
        AVG_STD={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
        AVG_STD_SKEW_KURT={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
        AVG_STD_LAST={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
        AVG_NONNAN_LAST={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
        MEDIAN_NONNAN_LAST={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
        AVG_MEDIAN_STD_NONNAN_LAST={"location": "upper right", "fontsize": 8, "framealpha": 0.92},
        TOTAL={"location": "best", "fontsize": 9, "framealpha": 0.92},
        PERCENTILES={"location": "upper right", "fontsize": 9, "framealpha": 0.92},
    ),
    # Figure settings
    FIGSIZE=(10.5, 5.8),
    DPI=150,
    # Line settings
    LINEWIDTH=1.15,
    LINESTYLE="-",
    MARKER=None,
    MARKERSIZE=5,
    ALPHA=1.0,
    # Text settings
    FONTSIZE=9.5,
    TITLE_FONTSIZE=12.5,
    LABEL_FONTSIZE=9.5,
    TICK_FONTSIZE=8.5,
    LEGEND_FONTSIZE=8.5,
    FONTWEIGHT="normal",
    FONTFAMILY="sans-serif",
    # Grid settings
    GRID_ALPHA=0.55,
    GRID_LINESTYLE="-",
    GRID_LINEWIDTH=0.35,
    # Legend settings
    LEGEND_LOC="best",
    LEGEND_FRAMEALPHA=0.92,
    LEGEND_EDGECOLOR="#E4E7EC",
    BBOX_TO_ANCHOR=None,
    # Date axis settings
    DATE_FORMAT="%Y",
    DATE_FREQ="auto",
    DATE_ROTATION=0,
    # Spine settings
    SPINE_LINEWIDTH=0.5,
    SPINE_VISIBLE=True,
    # Padding/Margins
    TITLE_PAD=12,
    LABEL_PAD=7,
    TICK_PAD=5,
    # ── Line width hierarchy ──
    LW_MAIN=1.55,
    LW_SECONDARY=1.15,
    LW_THIN=0.85,
    LW_HAIR=0.45,
    LW_EDGE=0.35,
    # ── Fill opacity hierarchy ──
    FILL_MAIN=0.055,
    FILL_LIGHT=0.08,
    FILL_HEAVY=0.78,
    FILL_HIST=0.50,
    # ── Benchmark styling ──
    BENCHMARK_COLOR="#C4932F",
    BENCHMARK_LS="--",
    # ── Edge color ──
    EDGE_COLOR="white",
    # ── Markers ──
    MARKER_SM=4,
    MARKER_LG=36,
    # ── Annotation fonts ──
    FONT_ANNOT=7.8,
    FONT_SMALL=6.5,
)
