"""
    Theme Manager
    ------------------------------------------------------------


    Manages plot themes and provides color utilities.

    Changes vs original:
    - Instance-based design (thread-safe, no global mutable state)
    - Reduced boilerplate — access sub-configs via properties
    - Palette selection via Enum instead of raw strings
    - Color generation extends base palette instead of discarding it
    - Hex colour validation helper
    - Module-level ``default_theme`` for convenience

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

import colorsys
from enum import Enum
from typing import List, Optional

from backtester.plots.theme.types import (
    ColorScheme,
    LegendConfig,
    PlotLabelConfig,
    PlotLineConfig,
    PlotTheme,
    ThemeConfig,
)
from backtester.plots.theme.configs import THEME_CONFIGS


__all__ = [
    "PaletteType",
    "ThemeManager",
    "default_theme",
]


# ---------------------------------------------------------------------------
# Palette enum (replaces raw strings)
# ---------------------------------------------------------------------------

class PaletteType(Enum):
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"


_PALETTE_ACCESSOR = {
    PaletteType.CATEGORICAL: lambda cs: cs.VIZ_CATEGORICAL,
    PaletteType.SEQUENTIAL: lambda cs: cs.VIZ_SEQUENTIAL,
    PaletteType.DIVERGING: lambda cs: cs.VIZ_DIVERGING,
}


# ---------------------------------------------------------------------------
# Hex colour helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert ``#RRGGBB`` to an (r, g, b) tuple with values in [0, 1]."""
    h = hex_color.lstrip("#")
    if len(h) != 6 or not all(c in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"Invalid hex colour: {hex_color!r}")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert (r, g, b) floats in [0, 1] to ``#rrggbb``."""
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _generate_colors_from_base(base_color: str, n: int) -> List[str]:
    """
    Generate *n* colours by rotating the hue of *base_color*.

    The base colour itself is included as the first entry.
    """
    h, s, v = colorsys.rgb_to_hsv(*_hex_to_rgb(base_color))
    return [
        _rgb_to_hex(*colorsys.hsv_to_rgb((h + i / n) % 1.0, s, v))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# ThemeManager
# ---------------------------------------------------------------------------

class ThemeManager:
    """
    Manages plot themes and provides colour utilities.

    Prefer creating an instance (or using the module-level ``default_theme``)
    rather than relying on class-level state.  This avoids race conditions
    when multiple threads or notebooks set different themes concurrently.
    """

    # Class-level fallback so legacy callers like
    # ``ThemeManager.get_current_theme()`` keep working.
    _default_theme: "ThemeManager | None" = None

    def __init__(self, theme: PlotTheme = PlotTheme.QUANTJOURNEY) -> None:
        if theme not in THEME_CONFIGS:
            raise ValueError(
                f"Unknown theme: {theme}. "
                f"Available: {list(THEME_CONFIGS.keys())}"
            )
        self._theme = theme
        # Track the most-recently created instance as class-level default
        ThemeManager._default_theme = self

    # -- Class-level convenience (backward-compat) --------------------------

    @classmethod
    def get_current_theme(cls) -> "ThemeConfig":
        """Return the current ThemeConfig (from the default instance)."""
        if cls._default_theme is None:
            cls._default_theme = cls()  # QuantJourney theme
        return cls._default_theme.config

    @classmethod
    def set_theme(cls, theme) -> None:
        """Set the global theme by enum or string name."""
        if isinstance(theme, str):
            theme = PlotTheme(theme)
        cls._default_theme = cls(theme)

    # -- Theme access -------------------------------------------------------

    @property
    def theme(self) -> PlotTheme:
        return self._theme

    @theme.setter
    def theme(self, value: PlotTheme) -> None:
        if value not in THEME_CONFIGS:
            raise ValueError(
                f"Unknown theme: {value}. "
                f"Available: {list(THEME_CONFIGS.keys())}"
            )
        self._theme = value

    @property
    def config(self) -> ThemeConfig:
        """Full theme configuration object."""
        return THEME_CONFIGS[self._theme]

    @property
    def colors(self) -> ColorScheme:
        return self.config.colors

    @property
    def plot_lines(self) -> PlotLineConfig:
        return self.config.plot_lines

    @property
    def plot_labels(self) -> PlotLabelConfig:
        return self.config.plot_labels

    @property
    def legend(self) -> LegendConfig:
        return self.config.legend

    # -- Colour generation --------------------------------------------------

    def get_n_colors(
        self,
        n: int,
        palette: PaletteType = PaletteType.CATEGORICAL,
    ) -> List[str]:
        """
        Return *n* colours from the requested palette.

        If *n* exceeds the base palette length, the base colours are kept and
        additional colours are generated by hue-rotating the last base colour.
        """
        if isinstance(palette, str):
            # Graceful fallback for callers passing raw strings
            palette = PaletteType(palette)

        accessor = _PALETTE_ACCESSOR.get(palette)
        if accessor is None:
            raise ValueError(
                f"Unknown palette: {palette}. "
                f"Options: {[p.value for p in PaletteType]}"
            )

        base_colors = list(accessor(self.colors))

        if n <= len(base_colors):
            return base_colors[:n]

        # Extend with generated colours instead of discarding the base palette
        extra_needed = n - len(base_colors)
        extra = _generate_colors_from_base(base_colors[-1], extra_needed + 1)[1:]
        return base_colors + extra

    # -- Dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<ThemeManager theme={self._theme.name}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThemeManager):
            return NotImplemented
        return self._theme == other._theme


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

default_theme = ThemeManager()