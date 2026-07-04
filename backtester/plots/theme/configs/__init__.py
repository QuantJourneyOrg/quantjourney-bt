# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.


from backtester.plots.theme.types import PlotTheme
from backtester.plots.theme.configs.quantjourney import QUANTJOURNEY_THEME

THEME_CONFIGS = {
    PlotTheme.QUANTJOURNEY: QUANTJOURNEY_THEME,
}

__all__ = [
    'THEME_CONFIGS',
    'QUANTJOURNEY_THEME',
]
