# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Focused allocation-chart layout regressions."""

import pytest


@pytest.mark.parametrize(
    "plot_name", ["plot_portfolio_weights", "plot_percentage_weights", "plot_rolling_weights"]
)
@pytest.mark.parametrize("symbol_count", [5, 50])
def test_allocation_legend_fits_at_300_dpi_without_dropping_labels(plot_name, symbol_count) -> None:
    import warnings
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from backtester.portfolio import PortfolioPlots

    dates = pd.date_range("2020-01-01", periods=4, tz="UTC")
    symbols = [f"S{i:03}" for i in range(symbol_count)]
    weights = pd.DataFrame(
        np.full((len(dates), symbol_count), 1 / symbol_count), index=dates, columns=symbols
    )
    calculator = SimpleNamespace(portfolio_data=SimpleNamespace(weights=weights))
    figure = None
    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            figure = getattr(PortfolioPlots, plot_name)(calculator)
            figure.set_dpi(300)
            figure.canvas.draw()
        legend = figure.axes[0].get_legend()
        expected = symbols + (["Cash"] if plot_name == "plot_percentage_weights" else [])
        assert [text.get_text() for text in legend.get_texts()] == expected
        bounds = legend.get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0 >= figure.bbox.x0 - 2
        assert bounds.y0 >= figure.bbox.y0 - 2
        assert bounds.x1 <= figure.bbox.x1 + 2
        assert bounds.y1 <= figure.bbox.y1 + 2
        assert not [
            warning for warning in emitted if "Tight layout not applied" in str(warning.message)
        ]
    finally:
        if figure is not None:
            plt.close(figure)
