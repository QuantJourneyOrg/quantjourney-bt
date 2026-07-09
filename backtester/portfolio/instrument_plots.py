"""
InstrumentPlots — Institutional-quality instrument-level visualisations.
========================================================================

All charts use the unified QuantJourney style engine from ``plot_compat.py``.
Every public method is ``@staticmethod`` and returns ``plt.Figure``.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from backtester.portfolio.instr_calc import InstrumentCalculations
from backtester.utils.logger import logger
from backtester.plots.plot_compat import (
    C, ensure_style, add_watermark, style_ax, smart_date_axis,
    endpoint_annotation, stats_box, make_figure, diverging_cmap,
)


# ── Helpers ──

def _clean_label(label) -> str:
    """Strip instrument suffix (e.g. 'AAPL-equity' -> 'AAPL')."""
    s = label[0] if isinstance(label, tuple) else str(label)
    return s.split("-")[0]


def _get_instruments(ic: InstrumentCalculations):
    """Get instrument list from InstrumentCalculations."""
    return list(ic.instruments) if hasattr(ic, "instruments") else []


# ═══════════════════════════════════════════════════════════════════════════
# InstrumentPlots class
# ═══════════════════════════════════════════════════════════════════════════

class InstrumentPlots:
    """Static methods producing institutional-quality instrument charts."""

    @staticmethod
    def plot_cumulative_returns(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot the instrument cumulative returns (growth of $1)."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()

        cum = (1 + returns).cumprod()
        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            label = _clean_label(inst)
            ax.plot(cum.index, cum[inst], color=colors[i], linewidth=1.8,
                    label=label, zorder=3)

        # Best/worst annotations
        if len(instruments) > 0:
            final_vals = {inst: cum[inst].iloc[-1] for inst in instruments}
            best_inst = max(final_vals, key=final_vals.get)
            worst_inst = min(final_vals, key=final_vals.get)
            best_idx = instruments.index(best_inst)
            worst_idx = instruments.index(worst_inst)
            endpoint_annotation(ax, cum[best_inst], _clean_label(best_inst),
                                colors[best_idx], fmt="ratio")
            if best_inst != worst_inst:
                endpoint_annotation(ax, cum[worst_inst], _clean_label(worst_inst),
                                    colors[worst_idx], fmt="ratio", offset=(8, -14))

        ax.axhline(1.0, color=C.SPINE, lw=0.7, ls=":")

        # Stats
        total_rets = {_clean_label(inst): (cum[inst].iloc[-1] - 1) * 100
                      for inst in instruments}
        best_name = max(total_rets, key=total_rets.get)
        subtitle = f"Best: {best_name} ({total_rets[best_name]:+.1f}%)  |  {len(instruments)} instruments"

        style_ax(ax, title="Cumulative Returns", ylabel="Growth of $1",
                 subtitle=subtitle)
        smart_date_axis(ax, cum)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_return_distribution(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot return distribution histograms for each instrument."""
        ensure_style()
        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()

        instruments = _get_instruments(instrument_calc)
        n = len(instruments)
        num_cols = min(3, n)
        num_rows = math.ceil(n / num_cols) if num_cols > 0 else 1

        fig, axs = plt.subplots(num_rows, num_cols,
                                figsize=(5 * num_cols, 4.5 * num_rows))
        if n == 1:
            axs = np.array([axs])
        axs = np.atleast_1d(axs).flatten()

        for i, inst in enumerate(instruments):
            ax = axs[i]
            data = returns[inst].dropna() * 100
            ax.hist(data, bins=50, color=C.BLUE, alpha=0.70,
                    edgecolor="white", linewidth=0.4)
            ax.axvline(data.mean(), color=C.BENCHMARK, ls=C.BENCHMARK_LS, lw=1.4,
                       label=f"Mean: {data.mean():.3f}%")
            ax.axvline(0, color=C.SPINE, lw=0.7)
            style_ax(ax, title=f"{_clean_label(inst)} Returns")
            ax.set_xlabel("Return (%)", fontsize=9, color=C.LABEL)
            ax.set_ylabel("Frequency", fontsize=9, color=C.LABEL)
            ax.legend(fontsize=8, framealpha=0.95, edgecolor=C.SPINE)

        # Hide unused subplots
        for j in range(n, len(axs)):
            fig.delaxes(axs[j])

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_rolling_volatility(
        instrument_calc: InstrumentCalculations, window=126, **kwargs
    ) -> plt.Figure:
        """Plot the instrument rolling volatility."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        volatility = instrument_calc.compute_rolling_volatility(periods=window)
        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(volatility.index, volatility[inst], color=colors[i],
                    linewidth=1.6, label=_clean_label(inst))

        style_ax(ax, title=f"Rolling Volatility ({window}-day window)",
                 ylabel="Volatility",
                 subtitle=f"{len(instruments)} instruments")
        smart_date_axis(ax, volatility)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    # ── Strategy analysis ──
