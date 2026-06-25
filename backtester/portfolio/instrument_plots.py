"""
InstrumentPlots — Institutional-quality instrument-level visualisations.
========================================================================

All charts use the unified QuantJourney style engine from ``plot_compat.py``.
Every public method is ``@staticmethod`` and returns ``plt.Figure``.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
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
import seaborn as sns
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
    def plot_returns(instrument_calc: InstrumentCalculations, **kwargs) -> plt.Figure:
        """Plot the instrument returns as time series."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))
        for i, inst in enumerate(instruments):
            ax.plot(returns.index, returns[inst], color=colors[i],
                    linewidth=1.2, alpha=0.80, label=_clean_label(inst))

        ax.axhline(0, color=C.SPINE, lw=0.7)

        style_ax(ax, title="Instrument Returns", ylabel="Daily Return")
        smart_date_axis(ax, returns)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

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
    def plot_pnl(instrument_calc: InstrumentCalculations, **kwargs) -> plt.Figure:
        """Plot the instrument P&L over time."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        pnl = instrument_calc.compute_pnl()
        if isinstance(pnl, pd.Series):
            pnl = pnl.to_frame()

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(pnl.index, pnl[inst], color=colors[i], linewidth=1.6,
                    label=_clean_label(inst))

        ax.axhline(0, color=C.SPINE, lw=0.7)

        style_ax(ax, title="Instrument Profit and Loss", ylabel="P&L")
        smart_date_axis(ax, pnl)
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

    @staticmethod
    def plot_rolling_sharpe_ratio(
        instrument_calc: InstrumentCalculations,
        risk_free_rate=0.0,
        window=252,
        **kwargs,
    ) -> plt.Figure:
        """Plot the instrument rolling Sharpe ratio."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        rolling_sharpe = instrument_calc.compute_rolling_sharpe_ratio(
            risk_free_rate=risk_free_rate, window=window
        )
        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(rolling_sharpe.index, rolling_sharpe[inst], color=colors[i],
                    linewidth=1.6, label=_clean_label(inst))

        ax.axhline(0, color=C.SPINE, lw=0.7)
        ax.axhline(1.0, color=C.MUTED, lw=0.6, ls=":")

        style_ax(ax, title=f"Rolling Sharpe Ratio ({window}-day window)",
                 ylabel="Sharpe Ratio")
        smart_date_axis(ax, rolling_sharpe)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_rolling_beta(
        instrument_calc: InstrumentCalculations, benchmark_returns, window=252, **kwargs
    ) -> plt.Figure:
        """Plot the instrument rolling beta."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        rolling_beta = instrument_calc.compute_rolling_beta(
            benchmark_returns=benchmark_returns, window=window
        )
        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(rolling_beta.index, rolling_beta[inst], color=colors[i],
                    linewidth=1.6, label=_clean_label(inst))

        for lev in [0.5, 1.0, 1.5]:
            ax.axhline(lev, color=C.SPINE, ls=":", lw=0.6, alpha=0.5)

        style_ax(ax, title=f"Rolling Beta ({window}-day window)",
                 ylabel="Beta")
        smart_date_axis(ax, rolling_beta)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_rolling_alpha(
        instrument_calc: InstrumentCalculations,
        benchmark_returns,
        risk_free_rate=0.0,
        window=252,
        **kwargs,
    ) -> plt.Figure:
        """Plot the instrument rolling alpha."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        rolling_alpha = instrument_calc.compute_rolling_alpha(
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            window=window,
        )
        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(rolling_alpha.index, rolling_alpha[inst], color=colors[i],
                    linewidth=1.6, label=_clean_label(inst))

        ax.axhline(0, color=C.SPINE, lw=0.7)

        style_ax(ax, title=f"Rolling Alpha ({window}-day window)",
                 ylabel="Alpha (Annualised)")
        smart_date_axis(ax, rolling_alpha)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_rolling_max_drawdown(
        instrument_calc: InstrumentCalculations, window=252, **kwargs
    ) -> plt.Figure:
        """Plot the instrument rolling maximum drawdown."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        rolling_mdd = instrument_calc.compute_rolling_max_drawdown(periods=window)
        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            series = rolling_mdd[inst] * 100  # percentage
            ax.plot(series.index, series, color=colors[i],
                    linewidth=1.6, label=_clean_label(inst))

        ax.axhline(0, color=C.SPINE, lw=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.0f}%"))

        style_ax(ax, title=f"Rolling Maximum Drawdown ({window}-day window)",
                 ylabel="Max Drawdown (%)")
        smart_date_axis(ax, rolling_mdd)
        ax.legend(loc="lower left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_return_quantiles(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot return quantiles (box plots) for each instrument."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = instrument_calc.returns
        instruments = _get_instruments(instrument_calc)
        data = [(returns[inst].dropna() * 100).values for inst in instruments]
        labels = [_clean_label(inst) for inst in instruments]

        bp = ax.boxplot(
            data, tick_labels=labels, patch_artist=True, notch=True,
            medianprops=dict(color=C.TITLE, lw=1.4),
            whiskerprops=dict(color=C.NAVY, lw=0.9),
            capprops=dict(color=C.NAVY, lw=0.9),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=C.MUTED,
                            markeredgecolor=C.MUTED, alpha=0.5),
        )

        colors = C.get(len(instruments))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor(C.NAVY)
            patch.set_linewidth(1.0)

        ax.axhline(0, color=C.SPINE, lw=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.1f}%"))

        style_ax(ax, title="Return Quantiles by Instrument",
                 ylabel="Return (%)")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_rolling_correlation(
        instrument_calc: InstrumentCalculations, window=252, **kwargs
    ) -> plt.Figure:
        """Plot rolling correlation heatmap (last N days)."""
        ensure_style()
        rets = instrument_calc.returns
        if isinstance(rets, pd.Series):
            rets = rets.to_frame()

        w = min(window, len(rets))
        corr_last = rets.iloc[-w:].corr()
        clean_cols = [_clean_label(c) for c in corr_last.columns]
        corr_last.index = clean_cols
        corr_last.columns = clean_cols

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = diverging_cmap()
        mask = np.triu(np.ones_like(corr_last, dtype=bool), k=1)

        sns.heatmap(
            corr_last, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            vmin=-1, vmax=1, center=0, ax=ax, square=True,
            annot_kws={"size": 10, "fontweight": "bold"},
            cbar_kws={"label": "Correlation", "shrink": 0.8},
            linewidths=1.0, linecolor=C.FIG_BG,
        )

        style_ax(ax, title=f"Rolling Correlation (last {w} days)")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_correlation_heatmap(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot the full-sample correlation heatmap (lower triangle)."""
        ensure_style()

        corr = instrument_calc.compute_correlation_matrix()
        clean_labels = [_clean_label(c) for c in corr.columns]
        corr.index = clean_labels
        corr.columns = clean_labels

        fig, ax = plt.subplots(figsize=(11, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cmap = diverging_cmap()

        sns.heatmap(
            corr, mask=mask, annot=True,
            fmt=kwargs.get("annot_fmt", ".2f"), cmap=cmap,
            vmin=-1, vmax=1, center=0, ax=ax, square=True,
            annot_kws={"size": 10, "fontweight": "bold"},
            cbar_kws={"label": "Correlation", "shrink": 0.8},
            linewidths=1.0, linecolor=C.FIG_BG,
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        # Summary stats
        no_diag = corr.values[~np.eye(len(corr), dtype=bool)]
        if len(no_diag) > 0:
            avg_c = np.nanmean(no_diag)
            stats_text = (
                f"Avg Corr: {avg_c:.2f}\n"
                f"Max Corr: {np.nanmax(no_diag):.2f}\n"
                f"Min Corr: {np.nanmin(no_diag):.2f}"
            )
            stats_box(ax, stats_text, loc="lower left")

        style_ax(ax, title="Instrument Correlation Matrix")

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_cumulative_pnl_comparison(
        instrument_calc: InstrumentCalculations, benchmark_pnl, **kwargs
    ) -> plt.Figure:
        """Plot cumulative P&L comparison with benchmark."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        pnl = instrument_calc.daily_pnl
        if isinstance(pnl, pd.Series):
            pnl = pnl.to_frame()

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(pnl.index, pnl[inst].cumsum(), color=colors[i],
                    linewidth=1.6, label=_clean_label(inst))

        # Benchmark
        ax.plot(benchmark_pnl.index, benchmark_pnl.cumsum(), color=C.GOLD,
                linewidth=2.0, linestyle="--", label="Benchmark", zorder=5)

        ax.axhline(0, color=C.SPINE, lw=0.7)

        style_ax(ax, title="Cumulative P&L Comparison", ylabel="Cumulative P&L")
        smart_date_axis(ax, pnl)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(4, len(instruments) + 1))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_annual_returns(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot annual returns as grouped bar chart."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
        returns.index = pd.to_datetime(returns.index)

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        annual_list = []
        for inst in instruments:
            annual = returns[inst].resample("YE").apply(
                lambda x: (1 + x).prod() - 1) * 100
            annual.name = _clean_label(inst)
            annual_list.append(annual)

        annual_df = pd.concat(annual_list, axis=1)
        annual_df.index = annual_df.index.year

        annual_df.plot(kind="bar", ax=ax, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.8, width=0.8)

        ax.axhline(0, color=C.SPINE, lw=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0f}%"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        style_ax(ax, title="Annual Returns", ylabel="Return (%)",
                 subtitle=f"{len(instruments)} instruments")
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_drawdown_comparison(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot drawdown comparison between instruments."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            cum = (1 + instrument_calc.returns[inst]).cumprod()
            dd = (cum / cum.cummax() - 1) * 100
            ax.plot(dd.index, dd, color=colors[i], linewidth=1.4,
                    label=_clean_label(inst), alpha=0.85)

        ax.axhline(0, color=C.SPINE, lw=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0f}%"))

        style_ax(ax, title="Drawdown Comparison", ylabel="Drawdown (%)")
        smart_date_axis(ax, instrument_calc.returns)
        ax.legend(loc="lower left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_returns_distribution(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot returns distributions in subplots grid."""
        ensure_style()
        returns = instrument_calc.returns
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()

        instruments = _get_instruments(instrument_calc)
        n = len(instruments)
        num_cols = min(3, n)
        num_rows = math.ceil(n / num_cols) if num_cols > 0 else 1

        fig, axs = plt.subplots(num_rows, num_cols,
                                figsize=(5.5 * num_cols, 4.5 * num_rows))
        if n == 1:
            axs = np.array([axs])
        axs = np.atleast_1d(axs).flatten()

        for i, inst in enumerate(instruments):
            ax = axs[i]
            data = returns[inst].dropna() * 100
            ax.hist(data, bins=50, color=C.BLUE, alpha=0.70,
                    edgecolor="white", linewidth=0.4)
            mu = data.mean()
            ax.axvline(mu, color=C.BENCHMARK, ls=C.BENCHMARK_LS, lw=1.4)
            ax.axvline(0, color=C.SPINE, lw=0.7)

            txt = f"Mean: {mu:.3f}%\nStd: {data.std():.3f}%"
            stats_box(ax, txt, loc="upper right")

            style_ax(ax, title=_clean_label(inst))
            ax.set_xlabel("Return (%)", fontsize=9, color=C.LABEL)
            ax.set_ylabel("Frequency", fontsize=9, color=C.LABEL)

        for j in range(n, len(axs)):
            fig.delaxes(axs[j])

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_prices(instrument_calc: InstrumentCalculations, **kwargs) -> plt.Figure:
        """Plot instrument prices over time."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        prices = instrument_calc.prices
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(prices.index, prices[inst], color=colors[i],
                    linewidth=1.8, label=_clean_label(inst))

        style_ax(ax, title="Instrument Prices", ylabel="Price")
        smart_date_axis(ax, prices)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_volatility(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot instrument volatility over time."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        volatility = instrument_calc.volatility
        if isinstance(volatility, pd.Series):
            volatility = volatility.to_frame()

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(volatility.index, volatility[inst], color=colors[i],
                    linewidth=1.6, label=_clean_label(inst))

        style_ax(ax, title="Instrument Volatility", ylabel="Volatility")
        smart_date_axis(ax, volatility)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_volume(instrument_calc: InstrumentCalculations, **kwargs) -> plt.Figure:
        """Plot instrument trading volume over time."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        volume = instrument_calc.instrument_data.volume
        if isinstance(volume, pd.Series):
            volume = volume.to_frame()

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            if inst in volume.columns:
                ax.plot(volume.index, volume[inst], color=colors[i],
                        linewidth=1.2, alpha=0.80, label=_clean_label(inst))

        style_ax(ax, title="Trading Volume", ylabel="Volume")
        smart_date_axis(ax, volume)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_exposures(instrument_calc: InstrumentCalculations, **kwargs) -> plt.Figure:
        """Plot instrument exposures as stacked area."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        exposures = instrument_calc.compute_exposures()
        if isinstance(exposures, pd.Series):
            exposures = exposures.to_frame()

        labels = [_clean_label(c) for c in exposures.columns]
        colors = C.get(len(labels))

        ax.stackplot(exposures.index,
                     *[exposures.iloc[:, i] for i in range(exposures.shape[1])],
                     labels=labels, colors=colors, alpha=0.80, linewidth=0)

        style_ax(ax, title="Instrument Exposures", ylabel="Exposure")
        smart_date_axis(ax, exposures)

        handles = [mpatches.Patch(facecolor=colors[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, loc="upper left", framealpha=0.95,
                  fontsize=9, edgecolor=C.SPINE,
                  ncol=min(3, len(labels)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_turnover(instrument_calc: InstrumentCalculations, **kwargs) -> plt.Figure:
        """Plot instrument turnover as stacked area."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        turnover = instrument_calc.compute_turnover()
        if isinstance(turnover, pd.Series):
            turnover = turnover.to_frame()

        labels = [_clean_label(c) for c in turnover.columns]
        colors = C.get(len(labels))

        ax.stackplot(turnover.index,
                     *[turnover.iloc[:, i] for i in range(turnover.shape[1])],
                     labels=labels, colors=colors, alpha=0.80, linewidth=0)

        style_ax(ax, title="Instrument Turnover", ylabel="Turnover")
        smart_date_axis(ax, turnover)

        handles = [mpatches.Patch(facecolor=colors[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, loc="upper left", framealpha=0.95,
                  fontsize=9, edgecolor=C.SPINE,
                  ncol=min(3, len(labels)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_transaction_costs(
        instrument_calc: InstrumentCalculations, **kwargs
    ) -> plt.Figure:
        """Plot transaction costs for each instrument."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        tc = instrument_calc.compute_transaction_costs()
        if isinstance(tc, pd.Series):
            tc = tc.to_frame()

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            ax.plot(tc.index, tc[inst], color=colors[i],
                    linewidth=1.4, label=_clean_label(inst))

        style_ax(ax, title="Transaction Costs", ylabel="Cost")
        smart_date_axis(ax, tc)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_drawdown(instrument_calc: InstrumentCalculations, **kwargs) -> plt.Figure:
        """Plot drawdowns for each instrument."""
        ensure_style()
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        instruments = _get_instruments(instrument_calc)
        colors = C.get(len(instruments))

        for i, inst in enumerate(instruments):
            cum = (1 + instrument_calc.returns[inst]).cumprod()
            dd = (cum / cum.cummax() - 1) * 100
            ax.fill_between(dd.index, dd, 0, color=colors[i], alpha=0.15)
            ax.plot(dd.index, dd, color=colors[i], linewidth=1.2,
                    label=_clean_label(inst), alpha=0.85)

        ax.axhline(0, color=C.SPINE, lw=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0f}%"))

        style_ax(ax, title="Instrument Drawdowns", ylabel="Drawdown (%)")
        smart_date_axis(ax, instrument_calc.returns)
        ax.legend(loc="lower left", framealpha=0.95, fontsize=9,
                  edgecolor=C.SPINE, ncol=min(3, len(instruments)))

        fig.tight_layout()
        add_watermark(fig)
        return fig

    @staticmethod
    def plot_liquidity(instrument_calc: InstrumentCalculations, **kwargs) -> Optional[plt.Figure]:
        """Plot liquidity metrics (placeholder)."""
        return None

    @staticmethod
    def plot_forecasts(instrument_calc: InstrumentCalculations, **kwargs) -> Optional[plt.Figure]:
        """Plot forecasts (placeholder)."""
        return None

    # ── Strategy analysis ──

    @staticmethod
    def strategy_common_plot_elements(
        ax1, instruments_data, instrument, positions,
        buy_signals, sell_signals, price_col, *strategy_lines,
    ):
        """Plot common strategy elements on axis."""
        ensure_style()

        invested = positions != 0
        ax1.fill_between(
            invested.index, 0, 1, where=invested,
            transform=ax1.get_xaxis_transform(),
            color=C.GOLD, alpha=0.15,
        )

        ax1.plot(
            instruments_data.data[instrument, price_col].dropna(),
            label="Price", color=C.NAVY, linewidth=1.8,
        )
        for line_label, line_style, line_color in strategy_lines:
            ax1.plot(
                instruments_data.data[instrument, line_label].dropna(),
                label=line_label, linestyle=line_style, color=line_color,
                linewidth=1.4,
            )

        ax1.scatter(
            buy_signals.index[buy_signals],
            instruments_data.data[instrument, price_col][buy_signals],
            marker="^", color=C.GREEN, s=C.MARKER_LG, label="Buy Signal",
            edgecolors=C.EDGE, linewidths=C.LW_EDGE, zorder=5,
        )
        ax1.scatter(
            sell_signals.index[sell_signals],
            instruments_data.data[instrument, price_col][sell_signals],
            marker="v", color=C.RED, s=C.MARKER_LG, label="Sell Signal",
            edgecolors=C.EDGE, linewidths=C.LW_EDGE, zorder=5,
        )

        smart_date_axis(ax1, instruments_data.data[instrument, price_col])
        ax1.legend(loc="upper left", fontsize=9, framealpha=0.95,
                   edgecolor=C.SPINE)

        ax1_twin = ax1.twinx()
        if "returns" in instruments_data.data[instrument]:
            returns = instruments_data.data[instrument, "returns"].dropna()
            strategy_returns = returns * positions.shift(1)
            cumulative_returns = (1 + strategy_returns).cumprod() - 1
            ax1_twin.plot(
                cumulative_returns.index, cumulative_returns.values,
                label="Cumulative Returns", color=C.BLUE,
                alpha=0.80, linewidth=1.6,
            )
            ax1_twin.set_ylabel("Cumulative Returns", color=C.BLUE)
            ax1_twin.tick_params(axis="y", labelcolor=C.BLUE)
        else:
            strategy_returns = pd.Series(index=positions.index)
            cumulative_returns = pd.Series(index=positions.index)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", fontsize=9, framealpha=0.95,
                   edgecolor=C.SPINE)

        return cumulative_returns, strategy_returns

    @staticmethod
    def strategy_summary_statistics(
        ax2, cumulative_returns, strategy_returns,
        buy_signals, sell_signals, positions, initial_capital,
    ):
        """Plot strategy summary statistics."""
        n_buys = buy_signals.sum()
        n_sells = sell_signals.sum()

        trades = pd.DataFrame({"returns": strategy_returns, "position": positions})
        trades["holding_period"] = trades["position"].diff().abs()
        trades["trade_end"] = trades["holding_period"].shift(-1)
        trades = trades[trades["trade_end"] == 1]

        profitable = trades["returns"][trades["returns"] > 0]
        losing = trades["returns"][trades["returns"] < 0]

        avg_profit = profitable.mean() if len(profitable) > 0 else 0
        avg_loss = losing.mean() if len(losing) > 0 else 0
        total_profit = (profitable * initial_capital).sum()
        total_loss = (losing * initial_capital).sum()

        trades["holding_period"] = (
            trades.index - trades.index.to_series().shift(1)
        ).dt.days
        avg_holding = trades["holding_period"].mean()

        total_holding = positions[positions != 0].count()
        total_range = len(positions)
        final_ret = cumulative_returns.iloc[-1] if not cumulative_returns.empty else 0
        final_capital = initial_capital * (1 + final_ret)

        summary = (
            f"Buys: {n_buys}, Sells: {n_sells}\n"
            f"Avg Profit: {avg_profit:.2%}, Avg Loss: {avg_loss:.2%}\n"
            f"Total Profit: ${total_profit:.2f}, Total Loss: ${total_loss:.2f}\n"
            f"Avg Holding: {avg_holding:.1f} days\n"
            f"Holding: {total_holding} / {total_range} days\n"
            f"Final Return: {final_ret:.2%}\n"
            f"Final Capital: ${final_capital:.2f}"
        )

        stats_box(ax2, summary, loc="lower left")

    @staticmethod
    def strategy_plot_instrument(
        instruments_data, price_col, strategy_lines, instrument, initial_capital
    ):
        """Plot full strategy analysis for a single instrument."""
        ensure_style()
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}
        )

        positions = instruments_data.data[instrument, "position"]
        buy_signals = positions.diff() > 0
        sell_signals = positions.diff() < 0

        cumulative_returns, strategy_returns = (
            InstrumentPlots.strategy_common_plot_elements(
                ax1, instruments_data, instrument, positions,
                buy_signals, sell_signals, price_col, *strategy_lines,
            )
        )

        ax2.plot(cumulative_returns.index, cumulative_returns.values,
                 color=C.BLUE, linewidth=1.8, label="Strategy Returns")
        ax2.fill_between(cumulative_returns.index, cumulative_returns.values, 0,
                         alpha=0.15, color=C.BLUE)

        style_ax(ax2, title=f"{instrument} — Strategy Cumulative Returns",
                 ylabel="Cumulative Returns")
        smart_date_axis(ax2, cumulative_returns)
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.95,
                   edgecolor=C.SPINE)

        InstrumentPlots.strategy_summary_statistics(
            ax2, cumulative_returns, strategy_returns,
            buy_signals, sell_signals, positions, initial_capital,
        )

        fig.tight_layout()
        add_watermark(fig)
        return fig
