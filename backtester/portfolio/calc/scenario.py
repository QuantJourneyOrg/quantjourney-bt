"""
Scenario Analysis - Historical Windows and Stress Testing

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import pandas as pd


def historical_scenario_analysis(returns: pd.DataFrame, scenarios: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    out = {}
    for name, (start, end) in scenarios.items():
        window = returns.loc[start:end]
        out[name] = (1 + window).prod() - 1
    return pd.DataFrame(out)


def stress_test(returns: pd.DataFrame, shocks: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = {}
    for scen, mapping in shocks.items():
        adj = returns.copy()
        for inst, shock in mapping.items():
            if inst in adj.columns:
                adj[inst] = adj[inst] + shock
        out[scen] = (1 + adj).prod() - 1
    return pd.DataFrame(out)


def stress_test_vectorized(
    returns: pd.DataFrame, scenarios: Dict[str, Union[pd.Series, pd.DataFrame]],
    *, mode: str = "additive",
) -> pd.DataFrame:
    out = {}
    for name, shock in scenarios.items():
        adj = returns.copy()
        if isinstance(shock, pd.Series):
            s = shock.reindex(adj.columns).fillna(0.0)
            if mode == "additive":
                adj = adj.add(s, axis=1)
            else:
                adj = adj.mul(1.0 + s, axis=1) - 1.0
        elif isinstance(shock, pd.DataFrame):
            s = shock.reindex(index=adj.index, columns=adj.columns).fillna(0.0)
            if mode == "additive":
                adj = adj + s
            else:
                adj = (1.0 + adj) * (1.0 + s) - 1.0
        else:
            raise TypeError("Shock must be a Series or DataFrame")
        out[name] = (1.0 + adj).prod() - 1.0
    return pd.DataFrame(out)
