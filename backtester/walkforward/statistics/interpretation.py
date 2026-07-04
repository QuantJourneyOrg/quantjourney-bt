"""
Interpretation engine — green / yellow / red traffic-light for WF metrics.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


Signal = Literal["green", "yellow", "red"]


@dataclass(frozen=True)
class MetricVerdict:
    """Single metric interpretation."""
    name: str
    value: float
    signal: Signal
    description: str


# ── Threshold table (from ADR-002 §7) ────────────────────────────────

_THRESHOLDS = {
    "overfit_ratio": {
        "green": lambda v: v < 1.5,
        "yellow": lambda v: 1.5 <= v <= 2.5,
        # else red
        "green_desc":  "< 1.5 — robust",
        "yellow_desc": "1.5–2.5 — caution",
        "red_desc":    "> 2.5 — likely overfit",
    },
    "efficiency": {
        "green": lambda v: v > 0.7,
        "yellow": lambda v: 0.4 <= v <= 0.7,
        "green_desc":  "> 0.7 — robust transfer",
        "yellow_desc": "0.4–0.7 — moderate degradation",
        "red_desc":    "< 0.4 — poor transfer",
    },
    "sharpe_decay": {
        "green": lambda v: v > -0.01,
        "yellow": lambda v: -0.05 <= v <= -0.01,
        "green_desc":  "> -0.01/fold — stable",
        "yellow_desc": "-0.01 to -0.05/fold — moderate decay",
        "red_desc":    "< -0.05/fold — alpha decaying",
    },
    "deflated_sharpe": {
        "green": lambda v: v > 2.0,
        "yellow": lambda v: 1.0 <= v <= 2.0,
        "green_desc":  "> 2.0 — robust vs multiple testing",
        "yellow_desc": "1.0–2.0 — marginal",
        "red_desc":    "< 1.0 — likely false positive",
    },
    "pbo": {
        "green": lambda v: v < 0.15,
        "yellow": lambda v: 0.15 <= v <= 0.40,
        "green_desc":  "< 0.15 — low overfit probability",
        "yellow_desc": "0.15–0.40 — moderate risk",
        "red_desc":    "> 0.40 — likely overfit",
    },
    "breakeven_bps": {
        "green": lambda v: v > 20,
        "yellow": lambda v: 10 <= v <= 20,
        "green_desc":  "> 20 bps — cost-robust",
        "yellow_desc": "10–20 bps — marginal",
        "red_desc":    "< 10 bps — cost-fragile",
    },
}


def _classify(metric_name: str, value: float) -> tuple[Signal, str]:
    """Return (signal, description) for a metric."""
    t = _THRESHOLDS.get(metric_name)
    if t is None:
        return "yellow", "no threshold defined"

    if t["green"](value):
        return "green", t["green_desc"]
    if t["yellow"](value):
        return "yellow", t["yellow_desc"]
    return "red", t["red_desc"]


def interpret_metrics(
    metrics: Dict[str, float],
) -> List[MetricVerdict]:
    """
    Classify a dict of WF metrics into traffic-light signals.

    Args:
        metrics: dict with keys like ``overfit_ratio``, ``efficiency``,
                 ``sharpe_decay``, ``deflated_sharpe``, ``pbo``, ``breakeven_bps``.

    Returns:
        List of ``MetricVerdict`` (one per metric that has a threshold).
    """
    verdicts = []
    for name, value in metrics.items():
        if name not in _THRESHOLDS:
            continue
        if value is None:
            continue
        signal, desc = _classify(name, value)
        verdicts.append(MetricVerdict(name=name, value=value, signal=signal, description=desc))
    return verdicts
