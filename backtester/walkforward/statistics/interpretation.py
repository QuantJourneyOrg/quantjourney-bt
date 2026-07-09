"""
Interpretation engine — green / yellow / red traffic-light for WF metrics.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import math
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


# ── Threshold table (walk-forward diagnostic heuristics) ─────────────
#
# NOTE: these thresholds are heuristic rules of thumb, not calibrated
# statistical tests. Boundaries (1.5, 0.7, -0.01, …) are conventions
# from practitioner literature — treat verdicts as indicative, never
# as formal inference.

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
    # DSR is a probability Φ(z) ∈ [0, 1] per Bailey & López de Prado (2014):
    # ≥ 0.95 → the Sharpe survives multiple-testing deflation at 95%
    # confidence; 0.80–0.95 marginal; < 0.80 likely false positive.
    "deflated_sharpe": {
        "green": lambda v: v >= 0.95,
        "yellow": lambda v: 0.80 <= v < 0.95,
        "green_desc":  ">= 0.95 — robust vs multiple testing",
        "yellow_desc": "0.80–0.95 — marginal",
        "red_desc":    "< 0.80 — likely false positive",
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


# Verdicts for these metrics are meaningless when the strategy loses
# out-of-sample: overfit_ratio(-1.2, -0.8) = 0.0 and a rising Sharpe
# slope on an always-losing strategy would otherwise render green.
_GATED_ON_OOS_SHARPE = ("overfit_ratio", "sharpe_decay")

# Below this fold count, fold-derived verdicts are indicative only.
_MIN_FOLDS_FOR_VERDICT = 6
_LOW_FOLD_METRICS = ("pbo", "sharpe_decay")


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
                 Context keys (no verdict of their own, used to gate the
                 others): ``composite_sharpe`` / ``oos_sharpe`` — the
                 aggregate OOS Sharpe; when <= 0, overfit_ratio and
                 sharpe_decay verdicts are forced red (a losing strategy
                 must never look "robust" or "stable"). ``n_folds`` —
                 when < 6, pbo and sharpe_decay descriptions are tagged
                 "(low fold count — indicative only)".

    Returns:
        List of ``MetricVerdict`` (one per metric that has a threshold).
        Metrics that are ``None`` or NaN (unavailable — e.g. PBO without
        per-trial OOS evaluation) are skipped and must be rendered as
        "n/a" by callers, never as a green verdict.
    """
    oos_sr = metrics.get("composite_sharpe", metrics.get("oos_sharpe"))
    losing = (
        oos_sr is not None
        and not (isinstance(oos_sr, float) and math.isnan(oos_sr))
        and oos_sr <= 0.0
    )
    n_folds = metrics.get("n_folds")
    low_folds = n_folds is not None and n_folds < _MIN_FOLDS_FOR_VERDICT

    verdicts = []
    for name, value in metrics.items():
        if name not in _THRESHOLDS:
            continue
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if name in _GATED_ON_OOS_SHARPE and losing:
            signal: Signal = "red"
            desc = (
                "OOS Sharpe <= 0 — strategy loses out-of-sample; "
                f"{name} verdict suppressed (never green on a losing strategy)"
            )
        elif name == "overfit_ratio" and value < 0.0:
            # IS Sharpe negative while OOS Sharpe positive: the ratio is
            # not interpretable — never let a negative ratio pass < 1.5
            # and render green.
            signal = "yellow"
            desc = "ratio < 0 — IS Sharpe negative; overfit ratio not interpretable"
        else:
            signal, desc = _classify(name, value)
        if low_folds and name in _LOW_FOLD_METRICS:
            desc = f"{desc} (low fold count — indicative only)"
        verdicts.append(MetricVerdict(name=name, value=value, signal=signal, description=desc))
    return verdicts
