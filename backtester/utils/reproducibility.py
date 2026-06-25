"""
Reproducibility — config fingerprinting, data versioning, execution assumptions.

Provides deterministic hashes so that two runs with the same inputs produce
the same fingerprint, enabling:
  - "Was this report generated with the same config?"
  - "Was the same dataset used?"
  - "What execution assumptions were in place?"

Usage:
    from backtester.utils.reproducibility import BacktestFingerprint

    fp = BacktestFingerprint.compute(
        strategy_params={...},
        returns=returns_series,
        config=perf_config_dict,
        fill_engine=engine,         # optional
    )
    fp.to_dict()   # embed in report
    fp.hash        # short hex digest

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestFingerprint:
    """Immutable reproducibility stamp for a backtest run."""

    # Short hex digests
    config_hash: str
    data_hash: str
    combined_hash: str

    # Human-readable metadata
    timestamp: str                # ISO-8601 UTC of computation
    data_points: int              # len(returns)
    data_start: str
    data_end: str

    # Execution assumptions (always present, may be "not specified")
    slippage_model: str
    slippage_params: Dict[str, Any]
    commission_model: str
    commission_params: Dict[str, Any]
    fill_at: str                  # "open" / "close" / "not specified"

    # Strategy config snapshot
    strategy_params: Dict[str, Any]

    @staticmethod
    def compute(
        *,
        strategy_params: Optional[Dict[str, Any]] = None,
        returns: Optional[pd.Series] = None,
        prices: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        fill_engine: Optional[Any] = None,
        initial_capital: float = 0.0,
        risk_free_rate: float = 0.0,
    ) -> "BacktestFingerprint":
        """Build a fingerprint from run inputs."""

        strat = strategy_params or {}

        # ── Config hash ──────────────────────────────────────────────
        config_payload = {
            "strategy_params": _make_hashable(strat),
            "initial_capital": initial_capital,
            "risk_free_rate": risk_free_rate,
            "config": _make_hashable(config or {}),
        }
        config_hash = _sha256_short(config_payload)

        # ── Data hash ────────────────────────────────────────────────
        data_parts: list[str] = []
        data_points = 0
        data_start = ""
        data_end = ""

        if returns is not None and len(returns) > 0:
            idx = returns.index
            data_start = str(idx[0].date()) if hasattr(idx[0], "date") else str(idx[0])
            data_end = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
            data_points = len(returns)
            # Hash: shape + first/last 10 values + sum (fast, deterministic)
            vals = returns.values.astype(np.float64)
            head = vals[:10].tobytes()
            tail = vals[-10:].tobytes()
            data_parts.append(hashlib.sha256(head + tail).hexdigest()[:16])
            data_parts.append(f"{vals.sum():.12f}")
            data_parts.append(str(data_points))

        if prices is not None and not prices.empty:
            shape_str = f"{prices.shape[0]}x{prices.shape[1]}"
            data_parts.append(shape_str)
            # Hash first & last row
            first_row = prices.iloc[0].values.astype(np.float64).tobytes()
            last_row = prices.iloc[-1].values.astype(np.float64).tobytes()
            data_parts.append(hashlib.sha256(first_row + last_row).hexdigest()[:16])

        data_hash = hashlib.sha256("|".join(data_parts).encode()).hexdigest()[:12]

        # ── Combined hash ────────────────────────────────────────────
        combined = hashlib.sha256(f"{config_hash}:{data_hash}".encode()).hexdigest()[:16]

        # ── Execution assumptions ────────────────────────────────────
        slip_model = "not specified"
        slip_params: Dict[str, Any] = {}
        comm_model = "not specified"
        comm_params: Dict[str, Any] = {}
        fill_at = "not specified"

        if fill_engine is not None:
            fill_at = getattr(fill_engine, "fill_at", "not specified")

            slip = getattr(fill_engine, "slippage", None)
            if slip is not None:
                slip_model = type(slip).__name__
                for attr in ("bps", "vol_factor", "sigma_daily", "adv", "eta"):
                    if hasattr(slip, attr):
                        slip_params[attr] = getattr(slip, attr)

            comm = getattr(fill_engine, "commission", None)
            if comm is not None:
                comm_model = type(comm).__name__
                for attr in ("bps", "cost_per_share", "min_per_order", "max_pct"):
                    if hasattr(comm, attr):
                        comm_params[attr] = getattr(comm, attr)

        return BacktestFingerprint(
            config_hash=config_hash,
            data_hash=data_hash,
            combined_hash=combined,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            data_points=data_points,
            data_start=data_start,
            data_end=data_end,
            slippage_model=slip_model,
            slippage_params=slip_params,
            commission_model=comm_model,
            commission_params=comm_params,
            fill_at=fill_at,
            strategy_params=strat,
        )

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def hash(self) -> str:
        """Short combined hash for display."""
        return self.combined_hash

    def to_dict(self) -> Dict[str, Any]:
        """Full fingerprint as a flat dict (for report embedding)."""
        d: Dict[str, Any] = {
            "fingerprint": self.combined_hash,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "generated_utc": self.timestamp,
            "data_points": self.data_points,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "fill_at": self.fill_at,
            "slippage_model": self.slippage_model,
            "commission_model": self.commission_model,
        }
        if self.slippage_params:
            for k, v in self.slippage_params.items():
                d[f"slippage_{k}"] = v
        if self.commission_params:
            for k, v in self.commission_params.items():
                d[f"commission_{k}"] = v
        return d

    def assumptions_text(self) -> str:
        """Human-readable execution assumptions block for report footer."""
        lines = [
            f"Backtest Fingerprint: {self.combined_hash}",
            f"Generated: {self.timestamp}",
            f"Data: {self.data_start} → {self.data_end} ({self.data_points} points, hash={self.data_hash})",
            f"Config hash: {self.config_hash}",
            f"Fill timing: {self.fill_at}",
            f"Slippage: {self.slippage_model}" + (f" ({self.slippage_params})" if self.slippage_params else ""),
            f"Commission: {self.commission_model}" + (f" ({self.commission_params})" if self.commission_params else ""),
        ]
        return "\n".join(lines)


# ── Sanity Guards ────────────────────────────────────────────────────

def run_sanity_checks(
    trade_analytics: Dict[str, Any],
    returns: pd.Series,
    initial_capital: float,
) -> list[str]:
    """
    Run pipeline-level sanity checks after all metrics are computed.

    Returns a list of human-readable warnings.  Empty list = all clear.

    Checks:
      1. Volume vs capital ratio (is $24M volume on $100k plausible?)
      2. Holding period vs track record (503 days avg on 252-day test?)
      3. Trade count vs round-trip consistency
      4. Turnover vs volume cross-check
      5. PnL sum should roughly match NAV change
    """
    warnings: list[str] = []
    ta = trade_analytics or {}

    n_days = len(returns)
    if n_days == 0:
        return ["⚠ No return data available"]

    # ── 1. Volume/capital ratio ──────────────────────────────────────
    total_vol = ta.get("total_volume", 0)
    if total_vol > 0 and initial_capital > 0:
        vol_to_cap = total_vol / initial_capital
        if isinstance(returns.index, pd.DatetimeIndex) and len(returns.index) >= 2:
            n_years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1e-9)
        else:
            n_years = n_days / 252
        annual_vol_to_cap = vol_to_cap / n_years if n_years > 0 else vol_to_cap
        if annual_vol_to_cap > 500:
            warnings.append(
                f"⚠ Volume/Capital: ${total_vol:,.0f} traded on ${initial_capital:,.0f} "
                f"= {annual_vol_to_cap:.0f}× annual ratio. Very high — check trade counting."
            )

    # ── 2. Holding period vs track record ────────────────────────────
    avg_hold = ta.get("avg_holding_days", 0)
    if avg_hold > n_days:
        warnings.append(
            f"⚠ Avg holding period ({avg_hold:.0f} days) exceeds track record "
            f"({n_days} trading days). Possible open positions counted as closed."
        )

    # ── 3. Trade/RT consistency ──────────────────────────────────────
    if not ta.get("trade_rt_consistent", True):
        ratio = ta.get("trade_to_rt_ratio", 0)
        warnings.append(
            f"⚠ Trade coverage strict lot-match check: blotter trades / (2 × FIFO lot round-trips) = {ratio:.2f}. "
            f"Simple entry/exit cycles should be near 1.0; weight-based rebalances can legitimately "
            f"create many partial lot matches, so interpret round-trip counts as lot-level analytics."
        )

    # ── 4. Volume consistency ────────────────────────────────────────
    if not ta.get("volume_consistent", True):
        vr = ta.get("volume_consistency_ratio", 0)
        warnings.append(
            f"⚠ Round-trip notional vs blotter volume ratio = {vr:.2f} "
            f"(expected ~1.0). Trade values may be incorrectly recorded."
        )

    # ── 5. PnL vs NAV change ────────────────────────────────────────
    nav_change = ta.get("net_profit", 0)
    n_rt = ta.get("total_round_trips", 0)
    expectancy = ta.get("expectancy", 0)
    if n_rt > 0 and abs(nav_change) > 0:
        implied_pnl = expectancy * n_rt
        # Allow 50% tolerance (open positions, unrealized PnL)
        if abs(nav_change) > 0 and abs(implied_pnl / nav_change - 1) > 0.5:
            warnings.append(
                f"⚠ Realized PnL ({n_rt} RTs × ${expectancy:,.0f} = ${implied_pnl:,.0f}) "
                f"differs from NAV change (${nav_change:,.0f}) by >{50}%. "
                f"Gap likely due to unrealized positions or cost booking."
            )

    return warnings


# ── Report Definitions ───────────────────────────────────────────────

METRIC_DEFINITIONS = {
    "Trade": (
        "A single fill execution (buy or sell) recorded in the blotter. "
        "A simple position cycle often has 2 trades (entry + exit), but "
        "scaling, partial fills, and FIFO lot matching can create a different "
        "trade-to-round-trip relationship."
    ),
    "Round Trip": (
        "A completed FIFO-matched lot: entry quantity paired with exit quantity "
        "until that lot is fully closed. P&L, holding period, and win/loss stats "
        "are computed per matched lot. In weight-based strategies this is not "
        "the same as a high-level strategy lifecycle because one rebalance fill "
        "can close several historical lots."
    ),
    "Trade Coverage vs 2xRT": (
        "Blotter trades divided by two times FIFO lot round-trips. A value near "
        "1.0 is expected for simple entry/exit pairs; lower values can occur "
        "when one rebalance trade closes several historical lots."
    ),
    "RT Position-Day Load": (
        "Average holding days times FIFO round-trips divided by track-record days. "
        "This is a lot-level load diagnostic, not a count of live positions."
    ),
    "Turnover (ann.)": (
        "Dollar turnover: (total notional traded / 2) / avg NAV × 252. "
        "Measures how many times the portfolio is 'turned over' per year. "
        "100% = entire portfolio replaced once per year."
    ),
    "Holding Period": (
        "Calendar days from entry fill to exit fill of a FIFO lot round-trip. "
        "Avg holding period is weighted equally across matched lots."
    ),
    "Win Rate (Trade)": (
        "Fraction of round-trips with positive net P&L (after costs)."
    ),
    "Expectancy": (
        "Average net P&L per round-trip. "
        "Expectancy × round_trips ≈ total realized P&L."
    ),
    "Profit Factor (Trade)": (
        "Sum of winning round-trip P&L / abs(sum of losing round-trip P&L). "
        "> 1.0 means gross wins exceed gross losses."
    ),
    "Worst Loss Magnitude": (
        "Positive loss magnitude from the daily return distribution. It is shown "
        "as a positive number by convention; it is not a positive return."
    ),
    "Kelly Criterion (Daily Returns)": (
        "Kelly fraction estimated from daily portfolio return win/loss frequency "
        "and average daily win/loss size, not from FIFO round-trip expectancy."
    ),
    "Exposure Path Checks": (
        "Full-period diagnostics from the realised weight path: gross exposure, "
        "cash sleeve bounds, and finite-weight checks."
    ),
    "Consistency Checks": (
        "Automated cross-validation between trade count, round-trip count, "
        "volume, and holding periods. ✓ = within expected bounds, ✗ = review needed."
    ),
    "Fingerprint": (
        "SHA-256 hash of config + data, enabling exact reproducibility verification. "
        "Two runs with identical inputs produce the same fingerprint."
    ),
}


def format_definitions_block() -> str:
    """Render definitions as a plain-text block for report footer."""
    lines = ["", "─── Definitions ───"]
    for term, defn in METRIC_DEFINITIONS.items():
        # Wrap at ~80 chars
        wrapped = defn.replace("\n", " ").strip()
        lines.append(f"  {term}: {wrapped}")
    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_hashable(obj: Any) -> Any:
    """Convert dicts/lists to a JSON-serializable, sorted form."""
    if isinstance(obj, dict):
        return {k: _make_hashable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_make_hashable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 10)
    if isinstance(obj, (np.integer, np.floating)):
        return round(float(obj), 10)
    return obj


def _sha256_short(payload: Any, length: int = 12) -> str:
    """Deterministic short SHA-256 hex digest of a JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:length]
