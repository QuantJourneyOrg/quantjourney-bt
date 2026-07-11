# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Shared inputs and result serialization for the native-engine benchmark.

This module deliberately contains no strategy builders and never reads the
strict benchmark's decision/execution matrices.  Every engine implementation
must calculate indicators, state, target weights, and rebalance dates itself.
Only the immutable market-data panel and reporting contract are shared.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_DIR = Path(__file__).resolve().parent
COMPARE_DIR = NATIVE_DIR.parent
DATA_PATH = COMPARE_DIR / "protocol_artifacts" / "ohlcv.parquet"
RESULTS_DIR = COMPARE_DIR / "native_results"

CONTRACT_VERSION = "native-close-v1"
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
STRATEGIES = [
    "01_sma_crossover",
    "02_rsi_reversion",
    "03_monthly_rebalance",
    "04_momentum_voltarget",
    "05_dual_momentum",
]
DAILY_STRATEGIES = {"01_sma_crossover", "02_rsi_reversion"}
ENGINE_PREFIXES = {
    "qj": "qj",
    "vectorbt": "vbt",
    "pm_bt": "pb",
    "zipline": "zipline",
    "backtrader": "bt",
    "quantconnect": "qc",
}

WARMUP_START = pd.Timestamp("2015-01-02")
PRIOR_SESSION = pd.Timestamp("2015-12-31")
EVALUATION_START = pd.Timestamp("2016-01-04")
EVALUATION_END = pd.Timestamp("2024-12-31")
INITIAL_CAPITAL = 100_000.0
CASH_BUFFER = 0.001
TOTAL_COST_BPS = 0.0

SMA_FAST = 50
SMA_SLOW = 200
SMA_POSITION_CAP = 0.25
RSI_PERIOD = 14
RSI_ENTRY = 30.0
RSI_EXIT = 70.0
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP_RECENT = 21
MOMENTUM_TOP_N = 3
VOLATILITY_LOOKBACK = 63
VOLATILITY_TARGET = 0.15
DUAL_MOMENTUM_TOP_N = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_data() -> Path:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Native benchmark data is missing: {DATA_PATH}. "
            "Provide the shared adjusted OHLCV panel described in "
            "compare/native/README.md (canonical SHA-256 is published there)."
        )
    return DATA_PATH


def load_ohlcv() -> pd.DataFrame:
    frame = pd.read_parquet(require_data()).copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    frame = frame.sort_index().loc[WARMUP_START:EVALUATION_END]
    expected = [(ticker, field) for ticker in TICKERS for field in ("open", "high", "low", "close", "volume")]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Native OHLCV panel is missing columns: {missing}")
    if frame.index.has_duplicates or frame.isna().any().any():
        raise ValueError("Native OHLCV panel must be complete with a unique calendar")
    return frame.loc[:, expected]


def load_close() -> pd.DataFrame:
    close = load_ohlcv().xs("close", axis=1, level=1)
    return close.reindex(columns=TICKERS).astype(float)


def evaluation_index() -> pd.DatetimeIndex:
    return load_close().loc[EVALUATION_START:EVALUATION_END].index


def first_session_flags(index: pd.DatetimeIndex) -> pd.Series:
    periods = pd.Series(index.to_period("M"), index=index)
    return (periods != periods.shift(1)).astype(bool)


def decision_flags(strategy: str, index: pd.DatetimeIndex) -> pd.Series:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown native strategy: {strategy}")
    if strategy in DAILY_STRATEGIES:
        return pd.Series(True, index=index, dtype=bool)
    return first_session_flags(index) & (index >= EVALUATION_START)


def execution_flags(strategy: str, index: pd.DatetimeIndex) -> pd.Series:
    shifted = decision_flags(strategy, index).shift(1, fill_value=False).astype(bool)
    return shifted & (index >= EVALUATION_START) & (index <= EVALUATION_END)


def execution_dates(strategy: str, index: pd.DatetimeIndex) -> tuple[pd.Timestamp, ...]:
    flags = execution_flags(strategy, index)
    return tuple(pd.Timestamp(value) for value in flags.loc[lambda values: values].index)


def normalize_nav(nav: pd.Series) -> pd.Series:
    out = pd.Series(nav, copy=True)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = pd.to_numeric(out, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.loc[EVALUATION_START:EVALUATION_END]
    expected = evaluation_index()
    missing = expected.difference(out.index)
    extra = out.index.difference(expected)
    if len(missing) or len(extra):
        raise AssertionError(
            f"Native NAV calendar mismatch: missing={list(missing[:5])}, "
            f"extra={list(extra[:5])}"
        )
    out = out.reindex(expected)
    if out.isna().any() or not np.isfinite(out.to_numpy(dtype=float)).all():
        raise AssertionError("Native NAV contains missing or non-finite values")
    out.name = "nav"
    return out.astype(float)


def metrics_from_nav(nav: pd.Series) -> dict[str, float | int]:
    clean = normalize_nav(nav)
    returns = clean.pct_change().dropna()
    years = len(returns) / 252.0
    final_nav = float(clean.iloc[-1])
    cagr = (final_nav / float(clean.iloc[0])) ** (1.0 / years) - 1.0 if years else 0.0
    volatility = float(returns.std(ddof=1)) if len(returns) else 0.0
    sharpe = float(returns.mean() / volatility * np.sqrt(252.0)) if volatility > 0 else 0.0
    max_dd = float((clean / clean.cummax() - 1.0).min())
    return {
        "final_nav": round(final_nav, 6),
        "cagr": round(cagr, 6),
        "sharpe": round(sharpe, 6),
        "max_dd": round(max_dd, 6),
        "n_bars": int(len(clean)),
    }


def _normalize_decisions(weights: pd.DataFrame) -> pd.DataFrame:
    out = weights.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.reindex(columns=TICKERS).astype(float)
    if out.isna().any().any() or not np.isfinite(out.to_numpy()).all():
        raise AssertionError("Native decision weights contain invalid values")
    if (out < -1e-12).any().any() or (out.sum(axis=1) > 1.0 + 1e-12).any():
        raise AssertionError("Native decision weights violate long-only gross-exposure limits")
    return out


def write_native_result(
    *,
    engine: str,
    strategy: str,
    nav: pd.Series,
    core_seconds: float,
    wall_seconds: float | None = None,
    decision_weights: pd.DataFrame | None = None,
    extra: dict | None = None,
) -> dict:
    if engine not in ENGINE_PREFIXES:
        raise ValueError(f"Unknown native engine: {engine}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown native strategy: {strategy}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_nav = normalize_nav(nav)
    prefix = ENGINE_PREFIXES[engine]
    clean_nav.to_csv(RESULTS_DIR / f"{prefix}_{strategy}_equity.csv", index_label="date")

    if decision_weights is not None:
        decisions = _normalize_decisions(decision_weights)
        decisions.to_csv(
            RESULTS_DIR / f"{prefix}_{strategy}_decision_weights.csv",
            index_label="date",
        )

    result = {
        "engine": engine,
        "strategy": strategy,
        "benchmark_kind": "native-strategy",
        "contract_version": CONTRACT_VERSION,
        "canonical_data_sha256": sha256_file(require_data()),
        "target_source": "computed independently inside the engine adapter",
        "core_seconds": round(float(core_seconds), 6),
        "elapsed_seconds": round(float(core_seconds), 6),
        **metrics_from_nav(clean_nav),
    }
    if wall_seconds is not None:
        result["wall_clock_seconds"] = round(float(wall_seconds), 6)
    if extra:
        result.update(extra)
    (RESULTS_DIR / f"{prefix}_{strategy}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
