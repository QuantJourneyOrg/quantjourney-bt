# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Orchestrate the 5 × 6 native-engine strategy benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

NATIVE_DIR = Path(__file__).resolve().parent
COMPARE_DIR = NATIVE_DIR.parent
REPO_ROOT = COMPARE_DIR.parent
if str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

from common import (  # noqa: E402
    CONTRACT_VERSION,
    DAILY_STRATEGIES,
    ENGINE_PREFIXES,
    EVALUATION_END,
    EVALUATION_START,
    PRIOR_SESSION,
    RESULTS_DIR,
    STRATEGIES,
    TICKERS,
    evaluation_index,
    require_data,
)

ENGINES = ["qj", "vectorbt", "pm_bt", "zipline", "backtrader", "quantconnect"]
SCRIPTS = {
    "qj": NATIVE_DIR / "qj_engine.py",
    "vectorbt": NATIVE_DIR / "vectorbt_engine.py",
    "pm_bt": NATIVE_DIR / "pm_bt_engine.py",
    "zipline": NATIVE_DIR / "zipline_engine.py",
    "backtrader": NATIVE_DIR / "backtrader_engine.py",
    "quantconnect": NATIVE_DIR / "lean_engine.py",
}


def python_for(engine: str) -> Path:
    paths = {
        "qj": REPO_ROOT / ".venv" / "bin" / "python",
        "vectorbt": COMPARE_DIR / ".envs" / "vbt" / "bin" / "python",
        "pm_bt": COMPARE_DIR / ".envs" / "pm_bt" / "bin" / "python",
        "zipline": COMPARE_DIR / ".envs" / "zipline" / "bin" / "python",
        "backtrader": COMPARE_DIR / ".envs" / "backtrader" / "bin" / "python",
        "quantconnect": REPO_ROOT / ".venv" / "bin" / "python",
    }
    path = paths[engine]
    if not path.exists():
        raise FileNotFoundError(f"Missing {engine} interpreter: {path}")
    return path


def clean_selected(engines: list[str], strategies: list[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for engine in engines:
        prefix = ENGINE_PREFIXES[engine]
        for strategy in strategies:
            for path in RESULTS_DIR.glob(f"{prefix}_{strategy}*"):
                if path.is_file():
                    path.unlink()


def _child_env(engine: str) -> dict[str, str]:
    matplotlib_config = Path(tempfile.gettempdir()) / "qj_native_matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(NATIVE_DIR), str(COMPARE_DIR), str(REPO_ROOT)]),
        "MPLCONFIGDIR": str(matplotlib_config),
    }
    if engine == "zipline":
        root = COMPARE_DIR / ".envs" / "native_zipline_data"
        root.mkdir(parents=True, exist_ok=True)
        env["ZIPLINE_ROOT"] = str(root)
    return env


def run_process(engine: str, strategies: list[str]) -> list[dict]:
    python = python_for(engine)
    script = SCRIPTS[engine]
    commands = (
        [[str(python), str(script)]]
        if engine == "quantconnect" and strategies == STRATEGIES
        else [[str(python), str(script), strategy] for strategy in strategies]
    )
    for command in commands:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=_child_env(engine),
            timeout=1800,
        )
        wall = time.perf_counter() - started
        print(completed.stdout, end="")
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            raise RuntimeError(
                f"Native {engine} command failed after {wall:.2f}s: {' '.join(command)}"
            )

    results = []
    prefix = ENGINE_PREFIXES[engine]
    for strategy in strategies:
        path = RESULTS_DIR / f"{prefix}_{strategy}.json"
        if not path.exists():
            raise RuntimeError(f"Native {engine} produced no result for {strategy}")
        results.append(json.loads(path.read_text()))
    return results


def collect_results() -> list[dict]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and value.get("benchmark_kind") == "native-strategy":
            rows.append(value)
    return rows


def build_summary() -> pd.DataFrame:
    rows = collect_results()
    if not rows:
        raise RuntimeError("No native results found")
    summary = pd.DataFrame(rows).sort_values(["strategy", "engine"])
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return summary


def validate_summary(summary: pd.DataFrame, *, require_complete: bool) -> None:
    expected = {(strategy, engine) for strategy in STRATEGIES for engine in ENGINES}
    observed = set(zip(summary["strategy"], summary["engine"], strict=True))
    if summary.duplicated(["strategy", "engine"]).any():
        raise AssertionError("Native summary contains duplicate engine/strategy rows")
    if require_complete and observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise AssertionError(f"Native result matrix mismatch: missing={missing}, extra={extra}")
    if not expected.issubset(observed):
        return

    complete = summary.set_index(["strategy", "engine"]).loc[sorted(expected)]
    if set(complete["contract_version"]) != {CONTRACT_VERSION}:
        raise AssertionError("Native results do not share one contract version")
    if complete["canonical_data_sha256"].nunique() != 1:
        raise AssertionError("Native results do not share one immutable data snapshot")
    expected_bars = len(evaluation_index())
    if not (complete["n_bars"] == expected_bars).all():
        raise AssertionError("Native results do not share the evaluation calendar")


def build_decision_divergence() -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        qj_path = RESULTS_DIR / f"qj_{strategy}_decision_weights.csv"
        if not qj_path.exists():
            continue
        comparison_start = PRIOR_SESSION if strategy in DAILY_STRATEGIES else EVALUATION_START
        qj = (
            pd.read_csv(qj_path, index_col=0, parse_dates=True)
            .reindex(columns=TICKERS)
            .loc[comparison_start:EVALUATION_END]
        )
        for engine, prefix in ENGINE_PREFIXES.items():
            path = RESULTS_DIR / f"{prefix}_{strategy}_decision_weights.csv"
            if not path.exists():
                continue
            other = (
                pd.read_csv(path, index_col=0, parse_dates=True)
                .reindex(columns=TICKERS)
                .loc[comparison_start:EVALUATION_END]
            )
            common_index = qj.index.intersection(other.index)
            delta = (
                qj.reindex(common_index).fillna(0.0) - other.reindex(common_index).fillna(0.0)
            ).abs()
            row_delta = delta.max(axis=1)
            divergent = row_delta.loc[lambda values: values > 1e-9]
            if divergent.empty:
                attribution = "none"
            elif engine == "vectorbt" and strategy == "02_rsi_reversion":
                attribution = "VectorBT rolling RSI versus Wilder-smoothed RSI"
            else:
                attribution = "unattributed"
            rows.append(
                {
                    "strategy": strategy,
                    "engine": engine,
                    "common_decision_rows": len(common_index),
                    "different_decision_rows": len(divergent),
                    "first_decision_divergence": (
                        None if divergent.empty else str(divergent.index[0].date())
                    ),
                    "max_abs_weight_difference": (
                        0.0 if row_delta.empty else float(row_delta.max())
                    ),
                    "attribution": attribution,
                }
            )
    report = pd.DataFrame(rows)
    report.to_csv(RESULTS_DIR / "decision_divergence.csv", index=False)
    return report


def validate_decision_divergence(report: pd.DataFrame, *, require_complete: bool) -> None:
    if not require_complete:
        return
    unexpected = report.loc[
        (report["different_decision_rows"] > 0)
        & ~((report["strategy"] == "02_rsi_reversion") & (report["engine"] == "vectorbt"))
    ]
    if not unexpected.empty:
        pairs = list(zip(unexpected["strategy"], unexpected["engine"], strict=True))
        raise AssertionError(f"Unattributed native target divergence: {pairs}")
    expected_rsi = report.loc[
        (report["strategy"] == "02_rsi_reversion") & (report["engine"] == "vectorbt"),
        "different_decision_rows",
    ]
    if len(expected_rsi) != 1 or int(expected_rsi.iloc[0]) == 0:
        raise AssertionError("Expected VectorBT native-RSI semantic divergence is absent")


def print_results(summary: pd.DataFrame) -> None:
    for metric in ("final_nav", "cagr", "sharpe", "max_dd", "core_seconds"):
        pivot = summary.pivot(index="strategy", columns="engine", values=metric)
        pivot = pivot.reindex(index=STRATEGIES, columns=ENGINES)
        print(f"\n{metric.upper()}\n{pivot.to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--strategy", type=int, choices=range(1, 6))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    require_data()

    if args.summary_only:
        summary = build_summary()
        divergence = build_decision_divergence()
        validate_summary(summary, require_complete=True)
        validate_decision_divergence(divergence, require_complete=True)
        print_results(summary)
        return

    engines = [args.engine] if args.engine else ENGINES
    strategies = [STRATEGIES[args.strategy - 1]] if args.strategy else STRATEGIES
    clean_selected(engines, strategies)
    for engine in engines:
        print(f"\n{'=' * 72}\nNATIVE ENGINE: {engine}\n{'=' * 72}")
        run_process(engine, strategies)

    summary = build_summary()
    divergence = build_decision_divergence()
    require_complete = not args.engine and not args.strategy
    validate_summary(summary, require_complete=require_complete)
    validate_decision_divergence(divergence, require_complete=require_complete)
    print_results(summary)


if __name__ == "__main__":
    main()
