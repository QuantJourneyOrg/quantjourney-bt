# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Run the native strategy suite in the official QuantConnect LEAN image."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

NATIVE_DIR = Path(__file__).resolve().parent
COMPARE_DIR = NATIVE_DIR.parent
QC_DIR = COMPARE_DIR / "quantconnect"
for path in (QC_DIR, COMPARE_DIR, NATIVE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import STRATEGIES, TICKERS, write_native_result  # noqa: E402
from run_lean import (  # noqa: E402
    LEAN_DATA_DIR,
    LEAN_IMAGE,
    convert_to_lean_format,
    extract_auxiliary_data,
    generate_lean_config,
    parse_algorithm_seconds,
)

LEAN_RESULTS_DIR = NATIVE_DIR / "lean_results"
ALGORITHM_TEMPLATE_PATH = NATIVE_DIR / "lean_algorithm.py"


def prepare() -> None:
    subprocess.run(["docker", "ps"], capture_output=True, check=True, timeout=10)
    image = subprocess.run(
        ["docker", "image", "inspect", LEAN_IMAGE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if image.returncode != 0:
        subprocess.run(["docker", "pull", LEAN_IMAGE], check=True, timeout=600)
    extract_auxiliary_data()
    convert_to_lean_format()


def run(strategy: str) -> dict:
    strategy_name = f"strat_{strategy}"
    output_dir = LEAN_RESULTS_DIR / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    config_path = generate_lean_config(strategy_name, output_dir)
    algorithm_path = output_dir / "main.py"
    algorithm_path.write_text(
        ALGORITHM_TEMPLATE_PATH.read_text().replace("__NATIVE_STRATEGY__", strategy)
    )
    container_name = f"lean_native_{strategy}"
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{LEAN_DATA_DIR}:/Lean/Data:ro",
        "-v",
        f"{algorithm_path}:/Lean/Algorithm.Python/main.py:ro",
        "-v",
        f"{config_path}:/Lean/Launcher/bin/Debug/config.json:ro",
        "-v",
        f"{output_dir}:/Results",
        "--name",
        container_name,
        LEAN_IMAGE,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        raise
    wall_seconds = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError("LEAN native run failed:\n" + "\n".join(output.splitlines()[-40:]))

    nav_path = output_dir / "native_nav.csv"
    decisions_path = output_dir / "native_decisions.csv"
    if not nav_path.exists() or not decisions_path.exists():
        raise RuntimeError(f"LEAN native artifacts missing for {strategy}")
    nav = pd.read_csv(nav_path, parse_dates=["date"]).set_index("date")["nav"]
    decisions = pd.read_csv(decisions_path, parse_dates=["date"]).set_index("date")
    decisions = decisions.reindex(columns=TICKERS)
    algorithm_seconds = parse_algorithm_seconds(output)
    core_seconds = float(algorithm_seconds if algorithm_seconds is not None else wall_seconds)
    result = write_native_result(
        engine="quantconnect",
        strategy=strategy,
        nav=nav,
        core_seconds=core_seconds,
        wall_seconds=wall_seconds,
        decision_weights=decisions,
        extra={
            "engine_mode": "native LEAN indicators/history + market-on-close orders",
            "share_model": "whole shares",
            "algorithm_seconds": algorithm_seconds,
            "decision_count": len(decisions),
        },
    )
    for artifact in (nav_path, decisions_path):
        shutil.copyfile(
            artifact,
            output_dir / f"raw_{artifact.name}",
        )
    print(
        f"LEAN native {strategy}: core={core_seconds:.4f}s "
        f"wall={wall_seconds:.4f}s NAV={result['final_nav']:,.6f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", nargs="?", choices=STRATEGIES)
    args = parser.parse_args()
    prepare()
    selected = [args.strategy] if args.strategy else STRATEGIES
    for strategy in selected:
        run(strategy)


if __name__ == "__main__":
    main()
