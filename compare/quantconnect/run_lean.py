# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

#!/usr/bin/env python3
"""
QuantConnect LEAN Runner
========================
Converts shared yfinance data to LEAN format and runs each strategy
via Docker. Requires Docker Desktop running.

Usage:
    python run_lean.py                    # run all 5 strategies
    python run_lean.py strat_03           # run single strategy
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
COMPARE_DIR = BASE_DIR.parent
if str(COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARE_DIR))

from canonical import (  # noqa: E402
    ARTIFACTS_DIR,
    CANONICAL_DATA_PATH,
    INTEGER_ENGINE_CAPITAL_SCALE,
    TICKERS,
    load_bundle,
)
from result_utils import write_result  # noqa: E402

RESULTS_DIR = COMPARE_DIR / "results"
LEAN_DATA_DIR = BASE_DIR / "lean_data"

LEAN_IMAGE = "quantconnect/lean:latest"


# ─── Extract auxiliary data from LEAN image ───────────────────────


def extract_auxiliary_data():
    """
    Copy market-hours, symbol-properties and other auxiliary files
    from the LEAN Docker image into our local lean_data/ directory.
    These are needed for the engine to run but are NOT in our parquet.
    """
    aux_dirs = ["market-hours", "symbol-properties", "alternative/interest-rate"]
    for d in aux_dirs:
        local = LEAN_DATA_DIR / d
        if local.exists() and any(local.iterdir()):
            continue  # already extracted
        local.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting {d} from LEAN image...")
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{local}:/out",
                "--entrypoint",
                "/bin/bash",
                LEAN_IMAGE,
                "-c",
                f"cp -r /Lean/Data/{d}/* /out/",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

    # Also extract existing map_files and factor_files from LEAN
    # (contains entries for thousands of tickers including splits etc.)
    for d in ["equity/usa/map_files", "equity/usa/factor_files"]:
        local = LEAN_DATA_DIR / d
        if local.exists() and len(list(local.iterdir())) > 10:
            continue
        local.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting {d} from LEAN image...")
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{local}:/out",
                "--entrypoint",
                "/bin/bash",
                LEAN_IMAGE,
                "-c",
                f"cp -r /Lean/Data/{d}/* /out/",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

    print("  Auxiliary data ready.")


# ─── Data Conversion: yfinance → LEAN format ─────────────────────


def convert_to_lean_format():
    """
    Convert shared OHLCV parquet to LEAN's binary daily format.

    LEAN daily equity format:
        Path: data/equity/usa/daily/{ticker}.zip
        Inside zip: {ticker}.csv
        CSV columns: Date(yyyyMMdd 00:00),Open*10000,High*10000,Low*10000,Close*10000,Volume
    """
    ohlcv_path = CANONICAL_DATA_PATH
    if not ohlcv_path.exists():
        print(f"ERROR: {ohlcv_path} not found. Run: python canonical.py")
        sys.exit(1)

    raw = pd.read_parquet(ohlcv_path)

    equity_dir = LEAN_DATA_DIR / "equity" / "usa" / "daily"
    equity_dir.mkdir(parents=True, exist_ok=True)

    # Map/factor dirs (already extracted from LEAN, we override per-ticker)
    map_dir = LEAN_DATA_DIR / "equity" / "usa" / "map_files"
    factor_dir = LEAN_DATA_DIR / "equity" / "usa" / "factor_files"
    map_dir.mkdir(parents=True, exist_ok=True)
    factor_dir.mkdir(parents=True, exist_ok=True)

    import zipfile

    for ticker in TICKERS:
        if ticker not in raw.columns.get_level_values(0):
            print(f"  SKIP {ticker}: not in parquet")
            continue

        df = raw[ticker].copy()
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["close"])

        # LEAN CSV format: yyyyMMdd 00:00,Open*10000,High*10000,Low*10000,Close*10000,Volume
        ticker_lower = ticker.lower()

        lines = []
        for date, row in df.iterrows():
            date_str = date.strftime("%Y%m%d 00:00")
            o = int(round(row["open"] * 10000))
            h = int(round(row["high"] * 10000))
            low = int(round(row["low"] * 10000))
            c = int(round(row["close"] * 10000))
            v = int(row["volume"])
            lines.append(f"{date_str},{o},{h},{low},{c},{v}")

        csv_content = "\n".join(lines)

        # Write as zip
        zip_path = equity_dir / f"{ticker_lower}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{ticker_lower}.csv", csv_content)

        # Override map file for our tickers (our data is already adjusted,
        # so map = identity from first_date to last_date)
        first_date = df.index[0].strftime("%Y%m%d")
        last_date = df.index[-1].strftime("%Y%m%d")
        map_content = (
            f"{first_date},{ticker_lower},{ticker_lower}\n"
            f"{last_date},{ticker_lower},{ticker_lower}\n"
        )
        map_path = map_dir / f"{ticker_lower}.csv"
        map_path.write_text(map_content)

        # Factor file: no adjustments (data is already adjusted via yfinance auto_adjust)
        factor_content = f"{first_date},1,1,0\n{last_date},1,1,0\n"
        factor_path = factor_dir / f"{ticker_lower}.csv"
        factor_path.write_text(factor_content)

        print(f"  {ticker}: {len(lines)} bars → {zip_path.name}")

    print(f"LEAN data ready: {equity_dir}")


# ─── LEAN Config Generation ──────────────────────────────────────


def generate_lean_config(strategy_name: str, output_dir: Path) -> Path:
    """Generate a config.json that LEAN can read at /Lean/Launcher/bin/Debug/config.json."""
    algo_class = get_class_name(strategy_name)
    # Algorithm location relative to /Lean/Launcher/bin/Debug/
    algo_location = "../../../Algorithm.Python/main.py"

    config = {
        "environment": "backtesting",
        "algorithm-type-name": algo_class,
        "algorithm-language": "Python",
        "algorithm-location": algo_location,
        "data-folder": "../../../Data/",
        "results-destination-folder": "/Results",
        "messaging-handler": "QuantConnect.Messaging.Messaging",
        "job-queue-handler": "QuantConnect.Queues.JobQueue",
        "api-handler": "QuantConnect.Api.Api",
        "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
        "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
        "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
        "object-store": "QuantConnect.Lean.Engine.Storage.LocalObjectStore",
        "data-permissions-manager": "QuantConnect.Data.DefaultDataPermissionManager",
        "log-handler": "QuantConnect.Logging.CompositeLogHandler",
        "close-automatically": True,
        "live-mode": False,
        "parameters": {"strategy": strategy_name.replace("strat_", "")},
        "environments": {
            "backtesting": {
                "live-mode": False,
                "setup-handler": "QuantConnect.Lean.Engine.Setup.ConsoleSetupHandler",
                "result-handler": "QuantConnect.Lean.Engine.Results.BacktestingResultHandler",
                "data-feed-handler": "QuantConnect.Lean.Engine.DataFeeds.FileSystemDataFeed",
                "real-time-handler": "QuantConnect.Lean.Engine.RealTime.BacktestingRealTimeHandler",
                "history-provider": (
                    "QuantConnect.Lean.Engine.HistoricalData.SubscriptionDataReaderHistoryProvider"
                ),
                "transaction-handler": (
                    "QuantConnect.Lean.Engine.TransactionHandlers.BacktestingTransactionHandler"
                ),
            }
        },
    }

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def get_class_name(strategy_name: str) -> str:
    """Map strategy folder name to Python class name."""
    mapping = {
        "strat_01_sma_crossover": "SMACrossoverAlgorithm",
        "strat_02_rsi_reversion": "RSIMeanReversionAlgorithm",
        "strat_03_monthly_rebalance": "MonthlyEqualWeightAlgorithm",
        "strat_04_momentum_voltarget": "MomentumVolTargetAlgorithm",
        "strat_05_dual_momentum": "DualMomentumAlgorithm",
    }
    return mapping.get(strategy_name, "QCAlgorithm")


# ─── Run via Docker ──────────────────────────────────────────────


def run_strategy(strategy_name: str) -> dict:
    """Run a single LEAN strategy via Docker."""
    strat_dir = BASE_DIR / strategy_name
    main_py = strat_dir / "main.py"
    if not main_py.exists():
        return {"error": f"Strategy not found: {main_py}"}

    # Create results dir inside strategy folder
    results_dir = strat_dir / "results"
    results_dir.mkdir(exist_ok=True)
    for old_result in results_dir.iterdir():
        if old_result.is_file():
            old_result.unlink()

    # Generate LEAN config
    config_path = generate_lean_config(strategy_name, strat_dir)

    print(f"\n{'=' * 60}")
    print(f"  Running LEAN: {strategy_name}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()

    # Run LEAN engine in Docker
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{LEAN_DATA_DIR}:/Lean/Data:ro",
        "-v",
        f"{main_py}:/Lean/Algorithm.Python/main.py:ro",
        "-v",
        f"{BASE_DIR}:/Lean/Algorithm.Python/qj_compare:ro",
        "-v",
        f"{ARTIFACTS_DIR}:/Canonical:ro",
        "-v",
        f"{config_path}:/Lean/Launcher/bin/Debug/config.json:ro",
        "-v",
        f"{results_dir}:/Results",
        "--name",
        f"lean_{strategy_name}",
        LEAN_IMAGE,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(f"  FAILED (exit={result.returncode})")
            # Print last 30 lines of stderr/stdout for debugging
            output = result.stdout + result.stderr
            for line in output.strip().split("\n")[-30:]:
                print(f"    {line}")
            return {
                "engine": "quantconnect",
                "strategy": strategy_name.replace("strat_", ""),
                "error": f"exit={result.returncode}",
                "elapsed_seconds": round(elapsed, 3),
            }

        # Parse results from LEAN output. Keep the internal algorithm time
        # separate from Docker/process startup and result export time.
        output = result.stdout + result.stderr
        algorithm_seconds = parse_algorithm_seconds(output)
        parsed = parse_lean_results(strategy_name, results_dir, elapsed, algorithm_seconds)
        if "error" in parsed:
            # Show LEAN output for debugging
            output = result.stdout + result.stderr
            lines = output.strip().split("\n")
            print(f"  LEAN stdout/stderr ({len(lines)} lines), last 40:")
            for line in lines[-40:]:
                print(f"    {line}")
        return parsed

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        subprocess.run(["docker", "kill", f"lean_{strategy_name}"], capture_output=True)
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {"engine": "quantconnect", "strategy": strategy_name, "error": "timeout"}


def parse_algorithm_seconds(output: str) -> float | None:
    """Extract LEAN's own algorithm timer from launcher output."""
    import re

    matches = re.findall(r"completed in ([0-9.]+) seconds", output)
    return float(matches[-1]) if matches else None


def parse_lean_results(
    strategy_name: str,
    results_dir: Path,
    elapsed: float,
    algorithm_seconds: float | None,
) -> dict:
    """Parse LEAN output and serialize its daily pre-trade NAV."""
    strat_label = strategy_name.replace("strat_", "")
    bundle = load_bundle(strat_label)
    canonical_nav_path = results_dir / "canonical_nav.csv"
    if not canonical_nav_path.exists():
        return {
            "engine": "quantconnect",
            "strategy": strat_label,
            "error": "canonical_nav.csv missing",
        }
    nav = pd.read_csv(canonical_nav_path, parse_dates=["date"]).set_index("date")["nav"]

    # LEAN writes results as {ClassName}.json
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        print(f"  No result files in {results_dir}")
        return {"engine": "quantconnect", "strategy": strat_label, "error": "no results"}

    # Find the main result file (the largest .json, or matching class name)
    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
        except json.JSONDecodeError:
            continue

        # Skip non-dict entries (some files are lists)
        if not isinstance(data, dict):
            continue

        # LEAN uses camelCase keys: statistics, totalPerformance, charts
        stats = data.get("statistics", data.get("Statistics", {}))
        if not stats:
            continue

        fees_str = stats.get("Total Fees", "$0")
        core_seconds = float(algorithm_seconds if algorithm_seconds is not None else elapsed)
        result = write_result(
            engine="quantconnect",
            prefix="qc",
            strategy=strat_label,
            nav=nav,
            core_seconds=core_seconds,
            wall_seconds=elapsed,
            extra={
                "algorithm_seconds": algorithm_seconds,
                "wrapper_overhead_seconds": (
                    None
                    if algorithm_seconds is None
                    else round(max(0.0, elapsed - algorithm_seconds), 6)
                ),
                "total_fees": fees_str,
                "engine_mode": "canonical target quantities on daily bars",
                "share_model": "fractional-equivalent via linear capital scaling",
                "capital_scale": INTEGER_ENGINE_CAPITAL_SCALE,
                "execution_count": int(bundle["execution_flags"].sum()),
            },
        )
        for artifact_name in ("observed_prices.csv", "realised_weights.csv"):
            source = results_dir / artifact_name
            if source.exists():
                shutil.copyfile(source, RESULTS_DIR / f"qc_{strat_label}_{artifact_name}")
        print(
            f"  LEAN {strat_label}: wall={elapsed:.2f}s "
            f"algorithm={algorithm_seconds if algorithm_seconds is not None else 'n/a'}s, "
            f"NAV={result['final_nav']:,.2f} protocol={result['protocol_hash'][:12]}"
        )
        return result

    return {"engine": "quantconnect", "strategy": strat_label, "error": "parse failed"}


# ─── Main ────────────────────────────────────────────────────────


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Check Docker
    try:
        subprocess.run(["docker", "ps"], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        print("ERROR: Docker is not running. Start Docker Desktop first.")
        sys.exit(1)

    # Pull only when the image is absent. Re-checking the registry before every
    # one of the five strategies adds network time to the benchmark harness.
    print("Checking LEAN Docker image...")
    image_check = subprocess.run(
        ["docker", "image", "inspect", LEAN_IMAGE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if image_check.returncode != 0:
        subprocess.run(["docker", "pull", LEAN_IMAGE], check=True, timeout=600)
    else:
        print(f"LEAN image ready: {LEAN_IMAGE}")

    # Extract auxiliary data (market-hours, symbol-properties, etc.)
    print("\nExtracting auxiliary data from LEAN image...")
    extract_auxiliary_data()

    # Convert data to LEAN format
    print("\nConverting data to LEAN format...")
    convert_to_lean_format()

    # Run strategies
    strategies = [
        "strat_01_sma_crossover",
        "strat_02_rsi_reversion",
        "strat_03_monthly_rebalance",
        "strat_04_momentum_voltarget",
        "strat_05_dual_momentum",
    ]

    # Filter by CLI arg if provided
    if len(sys.argv) > 1:
        filter_name = sys.argv[1]
        strategies = [s for s in strategies if filter_name in s]

    results = []
    for strat in strategies:
        result = run_strategy(strat)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("  QuantConnect LEAN — Summary")
    print(f"{'=' * 60}")
    for r in results:
        if "error" in r:
            print(f"  {r.get('strategy', '?')}: ERROR - {r['error']}")
        else:
            print(
                f"  {r['strategy']}: NAV={r['final_nav']}, CAGR={r['cagr']}, "
                f"Sharpe={r['sharpe']}, Time={r['elapsed_seconds']}s"
            )


if __name__ == "__main__":
    main()
