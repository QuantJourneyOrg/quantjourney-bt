# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Cross-platform launcher for repository strategy examples."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parent
STRATEGIES_DIR = ROOT / "strategies"
REQUIRED_MODULES = ("pandas", "httpx", "requests", "quantjourney_ti")


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without executing shell code."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def configure_process_environment() -> None:
    load_env_file(ROOT / ".env")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    os.environ.setdefault("QJ_REPLACE_EXISTING_SESSION", "1")
    current = os.environ.get("PYTHONPATH", "")
    paths = [str(ROOT), *(part for part in current.split(os.pathsep) if part)]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))


def strategy_files(*, examples_only: bool = False) -> list[Path]:
    pattern = "example_*.py" if examples_only else "*.py"
    return sorted(path for path in STRATEGIES_DIR.glob(pattern) if path.name != "__init__.py")


def strategy_description(path: Path) -> str:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(module, clean=True) or ""
    except (OSError, SyntaxError, UnicodeError):
        return ""
    return next((line.strip() for line in docstring.splitlines() if line.strip()), "")


def list_strategies() -> int:
    print("Available strategies:")
    for path in strategy_files():
        description = strategy_description(path)
        suffix = f"  {description}" if description else ""
        print(f"  {path.stem}{suffix}")
    return 0


def runtime_is_ready() -> bool:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return True

    print("Error: Python runtime is missing required dependencies.", file=sys.stderr)
    print(f"Missing: {', '.join(missing)}", file=sys.stderr)
    if os.name == "nt":
        print("Run:", file=sys.stderr)
        print("  py -3.11 -m venv .venv", file=sys.stderr)
        print(r"  .venv\Scripts\python.exe -m pip install -U pip", file=sys.stderr)
        print(r'  .venv\Scripts\python.exe -m pip install -e ".[dev,data]"', file=sys.stderr)
    else:
        print("Run:", file=sys.stderr)
        print("  python3 -m venv .venv", file=sys.stderr)
        print("  source .venv/bin/activate", file=sys.stderr)
        print('  python -m pip install -e ".[dev,data]"', file=sys.stderr)
    return False


def strategy_path(name: str) -> Path | None:
    candidate = STRATEGIES_DIR / f"{name}.py"
    if candidate.is_file() and candidate.parent == STRATEGIES_DIR:
        return candidate
    return None


def check_strategy(path: Path) -> int:
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load strategy module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"Import check failed: {path.name}: {exc}", file=sys.stderr)
        return 1
    print(f"Import check passed: {path.name}")
    return 0


def credentials_are_ready(*, sample_data: bool, check_only: bool) -> bool:
    if sample_data or check_only:
        return True
    if os.environ.get("QJ_API_KEY"):
        return True
    if os.environ.get("QJ_EMAIL") and os.environ.get("QJ_PASSWORD"):
        return True
    print("Error: No credentials set. Add QJ_API_KEY to .env or the environment.", file=sys.stderr)
    print("For a credential-free demo, add --sample-data.", file=sys.stderr)
    return False


def apply_run_options(args: argparse.Namespace) -> None:
    if args.sample_data:
        os.environ["QJ_SAMPLE_DATA"] = "1"
    if args.quiet:
        os.environ.setdefault("QJ_QUIET", "1")
        os.environ.setdefault("QJ_LOG_LEVEL", "ERROR")
    if args.no_reports:
        os.environ["QJ_NO_REPORTS"] = "1"
    if args.output:
        os.environ["QJ_OUTPUT_DIR"] = args.output
    if args.debug:
        os.environ["QJ_LOG_LEVEL"] = "DEBUG"
    if args.log_level:
        os.environ["QJ_LOG_LEVEL"] = args.log_level
    if args.theme:
        os.environ["QJ_PLOT_THEME"] = args.theme
    if args.dpi:
        os.environ["QJ_PLOT_DPI"] = str(args.dpi)


def backtester_version() -> str:
    try:
        from backtester.version import __version__

        return __version__
    except Exception:
        return "unknown"


def run_strategy(
    path: Path,
    args: argparse.Namespace,
    *,
    output: TextIO | None = None,
) -> int:
    apply_run_options(args)
    if not credentials_are_ready(sample_data=args.sample_data, check_only=args.check):
        return 1
    if args.check:
        if output is None:
            return check_strategy(path)
        with redirect_stdout(output), redirect_stderr(output):
            return check_strategy(path)

    destination = output or sys.stdout
    print(f"Running strategy: {path.stem}", file=destination)
    print(f"File: {path}", file=destination)
    print(f"Python: {sys.executable}", file=destination)
    print(f"Backtester: v{backtester_version()}", file=destination)
    print(f"Theme: {os.environ.get('QJ_PLOT_THEME', 'strategy default')}", file=destination)
    if args.sample_data:
        print("Data: deterministic sample data", file=destination)
    print(f"Plot DPI: {os.environ.get('QJ_PLOT_DPI', '300')}", file=destination)
    print(f"Log Level: {os.environ.get('QJ_LOG_LEVEL', 'INFO')}", file=destination)
    print(f"Output: {os.environ.get('QJ_OUTPUT_DIR', './reports')}", file=destination)
    if args.no_reports:
        print("Reports: disabled", file=destination)
    print("", file=destination)
    destination.flush()

    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=output,
        stderr=subprocess.STDOUT if output is not None else None,
        check=False,
    )
    return completed.returncode


def run_all(args: argparse.Namespace) -> int:
    if not credentials_are_ready(sample_data=args.sample_data, check_only=args.check):
        return 1

    output_root = Path(args.output or os.environ.get("QJ_OUTPUT_DIR", "./reports"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = output_root / "_batch" / batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary = batch_dir / "summary.tsv"

    files = strategy_files(examples_only=True)
    passed = 0
    failed = 0
    with summary.open("w", encoding="utf-8", newline="") as summary_file:
        summary_file.write("strategy\tstatus\texit_code\tduration_seconds\tlog\n")
        print(f"Running {len(files)} example strategies sequentially")
        print(f"Output root: {output_root}")
        print(f"Batch logs: {batch_dir}\n")

        for index, path in enumerate(files, start=1):
            log_path = logs_dir / f"{path.stem}.log"
            started = time.monotonic()
            print(f"[{index}/{len(files)}] START {path.stem}")
            with log_path.open("w", encoding="utf-8") as log_file:
                exit_code = run_strategy(path, args, output=log_file)
            duration = int(time.monotonic() - started)
            status = "passed" if exit_code == 0 else "failed"
            if exit_code == 0:
                passed += 1
                print(f"[{index}/{len(files)}] PASS  {path.stem}")
            else:
                failed += 1
                print(f"[{index}/{len(files)}] FAIL  {path.stem} (exit {exit_code}; {log_path})")
            summary_file.write(f"{path.stem}\t{status}\t{exit_code}\t{duration}\t{log_path}\n")

    print(f"\nBatch complete: total={len(files)} passed={passed} failed={failed}")
    print(f"Summary: {summary}")
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a QuantJourney Backtester strategy from this repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  strategy.py example_weights_01_sma_daily --sample-data
  strategy.py --all --output ./reports
  strategy.py --all --check""",
    )
    parser.add_argument("strategy", nargs="?", help="strategy filename without .py")
    parser.add_argument("--all", action="store_true", help="run every example_*.py strategy")
    parser.add_argument("-l", "--list", action="store_true", help="list available strategies")
    parser.add_argument("--check", action="store_true", help="import without running a backtest")
    parser.add_argument("--sample-data", action="store_true", help="use bundled sample data")
    parser.add_argument("-q", "--quiet", action="store_true", help="show only the final summary")
    parser.add_argument("--no-reports", action="store_true", help="skip report generation")
    parser.add_argument("-o", "--output", help="output root for reports")
    parser.add_argument("--debug", action="store_true", help="set QJ_LOG_LEVEL=DEBUG")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--theme", help="plot theme name")
    parser.add_argument("--dpi", type=int, help="plot DPI")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_process_environment()
    parser = build_parser()
    args = parser.parse_args(argv)

    selected_modes = int(bool(args.strategy)) + int(args.all) + int(args.list)
    if selected_modes != 1:
        parser.error("choose exactly one strategy name, --all, or --list")
    if args.list:
        return list_strategies()
    if not runtime_is_ready():
        return 1
    if args.all:
        return run_all(args)

    path = strategy_path(args.strategy)
    if path is None:
        print(
            f"Error: Strategy '{args.strategy}' was not found in {STRATEGIES_DIR}.", file=sys.stderr
        )
        list_strategies()
        return 1
    return run_strategy(path, args)


if __name__ == "__main__":
    raise SystemExit(main())
