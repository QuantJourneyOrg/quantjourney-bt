#!/usr/bin/env python3
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.
"""Run full-package mypy and reject diagnostics not present in the reviewed baseline.

The baseline is a migration aid, not a suppression list: mypy still checks every
module in the backtester package. Existing diagnostics remain visible, and any
new, moved, or fixed diagnostic requires a reviewed baseline refresh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "quality" / "mypy-baseline.json"
BASELINE_SCHEMA = 2
TYPE_PACKAGES = ("mypy", "pandas-stubs", "types-python-dateutil")


def _context() -> dict[str, Any]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    config = project.get("tool", {}).get("mypy")
    if not isinstance(config, dict):
        raise RuntimeError("pyproject.toml has no [tool.mypy] configuration")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    lock_path = ROOT / "uv.lock"
    if not lock_path.is_file():
        raise RuntimeError("uv.lock is required for deterministic type checking")
    packages: dict[str, str] = {}
    for package in TYPE_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "<missing>"
    return {
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "packages": packages,
    }


def _source_span_hash(path: Path, diagnostic: dict[str, Any]) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        start_line = int(diagnostic.get("line", 0))
        end_line = int(diagnostic.get("end_line") or start_line)
        start_column = int(diagnostic.get("column", 0))
        end_column = int(diagnostic.get("end_column") or start_column)
        if start_line < 1 or end_line < start_line:
            raise ValueError
        selected = lines[start_line - 1 : end_line]
        if not selected:
            raise ValueError
        if len(selected) == 1:
            span = selected[0][start_column:end_column]
        else:
            selected[0] = selected[0][start_column:]
            selected[-1] = selected[-1][:end_column]
            span = "\n".join(selected)
    except (OSError, UnicodeDecodeError, ValueError):
        span = "<unavailable>"
    normalized = " ".join(span.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _fingerprint(diagnostic: dict[str, Any]) -> str:
    path = Path(str(diagnostic["file"]))
    if not path.is_absolute():
        path = ROOT / path
    absolute_path = path.resolve()
    try:
        display_path = absolute_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        display_path = absolute_path.as_posix()
    return json.dumps(
        [
            display_path,
            str(diagnostic.get("code") or "unknown"),
            str(diagnostic["message"]),
            _source_span_hash(absolute_path, diagnostic),
        ],
        separators=(",", ":"),
    )


def _run_mypy() -> list[str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(ROOT / "pyproject.toml"),
            "--no-incremental",
            "-O",
            "json",
            "backtester",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stderr.strip():
        raise RuntimeError(f"mypy wrote to stderr:\n{completed.stderr.strip()}")
    if completed.returncode not in {0, 1}:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"mypy failed to run (exit {completed.returncode})")

    fingerprints: list[str] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            diagnostic = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Unexpected mypy output: {raw_line}") from exc
        if diagnostic.get("severity") == "error":
            fingerprints.append(_fingerprint(diagnostic))
    return sorted(fingerprints)


def _write_baseline(path: Path, errors: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BASELINE_SCHEMA,
        "context": _context(),
        "error_count": len(errors),
        "errors": errors,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {path.relative_to(ROOT)} with {len(errors)} reviewed diagnostics.")


def _load_baseline(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing mypy baseline: {path}") from exc
    if payload.get("schema") != BASELINE_SCHEMA:
        raise RuntimeError(f"Unsupported mypy baseline schema in {path}")
    expected_context = payload.get("context")
    actual_context = _context()
    if expected_context != actual_context:
        raise RuntimeError(
            "type-check configuration or dependency versions differ from the baseline; "
            "review them and update the baseline together"
        )
    errors = payload.get("errors")
    if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
        raise RuntimeError(f"Invalid errors list in {path}")
    if payload.get("error_count") != len(errors):
        raise RuntimeError(f"Invalid error_count in {path}")
    return errors


def _check_baseline(path: Path, actual: list[str]) -> None:
    reviewed = _load_baseline(path)
    new_errors = list((Counter(actual) - Counter(reviewed)).elements())
    removed_errors = list((Counter(reviewed) - Counter(actual)).elements())
    if new_errors or removed_errors:
        changes: list[str] = []
        if new_errors:
            changes.append(
                f"{len(new_errors)} new diagnostic(s):\n  " + "\n  ".join(new_errors[:20])
            )
        if removed_errors:
            changes.append(
                f"{len(removed_errors)} resolved or moved diagnostic(s):\n  "
                + "\n  ".join(removed_errors[:20])
            )
        raise RuntimeError(
            "\n".join(changes)
            + "\nReview the complete mypy output, then explicitly update the baseline."
        )
    print(f"Full-package mypy gate passed: {len(actual)} reviewed diagnostic(s), 0 changes.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Replace the baseline after reviewing the complete mypy output.",
    )
    args = parser.parse_args()

    try:
        errors = _run_mypy()
        if args.update:
            _write_baseline(args.baseline, errors)
        else:
            _check_baseline(args.baseline, errors)
    except RuntimeError as exc:
        print(f"Type-check gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
