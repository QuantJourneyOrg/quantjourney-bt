#!/usr/bin/env bash
# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PYTHON="$SCRIPT_DIR/.venv/bin/python"
PARENT_PYTHON="$SCRIPT_DIR/../.venv/bin/python"
SYSTEM_PYTHON="$(command -v python3 || command -v python || true)"

if [[ -x "$LOCAL_PYTHON" ]]; then
    PYTHON="$LOCAL_PYTHON"
elif [[ -x "$PARENT_PYTHON" ]]; then
    PYTHON="$PARENT_PYTHON"
elif [[ -n "$SYSTEM_PYTHON" ]]; then
    PYTHON="$SYSTEM_PYTHON"
else
    echo "Error: Python was not found." >&2
    echo "Run: python3 -m venv .venv" >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/strategy.py" "$@"
