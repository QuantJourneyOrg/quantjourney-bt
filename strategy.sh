#!/usr/bin/env bash
#
# strategy.sh — run a qj-backtester strategy by name
#
# Usage:
#   ./strategy.sh example_weights_01_sma_daily
#   ./strategy.sh example_weights_01_sma_daily --sample-data
#   ./strategy.sh example_weights_01_sma_daily --quiet
#   ./strategy.sh example_weights_01_sma_daily --no-reports
#   ./strategy.sh example_weights_01_sma_daily --output /tmp/qj-reports
#   ./strategy.sh example_weights_01_sma_daily --check
#   ./strategy.sh --list
#
# Looks for strategies/<name>.py and runs it with local .venv or python3.
# Set QJ_API_KEY or QJ_EMAIL + QJ_PASSWORD before running real-data backtests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STRATEGIES_DIR="$SCRIPT_DIR/strategies"
LOCAL_VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
SYSTEM_PYTHON="$(command -v python3 || command -v python || true)"

# Python 3.14 + quantjourney_ti/numba can fail during import when Numba tries
# to build on-disk caches from the installed package. Disable JIT by default
# for strategy runs; users can override this from the shell if needed.
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

python_has_runtime_deps() {
    local py="$1"
    [ -x "$py" ] || return 1
    "$py" -c 'import importlib.util; import pandas, httpx, requests; raise SystemExit(0 if importlib.util.find_spec("quantjourney_ti") else 1)' >/dev/null 2>&1
}

if python_has_runtime_deps "$LOCAL_VENV_PYTHON"; then
    VENV_PYTHON="$LOCAL_VENV_PYTHON"
elif [ -n "$SYSTEM_PYTHON" ] && python_has_runtime_deps "$SYSTEM_PYTHON"; then
    VENV_PYTHON="$SYSTEM_PYTHON"
else
    VENV_PYTHON="$LOCAL_VENV_PYTHON"
fi

# ── colours ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

usage() {
    local status="${1:-1}"
    echo -e "${CYAN}Usage:${NC} $0 <strategy_name> [options] | --list"
    echo ""
    echo "  Run a strategy:        $0 example_weights_01_sma_daily"
    echo "  Demo without API key:  $0 example_weights_01_sma_daily --sample-data"
    echo "  Import check only:     $0 example_weights_01_sma_daily --check"
    echo "  Quiet final summary:   $0 example_weights_01_sma_daily --quiet"
    echo "  No reports/plots:      $0 example_weights_01_sma_daily --no-reports"
    echo "  Custom output dir:     $0 example_weights_01_sma_daily --output /tmp/qj-reports"
    echo "  List strategies:       $0 --list"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "      --check            Import the strategy module without running a backtest"
    echo "      --sample-data      Run against deterministic bundled sample data; no API key required"
    echo "  -q, --quiet            Hide INFO output and text report; keep final summary"
    echo "      --no-reports       Skip text report and plots; keep run metadata"
    echo "  -o, --output DIR       Save reports and metadata under DIR/<strategy_name>/"
    echo "      --debug            Set QJ_LOG_LEVEL=DEBUG"
    echo "      --log-level LEVEL  Set Python logger level: DEBUG, INFO, WARNING, ERROR"
    echo "      --dpi N            Set QJ_PLOT_DPI for this run"
    echo ""
    echo -e "${YELLOW}Environment variables:${NC}"
    echo "  QJ_API_KEY       API key (preferred)"
    echo "  QJ_EMAIL         Email login (fallback)"
    echo "  QJ_PASSWORD      Password (fallback)"
    echo "  QJ_REPLACE_EXISTING_SESSION  1 replaces active auth session on 409 conflict (default: 1)"
    echo "  QJ_LOG_LEVEL     Python logger level"
    echo "  QJ_QUIET         1 disables text report and plot display"
    echo "  QJ_NO_REPORTS    1 skips report/plot generation"
    echo "  QJ_OUTPUT_DIR    Output root for reports and run metadata"
    exit "$status"
}

list_strategies() {
    echo -e "${CYAN}Available strategies:${NC}"
    for f in "$STRATEGIES_DIR"/*.py; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .py)
        # extract first non-empty docstring line as description
        desc=$(sed -n '2,5{/^[[:space:]]*$/d; s/^[[:space:]]*//; p; q;}' "$f")
        echo -e "  ${GREEN}${name}${NC}  ${desc}"
    done
}

# ── args ─────────────────────────────────────────────────────────────
[ $# -lt 1 ] && usage 1

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage 0
fi

if [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
    list_strategies
    exit 0
fi

STRATEGY_NAME="$1"
shift
CHECK_ONLY=false
SAMPLE_DATA=false

while [ $# -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --sample-data)
            SAMPLE_DATA=true
            export QJ_SAMPLE_DATA="1"
            shift
            ;;
        -q|--quiet)
            export QJ_QUIET="${QJ_QUIET:-1}"
            export QJ_LOG_LEVEL="${QJ_LOG_LEVEL:-ERROR}"
            shift
            ;;
        --no-reports)
            export QJ_NO_REPORTS="1"
            shift
            ;;
        -o|--output)
            [ $# -lt 2 ] && usage 1
            if [ -z "$2" ]; then
                echo -e "${RED}Error:${NC} --output requires a non-empty directory"
                usage 1
            fi
            export QJ_OUTPUT_DIR="$2"
            shift 2
            ;;
        --output=*)
            output_dir="${1#*=}"
            if [ -z "$output_dir" ]; then
                echo -e "${RED}Error:${NC} --output requires a non-empty directory"
                usage 1
            fi
            export QJ_OUTPUT_DIR="$output_dir"
            shift
            ;;
        --debug)
            export QJ_LOG_LEVEL="DEBUG"
            shift
            ;;
        --log-level)
            [ $# -lt 2 ] && usage 1
            export QJ_LOG_LEVEL="$2"
            shift 2
            ;;
        --dpi)
            [ $# -lt 2 ] && usage 1
            export QJ_PLOT_DPI="$2"
            shift 2
            ;;
        --list|-l)
            list_strategies
            exit 0
            ;;
        --help|-h)
            usage 0
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown option: $1"
            usage 1
            ;;
    esac
done

STRATEGY_FILE="$STRATEGIES_DIR/${STRATEGY_NAME}.py"

# ── validate ─────────────────────────────────────────────────────────
if [ ! -f "$STRATEGY_FILE" ]; then
    echo -e "${RED}Error:${NC} Strategy '${STRATEGY_NAME}' not found at ${STRATEGY_FILE}"
    echo ""
    list_strategies
    exit 1
fi

if ! python_has_runtime_deps "$VENV_PYTHON"; then
    echo -e "${RED}Error:${NC} Python runtime with required dependencies not found."
    echo -e "Checked:"
    echo -e "  ${LOCAL_VENV_PYTHON}"
    echo -e "  python3 on PATH"
    echo -e "Run:"
    echo -e "  python3 -m venv .venv"
    echo -e "  source .venv/bin/activate"
    echo -e "  python -m pip install -U pip"
    echo -e "  python -m pip install -e ."
    exit 1
fi

if [ "$CHECK_ONLY" = true ]; then
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
    "$VENV_PYTHON" - "$STRATEGY_FILE" <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(path.stem, path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot load strategy module: {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"Import check passed: {path.name}")
PY
    exit 0
fi

# ── check credentials ────────────────────────────────────────────────
if [ "$SAMPLE_DATA" != true ] && [ -z "${QJ_API_KEY:-}" ] && { [ -z "${QJ_EMAIL:-}" ] || [ -z "${QJ_PASSWORD:-}" ]; }; then
    echo -e "${RED}Error:${NC} No credentials set. Export QJ_API_KEY or QJ_EMAIL + QJ_PASSWORD."
    echo -e "For a credential-free demo, run: ${GREEN}$0 ${STRATEGY_NAME} --sample-data${NC}"
    exit 1
fi

BACKTESTER_VERSION="$(PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" "$VENV_PYTHON" -c 'from backtester.version import __version__; print(__version__)' 2>/dev/null || true)"
[ -n "$BACKTESTER_VERSION" ] || BACKTESTER_VERSION="unknown"

# ── run ──────────────────────────────────────────────────────────────
echo -e "${GREEN}Running strategy:${NC} ${STRATEGY_NAME}"
echo -e "${CYAN}File:${NC} ${STRATEGY_FILE}"
echo -e "${CYAN}Python:${NC} ${VENV_PYTHON}"
echo -e "${CYAN}Backtester:${NC} v${BACKTESTER_VERSION}"
echo -e "${CYAN}Theme:${NC} quantjourney"
if [ "$SAMPLE_DATA" = true ]; then
    echo -e "${CYAN}Data:${NC} deterministic sample data"
fi
echo -e "${CYAN}Plot DPI:${NC} ${QJ_PLOT_DPI:-300}"
echo -e "${CYAN}Log Level:${NC} ${QJ_LOG_LEVEL:-INFO}"
echo -e "${CYAN}Output:${NC} ${QJ_OUTPUT_DIR:-./reports}"
if [ "${QJ_NO_REPORTS:-0}" = "1" ]; then
    echo -e "${CYAN}Reports:${NC} disabled"
fi
echo ""

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
exec "$VENV_PYTHON" "$STRATEGY_FILE"
