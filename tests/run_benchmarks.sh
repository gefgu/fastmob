#!/usr/bin/env bash
# Run standalone perf-counter benchmark suites.
#
# Usage:
#   bash tests/run_benchmarks.sh
#   bash tests/run_benchmarks.sh --sizes 1000 --iterations 1 --sleep 0
#
# Any extra arguments are forwarded to each standalone speed suite.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_VENV="$REPO_ROOT/.venv"
SKMOB_VENV="$REPO_ROOT/.venv-skmob"
MOVINGPANDAS_VENV="${MOVINGPANDAS_VENV:-$MAIN_VENV}"

if [ ! -f "$MAIN_VENV/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash tests/setup_env.sh' first."
    exit 1
fi

run_python() {
    local venv="$1"
    shift
    env -u CONDA_PREFIX \
        VIRTUAL_ENV="$venv" \
        PATH="$venv/bin:$PATH" \
        "$venv/bin/python" "$@"
}

can_import() {
    local venv="$1"
    local module="$2"
    run_python "$venv" -c "import ${module}" >/dev/null 2>&1
}

echo "==> Building skmob2._core in .venv ..."
run_python "$MAIN_VENV" -m maturin develop

echo "==> Running skmob2 pandas/Polars benchmark suites in .venv ..."
run_python "$MAIN_VENV" tests/benchmarks/speed_spatial_suite.py --library skmob2 --backend both "$@"
run_python "$MAIN_VENV" tests/benchmarks/speed_visits_suite.py --library skmob2 --backend both "$@"

if [ -x "$SKMOB_VENV/bin/python" ]; then
    echo "==> Running original skmob benchmark suites in .venv-skmob ..."
    run_python "$SKMOB_VENV" tests/benchmarks/speed_spatial_suite.py --library skmob "$@"
    run_python "$SKMOB_VENV" tests/benchmarks/speed_visits_suite.py --library skmob "$@"
else
    echo "WARNING: .venv-skmob not found; skipping skmob comparison benchmarks."
    echo "         Create it with the legacy skmob stack before running skmob comparisons."
fi

if [ -x "$MOVINGPANDAS_VENV/bin/python" ] && can_import "$MOVINGPANDAS_VENV" movingpandas; then
    label="${MOVINGPANDAS_VENV#$REPO_ROOT/}"
    echo "==> Running MovingPandas spatial benchmark suite in ${label} ..."
    run_python "$MOVINGPANDAS_VENV" tests/benchmarks/speed_spatial_suite.py --library movingpandas "$@"
else
    echo "WARNING: movingpandas is not importable in ${MOVINGPANDAS_VENV#$REPO_ROOT/}; skipping MovingPandas benchmarks."
    echo "         Install it with: source .venv/bin/activate && uv pip install -e '.[dev-movingpandas]'"
fi
