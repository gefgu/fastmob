#!/usr/bin/env bash
# Run Brightkite py-spy profiling jobs.
# Flame graphs, Speedscope JSON files, and manifests are written to .profiles/py-spy/ by default.
#
# Usage:
#   bash scripts/run_py_spy_profiles.sh --list
#   bash scripts/run_py_spy_profiles.sh --dry-run --rows 10000 --workload radius_of_gyration
#   bash scripts/run_py_spy_profiles.sh --rows 10000 --workload radius_of_gyration
#   bash scripts/run_py_spy_profiles.sh --rows 10000 --workload radius_of_gyration --implementation both
#   bash scripts/run_py_spy_profiles.sh --rows 10000 --workload radius_of_gyration --scope full
#   bash scripts/run_py_spy_profiles.sh                              # full 4M-row sweep
#
# Any extra arguments are forwarded directly to scripts/profile_brightkite_py_spy.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash scripts/setup_env.sh' first."
    exit 1
fi

source .venv/bin/activate
unset CONDA_PREFIX

PY_SPY_BIN="$REPO_ROOT/.venv/bin/py-spy"

SKIP_BUILD=0
for arg in "$@"; do
    if [ "$arg" = "--list" ] || [ "$arg" = "--dry-run" ]; then
        SKIP_BUILD=1
        break
    fi
done

if [ "$SKIP_BUILD" -eq 0 ]; then
    if [ ! -x "$PY_SPY_BIN" ]; then
        echo "==> Installing py-spy into .venv ..."
        uv pip install py-spy
    fi
    "$PY_SPY_BIN" --version

    # Use release speed while keeping Rust debug symbols available to py-spy --native.
    export CARGO_PROFILE_RELEASE_DEBUG=1
    if ! maturin develop --release; then
        maturin develop --release --uv
    fi
fi

python scripts/profile_brightkite_py_spy.py --py-spy-bin "$PY_SPY_BIN" "$@"
