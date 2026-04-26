#!/usr/bin/env bash
# Run Brightkite pytest-memray profiling jobs.
# Binary dumps, flame graphs, and manifests are written to .profiles/memray/ by default.
#
# Usage:
#   bash tests/run_memray_profiles.sh --list
#   bash tests/run_memray_profiles.sh --dry-run --rows 10000 --workload radius_of_gyration
#   bash tests/run_memray_profiles.sh --rows 10000 --workload radius_of_gyration
#   bash tests/run_memray_profiles.sh                              # full 4M-row sweep
#
# Any extra arguments are forwarded directly to scripts/profile_brightkite_memray.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash tests/setup_env.sh' first."
    exit 1
fi

source .venv/bin/activate
unset CONDA_PREFIX

PYTEST_BIN="$REPO_ROOT/.venv/bin/pytest"
MEMRAY_BIN="$REPO_ROOT/.venv/bin/memray"

SKIP_BUILD=0
for arg in "$@"; do
    if [ "$arg" = "--list" ] || [ "$arg" = "--dry-run" ]; then
        SKIP_BUILD=1
        break
    fi
done

if [ "$SKIP_BUILD" -eq 0 ]; then
    if [ ! -x "$MEMRAY_BIN" ] || ! python -c "import pytest_memray" >/dev/null 2>&1; then
        echo "==> Installing memray and pytest-memray into .venv ..."
        uv pip install memray pytest-memray
    fi
    "$MEMRAY_BIN" --version

    # Use release speed while keeping Rust debug symbols available to pytest-memray --native.
    export CARGO_PROFILE_RELEASE_DEBUG=1
    if ! maturin develop --release; then
        maturin develop --release --uv
    fi
fi

python scripts/profile_brightkite_memray.py \
    --pytest-bin "$PYTEST_BIN" \
    --memray-bin "$MEMRAY_BIN" \
    "$@"
