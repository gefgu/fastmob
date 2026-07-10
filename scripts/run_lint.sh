#!/usr/bin/env bash
# Run Ruff and Clippy lint checks.
#
# Usage:
#   bash scripts/run_lint.sh           # check only
#   bash scripts/run_lint.sh --fix     # apply Ruff fixes/formatting and Clippy fixes
#
# Any extra arguments are forwarded directly to `ruff check`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash scripts/setup_env.sh' first."
    exit 1
fi

source .venv/bin/activate

FIX=0
CHECK_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--fix" ]; then
        FIX=1
    else
        CHECK_ARGS+=("$arg")
    fi
done

if [ "$FIX" -eq 1 ]; then
    ruff check --fix fkmob scripts tests "${CHECK_ARGS[@]}"
    ruff format fkmob scripts tests
    cargo clippy --fix --all-targets --all-features --allow-dirty -- -D warnings
else
    ruff check fkmob scripts tests "${CHECK_ARGS[@]}"
    ruff format --check fkmob scripts tests
    cargo clippy --all-targets --all-features -- -D warnings
fi
