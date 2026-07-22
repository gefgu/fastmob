#!/usr/bin/env bash
# Populate the PTRAIL reference cache used by cached comparison tests.
#
# Runs the populate script inside .venv-ptrail. The resulting parquet files
# written to tests/shared/ptrail_reference/ should be committed to git so
# the normal .venv can run comparison tests without ptrail installed.
#
# First-time setup of .venv-ptrail:
#   bash scripts/setup_env.sh --venv .venv-ptrail --ptrail
#
# Usage:
#   bash scripts/populate_ptrail_cache.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv-ptrail" ]; then
    echo "ERROR: .venv-ptrail not found."
    echo "  Set it up first: bash scripts/setup_env.sh --venv .venv-ptrail --ptrail"
    exit 1
fi

# Rebuild fastmob._core for the ptrail comparison environment so the
# populate script can import fastmob alongside ptrail (not required by this
# script today, but kept for parity with populate_skmob_cache.sh and other
# oracle scripts that call into fastmob directly).
echo "==> Rebuilding fastmob._core for .venv-ptrail ..."
unset CONDA_PREFIX
VIRTUAL_ENV="$PWD/.venv-ptrail" \
PATH="$PWD/.venv-ptrail/bin:$PATH" \
    .venv/bin/maturin develop

echo ""
echo "==> Populating PTRAIL reference cache ..."
.venv-ptrail/bin/python tests/populate_ptrail_cache.py "$@"
