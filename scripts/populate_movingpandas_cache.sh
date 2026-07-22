#!/usr/bin/env bash
# Populate the MovingPandas reference cache used by cached comparison tests.
#
# Runs the populate script inside .venv-movingpandas. The resulting parquet
# files written to tests/shared/movingpandas_reference/ should be committed
# to git so the normal .venv can run comparison tests without movingpandas
# installed.
#
# First-time setup of .venv-movingpandas:
#   bash scripts/setup_env.sh --venv .venv-movingpandas --movingpandas
#
# Usage:
#   bash scripts/populate_movingpandas_cache.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv-movingpandas" ]; then
    echo "ERROR: .venv-movingpandas not found."
    echo "  Set it up first: bash scripts/setup_env.sh --venv .venv-movingpandas --movingpandas"
    exit 1
fi

# Rebuild fastmob._core for the movingpandas comparison environment so the
# populate script can import fastmob alongside movingpandas (not required by
# this script today, but kept for parity with populate_skmob_cache.sh and
# future oracle scripts that call into fastmob directly).
echo "==> Rebuilding fastmob._core for .venv-movingpandas ..."
unset CONDA_PREFIX
VIRTUAL_ENV="$PWD/.venv-movingpandas" \
PATH="$PWD/.venv-movingpandas/bin:$PATH" \
    .venv/bin/maturin develop

echo ""
echo "==> Populating MovingPandas reference cache ..."
.venv-movingpandas/bin/python tests/populate_movingpandas_cache.py "$@"
