#!/usr/bin/env bash
# Populate the PyMove reference cache used by cached comparison tests.
#
# Runs the populate script inside .venv-pymove. The resulting parquet files
# written to tests/shared/pymove_reference/ should be committed to git so
# the normal .venv can run comparison tests without pymove installed.
#
# First-time setup of .venv-pymove (needs Python <=3.10; pandas 1.3.5 has
# no prebuilt wheel past 3.10):
#   bash scripts/setup_benchmark_env.sh pymove --python 3.9
#
# Usage:
#   bash scripts/populate_pymove_cache.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv-pymove" ]; then
    echo "ERROR: .venv-pymove not found."
    echo "  Set it up first: bash scripts/setup_benchmark_env.sh pymove --python 3.9"
    exit 1
fi

echo "==> Populating PyMove reference cache ..."
.venv-pymove/bin/python tests/populate_pymove_cache.py "$@"
