#!/usr/bin/env bash
# Populate the skmob reference cache used by cached comparison tests.
#
# Runs the populate script inside .venv-skmob (Python 3.10 + scikit-mobility 1.3.1).
# The resulting parquet/JSON files written to tests/shared/skmob_reference/ should be
# committed to git so the normal .venv can run comparison tests without skmob installed.
#
# Usage:
#   bash scripts/populate_skmob_cache.sh
#   bash scripts/populate_skmob_cache.sh --datasets brightkite
#   bash scripts/populate_skmob_cache.sh --datasets geolife,foursquare
#   bash scripts/populate_skmob_cache.sh --geolife-rows 5000

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv-skmob" ]; then
    echo "ERROR: .venv-skmob not found."
    echo "  Set up the skmob comparison environment first (see CLAUDE.md)."
    exit 1
fi

# Rebuild skmob2._core for the skmob comparison environment so the populate
# script can import skmob2 alongside skmob.
echo "==> Rebuilding skmob2._core for .venv-skmob ..."
unset CONDA_PREFIX
VIRTUAL_ENV="$PWD/.venv-skmob" \
PATH="$PWD/.venv-skmob/bin:$PATH" \
    .venv/bin/maturin develop

echo ""
echo "==> Populating skmob reference cache ..."
.venv-skmob/bin/python tests/populate_skmob_cache.py "$@"
