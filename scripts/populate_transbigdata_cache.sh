#!/usr/bin/env bash
# Populate the TransBigData reference cache used by cached comparison tests.
#
# Runs the populate script inside .venv-transbigdata. Installing transbigdata
# (via its transitive pykalman/scikit-base dependencies) into an
# already-populated .venv can trigger a full dependency re-resolution that
# silently downgrades unrelated packages (numpy in particular) -- a
# dedicated venv, resolved from scratch, avoids that risk entirely. The
# resulting parquet files written to tests/shared/transbigdata_reference/
# should be committed to git so the normal .venv can run comparison tests
# without transbigdata installed.
#
# First-time setup of .venv-transbigdata:
#   bash scripts/setup_env.sh --venv .venv-transbigdata --transbigdata
#
# Usage:
#   bash scripts/populate_transbigdata_cache.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv-transbigdata" ]; then
    echo "ERROR: .venv-transbigdata not found."
    echo "  Set it up first: bash scripts/setup_env.sh --venv .venv-transbigdata --transbigdata"
    exit 1
fi

echo "==> Populating TransBigData reference cache ..."
.venv-transbigdata/bin/python tests/populate_transbigdata_cache.py "$@"
