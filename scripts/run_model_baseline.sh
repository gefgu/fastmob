#!/usr/bin/env bash
# Generate statistical baselines for model generation comparison.
#
# Runs the baseline generator in both .venv (skmob2) and .venv-skmob (skmob).
# The resulting JSON files are committed to tests/shared/ and used by
# tests/correctness/models/test_statistical_model_parity.py.
#
# Usage:
#   bash scripts/run_model_baseline.sh [--n-runs N] [--n-agents N]
#
# Options passed through to the Python script:
#   --n-runs   number of random seeds (default 100)
#   --n-agents number of agents per run (default 2, matches existing test fixtures)

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO/benchmarks/model_statistical_baseline.py"

# Forward any extra arguments (e.g. --n-runs 50) to both invocations.
EXTRA_ARGS=("$@")

echo "=== skmob2 baseline (.venv) ==="
source "$REPO/.venv/bin/activate"
python "$SCRIPT" --library skmob2 "${EXTRA_ARGS[@]}"

echo ""
echo "=== skmob baseline (.venv-skmob) ==="
deactivate || true
unset CONDA_PREFIX || true
"$REPO/.venv-skmob/bin/python" "$SCRIPT" --library skmob "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Baseline files written to tests/shared/."
echo "Commit both JSON files to git after reviewing the values."
