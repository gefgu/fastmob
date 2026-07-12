#!/usr/bin/env bash
# Benchmark the current fastmob MarkovDiaryGenerator.
#
# Usage:
#   bash scripts/run_markov_diary_benchmark.sh
#   bash scripts/run_markov_diary_benchmark.sh --n-agents 100 500 1000 --diary-length 168 --iterations 3
#
# Extra arguments are forwarded to benchmarks/markov_diary_speed_suite.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${FKMOB_BENCH_VENV:-$REPO_ROOT/.venv-py312}"
if [ ! -x "$VENV/bin/python" ]; then
    VENV="$REPO_ROOT/.venv"
fi
if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: no benchmark Python found. Expected .venv-py312 or .venv."
    echo "       Run 'bash scripts/setup_env.sh --python 3.12 --venv .venv-py312' first."
    exit 1
fi

env -u CONDA_PREFIX \
    VIRTUAL_ENV="$VENV" \
    PATH="$VENV/bin:$PATH" \
    "$VENV/bin/python" \
    benchmarks/markov_diary_speed_suite.py \
    --n-agents 500 \
    "$@"
