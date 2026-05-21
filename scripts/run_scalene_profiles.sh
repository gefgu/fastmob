#!/usr/bin/env bash
# Run Brightkite Scalene profiling jobs.
# JSON profiles, HTML reports, and manifests are written to .profiles/scalene/ by default.
#
# Usage:
#   bash scripts/run_scalene_profiles.sh --list
#   bash scripts/run_scalene_profiles.sh --dry-run --rows 10000 --workload jump_lengths --implementation both
#   bash scripts/run_scalene_profiles.sh --rows 10000 --workload jump_lengths --implementation both
#   bash scripts/run_scalene_profiles.sh --rows 10000 --workload jump_lengths --implementation skmob2 --jump-lengths-entrypoint function
#   bash scripts/run_scalene_profiles.sh --rows 10000 --workload radius_of_gyration --implementation skmob2
#   bash scripts/run_scalene_profiles.sh --rows 10000 --workload radius_of_gyration --implementation skmob2 --backend polars
#   bash scripts/run_scalene_profiles.sh                              # jump_lengths skmob2 vs skmob on full 4M-row sweep
#
# Any extra arguments are forwarded directly to scripts/profile_brightkite_scalene.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash scripts/setup_env.sh' first."
    exit 1
fi

source .venv/bin/activate
unset CONDA_PREFIX
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.profiles/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

SCALENE_BIN="$REPO_ROOT/.venv/bin/scalene"

SKIP_BUILD=0
REQUESTED_IMPLEMENTATION="both"
EXPECT_IMPLEMENTATION_VALUE=0
for arg in "$@"; do
    if [ "$EXPECT_IMPLEMENTATION_VALUE" -eq 1 ]; then
        REQUESTED_IMPLEMENTATION="$arg"
        EXPECT_IMPLEMENTATION_VALUE=0
        continue
    fi
    if [ "$arg" = "--list" ] || [ "$arg" = "--dry-run" ]; then
        SKIP_BUILD=1
    elif [ "$arg" = "--implementation" ]; then
        EXPECT_IMPLEMENTATION_VALUE=1
    elif [[ "$arg" == --implementation=* ]]; then
        REQUESTED_IMPLEMENTATION="${arg#--implementation=}"
    fi
done

if [ "$SKIP_BUILD" -eq 0 ]; then
    if [ ! -x "$SCALENE_BIN" ]; then
        echo "==> Installing Scalene into .venv ..."
        uv pip install scalene
    fi
    "$SCALENE_BIN" --help >/dev/null

    if [ "$REQUESTED_IMPLEMENTATION" = "both" ] || [ "$REQUESTED_IMPLEMENTATION" = "skmob" ]; then
        SKMOB_IMPORT_CHECK="import shapely.ops as ops
if not hasattr(ops, 'cascaded_union') and hasattr(ops, 'unary_union'):
    ops.cascaded_union = ops.unary_union
import skmob.measures.individual"
        if ! python -c "$SKMOB_IMPORT_CHECK" >/dev/null 2>&1; then
            echo "==> Installing scikit-mobility comparison extra into .venv ..."
            uv pip install -e ".[dev-skmob]"
        fi
        python -c "$SKMOB_IMPORT_CHECK" >/dev/null 2>&1 || {
            echo "ERROR: scikit-mobility is required for --implementation $REQUESTED_IMPLEMENTATION."
            echo "       Install it with: uv pip install -e '.[dev-skmob]'"
            exit 1
        }
    fi

    # Use release speed for the Rust extension while profiling Python-facing workload behavior.
    export CARGO_PROFILE_RELEASE_DEBUG=1
    if ! maturin develop --release; then
        maturin develop --release --uv
    fi
fi

python scripts/profile_brightkite_scalene.py --scalene-bin "$SCALENE_BIN" "$@"
