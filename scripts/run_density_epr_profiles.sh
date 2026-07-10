#!/usr/bin/env bash
# Profile the DensityEPR benchmark workload with Scalene and Samply.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N_AGENTS="${N_AGENTS:-500}"
N_LOCATIONS="${N_LOCATIONS:-10000}"
DAYS="${DAYS:-7}"
RATE="${RATE:-1000}"
RANDOM_STATE="${RANDOM_STATE:-2}"
SCALENE_SCOPE="${SCALENE_SCOPE:-fkmob}"
WORKLOAD="benchmarks/profile_density_epr_workload.py"

if [ "$DAYS" = "7" ]; then
    DEFAULT_PROFILE_NAME="density_epr_${N_AGENTS}a_${N_LOCATIONS}l_1w"
else
    DEFAULT_PROFILE_NAME="density_epr_${N_AGENTS}a_${N_LOCATIONS}l_${DAYS}d"
fi
PROFILE_NAME="${PROFILE_NAME:-$DEFAULT_PROFILE_NAME}"

SCALENE_DIR=".profiles/scalene"
SAMPLY_DIR=".profiles/samply"
SCALENE_JSON="$SCALENE_DIR/$PROFILE_NAME.json"
SCALENE_HTML="$SCALENE_DIR/$PROFILE_NAME.html"
SAMPLY_JSON="$SAMPLY_DIR/$PROFILE_NAME.json.gz"

mkdir -p "$SCALENE_DIR" "$SAMPLY_DIR"

if [ ! -f "$WORKLOAD" ]; then
    echo "ERROR: workload script not found: $WORKLOAD" >&2
    exit 1
fi

if [ ! -f "tests/shared/skmob_reference/models/input.parquet" ]; then
    echo "ERROR: model benchmark input is missing." >&2
    echo "Run: bash scripts/populate_skmob_cache.sh --datasets models" >&2
    exit 1
fi

echo "==> Building fkmob release extension with Rust debug symbols and frame pointers"
RUSTFLAGS="${RUSTFLAGS:--C force-frame-pointers=yes}" \
CARGO_PROFILE_RELEASE_DEBUG=1 \
env -u CONDA_PREFIX uv run maturin develop --release

echo "==> Recording Scalene profile: $SCALENE_JSON"
rm -f "$SCALENE_JSON" "$SCALENE_HTML" "$SCALENE_DIR/scalene-profile.html"
uv run scalene run \
    --memory \
    --profile-only "$SCALENE_SCOPE" \
    -o "$SCALENE_JSON" \
    "$WORKLOAD" \
    --n-agents "$N_AGENTS" \
    --n-locations "$N_LOCATIONS" \
    --days "$DAYS" \
    --random-state "$RANDOM_STATE"

echo "==> Rendering Scalene HTML: $SCALENE_HTML"
(
    cd "$SCALENE_DIR"
    uv run scalene view --standalone "$(basename "$SCALENE_JSON")"
    mv scalene-profile.html "$(basename "$SCALENE_HTML")"
)

if [ ! -s "$SCALENE_HTML" ]; then
    echo "ERROR: Scalene HTML was not created or is empty: $SCALENE_HTML" >&2
    exit 1
fi

echo "==> Recording Samply profile: $SAMPLY_JSON"
if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
    PERF_EVENT_PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
    if [ "$PERF_EVENT_PARANOID" -gt 1 ]; then
        echo "ERROR: kernel.perf_event_paranoid=$PERF_EVENT_PARANOID, which is too restrictive for Samply." >&2
        echo "Run this first, then rerun this script:" >&2
        echo "  sudo sysctl kernel.perf_event_paranoid=1" >&2
        exit 1
    fi
fi

rm -f "$SAMPLY_JSON"
uv run samply record \
    --save-only \
    --rate "$RATE" \
    -o "$SAMPLY_JSON" \
    -- \
    python "$WORKLOAD" \
    --n-agents "$N_AGENTS" \
    --n-locations "$N_LOCATIONS" \
    --days "$DAYS" \
    --random-state "$RANDOM_STATE"

echo
echo "Profiles written:"
echo "  Scalene JSON: $SCALENE_JSON"
echo "  Scalene HTML: $SCALENE_HTML"
echo "  Samply JSON:  $SAMPLY_JSON"
echo
echo "Open Samply in the browser with:"
echo "  uv run samply load $SAMPLY_JSON"
