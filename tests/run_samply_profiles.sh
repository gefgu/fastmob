#!/usr/bin/env bash
# Run Brightkite samply profiling jobs.
# Firefox Profiler JSON traces and manifests are written to .profiles/samply/ by default.
#
# Usage:
#   bash tests/run_samply_profiles.sh --list
#   bash tests/run_samply_profiles.sh --dry-run --rows 10000 --workload radius_of_gyration
#   bash tests/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration
#   bash tests/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration --implementation both
#   bash tests/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration --scope full
#   bash tests/run_samply_profiles.sh                              # full 4M-row sweep
#
# View a recorded trace with:
#   samply load .profiles/samply/skmob2/<workload>.json.gz
#
# Any extra arguments are forwarded directly to scripts/profile_brightkite_samply.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash tests/setup_env.sh' first."
    exit 1
fi

source .venv/bin/activate
unset CONDA_PREFIX

if command -v samply >/dev/null 2>&1; then
    SAMPLY_BIN="$(command -v samply)"
elif [ -x "$HOME/.cargo/bin/samply" ]; then
    SAMPLY_BIN="$HOME/.cargo/bin/samply"
else
    SAMPLY_BIN=""
fi

SKIP_BUILD=0
for arg in "$@"; do
    if [ "$arg" = "--list" ] || [ "$arg" = "--dry-run" ]; then
        SKIP_BUILD=1
        break
    fi
done

if [ "$SKIP_BUILD" -eq 0 ]; then
    if [ -z "$SAMPLY_BIN" ] || [ ! -x "$SAMPLY_BIN" ]; then
        if ! command -v cargo >/dev/null 2>&1; then
            echo "ERROR: 'samply' is not installed and 'cargo' is unavailable."
            echo "       Install Rust (https://rustup.rs) and re-run, or install samply manually:"
            echo "         cargo install --locked samply"
            exit 1
        fi
        echo "==> Installing samply via cargo (this may take a while) ..."
        cargo install --locked samply
        if command -v samply >/dev/null 2>&1; then
            SAMPLY_BIN="$(command -v samply)"
        else
            SAMPLY_BIN="$HOME/.cargo/bin/samply"
        fi
    fi
    "$SAMPLY_BIN" --version

    if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
        PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
        if [ "$PARANOID" -gt 1 ] 2>/dev/null; then
            echo "WARNING: kernel.perf_event_paranoid=$PARANOID. samply may fail to record."
            echo "         Lower it for this session with:  sudo sysctl kernel.perf_event_paranoid=1"
        fi
    fi

    # Use release speed while keeping Rust debug symbols available to samply.
    export CARGO_PROFILE_RELEASE_DEBUG=1
    if ! maturin develop --release; then
        maturin develop --release --uv
    fi
fi

python scripts/profile_brightkite_samply.py --samply-bin "${SAMPLY_BIN:-samply}" "$@"
