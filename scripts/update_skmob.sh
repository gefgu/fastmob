#!/usr/bin/env bash
# Rebuild/install fastmob._core into an existing local virtual environment.
#
# This script runs maturin against the selected environment. If the target uv
# virtualenv lacks pip and a usable uv executable is unavailable, it bootstraps
# pip inside that environment with ensurepip.
#
# Usage:
#   bash scripts/update_skmob.sh                 # update .venv
#   bash scripts/update_skmob.sh .venv
#   bash scripts/update_skmob.sh .venv-skmob
#   bash scripts/update_skmob.sh --env .venv-skmob

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME=".venv"

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --env)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --env requires .venv or .venv-skmob."
                exit 2
            fi
            ENV_NAME="$2"
            shift 2
            ;;
        .venv|.venv-skmob)
            ENV_NAME="$1"
            shift
            ;;
        *)
            echo "ERROR: unsupported argument '$1'."
            echo "       Use .venv, .venv-skmob, or --env <name>."
            exit 2
            ;;
    esac
done

case "$ENV_NAME" in
    .venv|.venv-skmob) ;;
    *)
        echo "ERROR: environment must be .venv or .venv-skmob."
        exit 2
        ;;
esac

ENV_DIR="$REPO_ROOT/$ENV_NAME"
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "ERROR: $ENV_NAME not found or missing bin/python."
    echo "       Create it first, then re-run this script."
    exit 1
fi

MATURIN_CMD=()
if [ -x "$ENV_DIR/bin/maturin" ]; then
    MATURIN_CMD=("$ENV_DIR/bin/maturin")
elif [ -x "$REPO_ROOT/.venv/bin/maturin" ]; then
    MATURIN_CMD=("$REPO_ROOT/.venv/bin/maturin")
elif command -v maturin >/dev/null 2>&1; then
    MATURIN_CMD=(maturin)
elif command -v uv >/dev/null 2>&1; then
    MATURIN_CMD=(uv tool run --from "maturin>=1.13,<2.0" maturin)
else
    echo "ERROR: maturin was not found in $ENV_NAME, .venv, or PATH, and uv is unavailable."
    echo "       Install uv or maturin first."
    exit 1
fi

echo "==> Updating fastmob in $ENV_NAME with maturin develop --release ..."
DEVELOP_ARGS=(develop --release)
if [ ! -x "$ENV_DIR/bin/pip" ]; then
    if command -v uv >/dev/null 2>&1 && uv --version >/dev/null 2>&1; then
        DEVELOP_ARGS=(develop --uv --release)
    else
        echo "==> $ENV_NAME has no pip and uv is unavailable; bootstrapping pip with ensurepip ..."
        "$ENV_DIR/bin/python" -m ensurepip --upgrade
    fi
fi

env -u CONDA_PREFIX \
    NK_TARGET_SAPPHIRE="${NK_TARGET_SAPPHIRE:-0}" \
    NK_TARGET_SAPPHIREAMX="${NK_TARGET_SAPPHIREAMX:-0}" \
    NK_TARGET_GRANITEAMX="${NK_TARGET_GRANITEAMX:-0}" \
    NK_TARGET_DIAMOND="${NK_TARGET_DIAMOND:-0}" \
    NK_TARGET_TURIN="${NK_TARGET_TURIN:-0}" \
    NK_TARGET_SIERRA="${NK_TARGET_SIERRA:-0}" \
    VIRTUAL_ENV="$ENV_DIR" \
    PATH="$ENV_DIR/bin:$PATH" \
    "${MATURIN_CMD[@]}" "${DEVELOP_ARGS[@]}"

echo "Done."
