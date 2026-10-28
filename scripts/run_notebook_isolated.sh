#!/usr/bin/env bash
# Execute a docs notebook top to bottom in a fresh virtualenv and an empty temp/cache
# directory outside the checkout, then save the executed notebook, a log and the wall time.
#
# Usage: scripts/run_notebook_isolated.sh NOTEBOOK OUT_DIR [WHEEL]
#   WHEEL  optional local wheel(s), separated by spaces (fastmob, fastmob-vis), for notebooks that
#          need an unreleased API;
#          without it the notebook installs the published release from PyPI.
#   A call log ``NAME.calls.json`` of the fastmob functions called directly by notebook cells is
#   written next to the other outputs (see scripts/notebook_trace/sitecustomize.py).
#   OVERLAY_SRC  optional checkout whose Python files are copied over the installed wheel
#          (pre-release testing of Python-only fixes; not a substitute for a clean run).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
notebook="$(realpath "$1")"
out="$(mkdir -p "$2" && realpath "$2")"
wheel="${3:-}"
name="$(basename "$notebook" .ipynb)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cd "$work"
uv venv --seed -q --python "${PYTHON:-3.12}" "$work/venv"
export VIRTUAL_ENV="$work/venv" PATH="$work/venv/bin:$PATH" TMPDIR="$work/tmp"
mkdir -p "$TMPDIR"
# shellcheck disable=SC2086  # WHEEL may list several wheels separated by spaces
uv pip install -q jupyter nbconvert ipykernel ${wheel:+$wheel}
if [ -n "${OVERLAY_SRC:-}" ]; then
    site="$(python -c 'import fastmob, pathlib; print(pathlib.Path(fastmob.__file__).parent)')"
    (cd "$OVERLAY_SRC/fastmob" && find . -name '*.py' -exec cp --parents {} "$site" \;)
fi
cp "$notebook" "$work/$name.ipynb"
start=$(date +%s)
export FASTMOB_CALL_LOG="$out/$name.calls.json"
export PYTHONPATH="$ROOT/scripts/notebook_trace${PYTHONPATH:+:$PYTHONPATH}"
python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
    --output "$name.executed.ipynb" "$work/$name.ipynb" >"$out/$name.log" 2>&1 && status=0 || status=$?
echo "exit=$status wall_seconds=$(( $(date +%s) - start )) python=$(python -V 2>&1)" | tee "$out/$name.status"
[ -f "$work/$name.executed.ipynb" ] && cp "$work/$name.executed.ipynb" "$out/$name.executed.ipynb"
exit "$status"
