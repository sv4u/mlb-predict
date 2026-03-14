#!/usr/bin/env bash
set -euo pipefail
SCRIPT="$1"; shift
if command -v uv > /dev/null 2>&1; then
    exec uv run python "$SCRIPT" "$@"
elif [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    exec python3 "$SCRIPT" "$@"
else
    exec python3 "$SCRIPT" "$@"
fi
