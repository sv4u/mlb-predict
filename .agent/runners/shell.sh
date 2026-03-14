#!/usr/bin/env bash
set -euo pipefail
SCRIPT="$1"; shift
exec bash "$SCRIPT" "$@"
