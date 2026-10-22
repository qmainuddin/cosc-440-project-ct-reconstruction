#!/usr/bin/env bash
# Explicit experiment settings; validation and logging are handled by the launcher.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/run_lab.py" "$@"
