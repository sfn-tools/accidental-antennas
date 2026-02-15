#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENEMS_ROOT="${OPENEMS_ROOT:-$ROOT/opt/openEMS}"

export OPENEMS_ROOT
export PATH="$OPENEMS_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$OPENEMS_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/antenna_fdtdx:$ROOT/antenna_opt:${PYTHONPATH:-}"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
