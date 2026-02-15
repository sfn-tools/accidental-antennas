#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/antenna_fdtdx/runs}"
RUN="${RUN:-}"
OUT_BASE="${OUT_BASE:-$ROOT/antenna_opt/runs_fdtdx_sweep}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
OPENEMS_ROOT="${OPENEMS_ROOT:-$ROOT/opt/openEMS}"
export RUN_ROOT

if [[ -z "$RUN" ]]; then
  RUN="$(python3 - <<'PY'
import os
run_root = os.environ.get("RUN_ROOT", "")
if not os.path.isdir(run_root):
    raise SystemExit(0)
latest = None
for name in os.listdir(run_root):
    run_dir = os.path.join(run_root, name)
    metrics = os.path.join(run_dir, "metrics.json")
    if not os.path.isdir(run_dir) or not os.path.exists(metrics):
        continue
    mtime = os.path.getmtime(metrics)
    if latest is None or mtime > latest[0]:
        latest = (mtime, run_dir)
if latest:
    print(latest[1])
PY
)"
fi

if [[ -z "$RUN" ]]; then
  echo "[run_openems_sweep] ERROR: no run found. Set RUN=<fdtdx_run_dir> or ensure RUN_ROOT has runs." >&2
  exit 1
fi
if [[ ! -d "$RUN" ]]; then
  echo "[run_openems_sweep] ERROR: run directory does not exist: $RUN" >&2
  exit 1
fi

RUN_ID="$(basename "$RUN")"
export LD_LIBRARY_PATH="${OPENEMS_ROOT}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT_BASE"
cd "$ROOT/antenna_opt"

echo "[run_openems_sweep] ROOT=$ROOT"
echo "[run_openems_sweep] RUN=$RUN"
echo "[run_openems_sweep] OUT_BASE=$OUT_BASE"
echo "[run_openems_sweep] PY=$PY"
echo "[run_openems_sweep] OPENEMS_ROOT=$OPENEMS_ROOT"

cases=(
  "fdtdx msl 0.0"
  "fdtdx msl 0.25"
  "fdtdx msl 0.5"
  "fdtdx lumped 0.0"
  "fdtdx lumped 0.25"
  "fdtdx lumped 0.5"
  "fdtdx_line lumped 0.0"
  "fdtdx_line lumped 0.5"
)

for case in "${cases[@]}"; do
  read -r port_mode port_type overlap <<<"$case"
  ov_tag="${overlap/./p}"
  tag="${RUN_ID}_${port_mode}_${port_type}_ov${ov_tag}"
  out_dir="${OUT_BASE}/${tag}"
  echo "=== $(date '+%F %T') running ${tag} ==="
  "${PY}" -m tools.run_openems_from_fdtdx \
    --fdtdx-run "${RUN}" \
    --quality fast \
    --port-mode "${port_mode}" \
    --port-type "${port_type}" \
    --metrics \
    --prune \
    --match-fdtdx-mesh \
    --mirror-fdtdx \
    --no-clip-to-wedge \
    --overlap-mm "${overlap}" \
    --out-dir "${out_dir}"
done
