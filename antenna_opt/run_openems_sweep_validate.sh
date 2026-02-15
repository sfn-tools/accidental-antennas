#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/antenna_fdtdx/runs}"
OUT_BASE="${OUT_BASE:-$ROOT/antenna_opt/runs_fdtdx_sweep_validate}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
OPENEMS_ROOT="${OPENEMS_ROOT:-$ROOT/opt/openEMS}"
RUN_IDS_CSV="${RUN_IDS_CSV:-}"
VALIDATE_COUNT="${VALIDATE_COUNT:-4}"

export LD_LIBRARY_PATH="${OPENEMS_ROOT}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${OUT_BASE}"
cd "$ROOT/antenna_opt"

echo "[run_openems_sweep_validate] ROOT=$ROOT"
echo "[run_openems_sweep_validate] RUN_ROOT=$RUN_ROOT"
echo "[run_openems_sweep_validate] OUT_BASE=$OUT_BASE"
echo "[run_openems_sweep_validate] PY=$PY"
echo "[run_openems_sweep_validate] OPENEMS_ROOT=$OPENEMS_ROOT"

if [[ -n "$RUN_IDS_CSV" ]]; then
  IFS=',' read -r -a RUN_IDS <<<"$RUN_IDS_CSV"
else
  export RUN_ROOT
  export VALIDATE_COUNT
  mapfile -t RUN_IDS < <(python3 - <<'PY'
import os

run_root = os.environ.get("RUN_ROOT")
count = int(os.environ.get("VALIDATE_COUNT", "4"))
entries = []
if os.path.isdir(run_root):
    for name in os.listdir(run_root):
        run = os.path.join(run_root, name)
        metrics = os.path.join(run, "metrics.json")
        if not os.path.isdir(run) or not os.path.exists(metrics):
            continue
        entries.append((os.path.getmtime(metrics), name))
entries.sort(reverse=True)
for _, name in entries[:count]:
    print(name)
PY
)
fi

if [[ "${#RUN_IDS[@]}" -eq 0 ]]; then
  echo "[run_openems_sweep_validate] ERROR: no run IDs selected." >&2
  exit 1
fi

cases=(
  "fdtdx msl 0.0"
  "fdtdx msl 0.5"
  "fdtdx lumped 0.5"
  "fdtdx_line lumped 0.5"
)

for run_id in "${RUN_IDS[@]}"; do
  run_id="$(echo "$run_id" | xargs)"
  [[ -z "$run_id" ]] && continue
  run_dir="${RUN_ROOT}/${run_id}"
  if [[ ! -d "${run_dir}" ]]; then
    echo "[run_openems_sweep_validate] Skipping ${run_id}: missing run dir ${run_dir}" >&2
    continue
  fi
  for case in "${cases[@]}"; do
    read -r port_mode port_type overlap <<<"${case}"
    ov_tag="${overlap/./p}"
    tag="${run_id}_${port_mode}_${port_type}_ov${ov_tag}"
    out_dir="${OUT_BASE}/${tag}"
    echo "=== $(date '+%F %T') running ${tag} ==="
    "${PY}" -m tools.run_openems_from_fdtdx \
      --fdtdx-run "${run_dir}" \
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
done
