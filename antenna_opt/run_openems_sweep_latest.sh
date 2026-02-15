#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COUNT="${COUNT:-4}"
RUN_ROOT="${RUN_ROOT:-$ROOT/antenna_fdtdx/runs}"
OUT_BASE="${OUT_BASE:-$ROOT/antenna_opt/runs_fdtdx_sweep_latest}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
OPENEMS_ROOT="${OPENEMS_ROOT:-$ROOT/opt/openEMS}"

export LD_LIBRARY_PATH="${OPENEMS_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export RUN_ROOT
export COUNT
mkdir -p "${OUT_BASE}"
cd "$ROOT/antenna_opt"

mapfile -t RUN_IDS < <(python3 - <<PY
import os
import time

run_root = os.environ.get("RUN_ROOT")
count = int(os.environ.get("COUNT", "4"))
entries = []
for name in os.listdir(run_root):
    run = os.path.join(run_root, name)
    metrics = os.path.join(run, "metrics.json")
    if not os.path.isdir(run) or not os.path.exists(metrics):
        continue
    try:
        mtime = os.path.getmtime(metrics)
    except OSError:
        continue
    entries.append((mtime, name))
entries.sort(reverse=True)
for _, name in entries[:count]:
    print(name)
PY
)

if [ "${#RUN_IDS[@]}" -eq 0 ]; then
  echo "No runs found in ${RUN_ROOT}"
  exit 0
fi

cases=(
  "fdtdx msl 0.0"
  "fdtdx msl 0.5"
  "fdtdx lumped 0.5"
  "fdtdx_line lumped 0.5"
)

for run_id in "${RUN_IDS[@]}"; do
  run_dir="${RUN_ROOT}/${run_id}"
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
