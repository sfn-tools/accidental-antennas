#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_IN="$ROOT/antenna_fdtdx/campaign_queue.json"
QUEUE_OUT="$ROOT/antenna_fdtdx/campaign_queue.local.json"

GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi

export QUEUE_IN
export QUEUE_OUT
export GPU_COUNT

python3 - <<PY
import json
import os

queue_in = os.environ["QUEUE_IN"]
queue_out = os.environ["QUEUE_OUT"]
gpu_count = int(os.environ.get("GPU_COUNT", "0"))

with open(queue_in, "r", encoding="utf-8") as handle:
    data = json.load(handle)

root = os.path.abspath(os.path.join(os.path.dirname(queue_in), ".."))
data.setdefault("cpu_threshold", {})["run_root"] = os.path.join(root, "runs_threshold")
openems = data.setdefault("openems", {})
openems["openems_root"] = os.path.join(root, "antenna_opt", "runs_fdtdx")
openems["archive_root"] = os.path.join(root, "passing_designs")
openems["scan_roots"] = [os.path.join(root, "runs")]

runs = data.get("runs", [])
if gpu_count > 0:
    for idx, run in enumerate(runs):
        run["gpu"] = idx % gpu_count
else:
    for run in runs:
        run.pop("gpu", None)

with open(queue_out, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
print(f"Wrote {queue_out} (gpu_count={gpu_count})")
PY
