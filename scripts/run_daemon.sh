#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

QUEUE="$ROOT/antenna_fdtdx/campaign_queue.local.json"
if [ ! -f "$QUEUE" ]; then
  "$ROOT/scripts/configure_queue.sh"
fi

cd "$ROOT/antenna_fdtdx"
nohup "$ROOT/.venv/bin/python" -m tools.campaign_daemon \
  --queue "$QUEUE" \
  --log "$ROOT/antenna_fdtdx/campaign_daemon.log" \
  > "$ROOT/antenna_fdtdx/nohup_campaign_daemon.out" 2>&1 &

echo "campaign_daemon started with queue: $QUEUE"
