#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

APT_PACKAGES=(
  build-essential
  cmake
  git
  libhdf5-dev
  libvtk9-dev
  libboost-all-dev
  libcgal-dev
  libtinyxml-dev
  python3-venv
  python3-pip
)

echo "[bootstrap] installing system dependencies"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y "${APT_PACKAGES[@]}"
else
  echo "apt-get not found; install dependencies from manifests/system_packages.txt"
fi

echo "[bootstrap] creating venv"
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[bootstrap] installing python deps"
python -m pip install -r "$ROOT/manifests/requirements_base.txt"
python -m pip install -e "$ROOT/fdtdx"

echo "[bootstrap] building openEMS + Python bindings"
mkdir -p "$ROOT/opt"
cd "$ROOT/openEMS-Project"
./update_openEMS.sh "$ROOT/opt/openEMS" --disable-GUI --python --python-venv-mode disable

echo "[bootstrap] configuring local campaign queue"
"$ROOT/scripts/configure_queue.sh"

echo "[bootstrap] done"
echo "Next:"
echo "  source $ROOT/scripts/env.sh"
echo "  $ROOT/scripts/run_daemon.sh"
