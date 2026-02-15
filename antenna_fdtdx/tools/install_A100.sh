#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
CUDA_PIP_TAG="${CUDA_PIP_TAG:-cuda12_pip}"

run_step() {
  local name="$1"
  shift
  echo "==> ${name}"
  local start_ts
  start_ts="$(date +%s)"
  "$@"
  local end_ts
  end_ts="$(date +%s)"
  echo "<== ${name} (${end_ts} - ${start_ts} s)"
}

echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "VENV_DIR=${VENV_DIR}"
echo "CUDA_PIP_TAG=${CUDA_PIP_TAG}"

#if type module >/dev/null 2>&1; then
#  echo "Module system detected. Load CUDA if needed, e.g.:"
#  echo "  module load cuda/12.8"
#fi

run_step "Create venv" "${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

run_step "Upgrade pip" pip install --upgrade pip setuptools wheel

run_step "Install Python deps" pip install -r "${ROOT_DIR}/requirements.txt"

run_step "Install JAX CUDA wheel" pip install -U "jax[cuda12]"

if [[ -f "${REPO_ROOT}/fdtdx/pyproject.toml" ]]; then
  run_step "Install local fdtdx" pip install -e "${REPO_ROOT}/fdtdx"
fi

mkdir -p "${ROOT_DIR}/.mplconfig"

echo "Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
