#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./runtime-env.sh

BASE_PYTHON="${BASE_PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "creating virtualenv ${VENV_DIR} with ${BASE_PYTHON}"
  "${BASE_PYTHON}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

echo "dependencies installed python=${VENV_DIR}/bin/python"
