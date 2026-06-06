#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"

find_python() {
  if [ -n "${PYTHON_BIN:-}" ] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if "${PYTHON_BIN}" - <<PY
import sys
raise SystemExit(0 if sys.version_info[:2] == (${PYTHON_VERSION%.*}, ${PYTHON_VERSION#*.}) else 1)
PY
    then
      printf '%s\n' "${PYTHON_BIN}"
      return
    fi
  fi

  for candidate in "python${PYTHON_VERSION}" python3.12 python; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" - <<PY
import sys
raise SystemExit(0 if sys.version_info[:2] == (${PYTHON_VERSION%.*}, ${PYTHON_VERSION#*.}) else 1)
PY
    then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  return 1
}

install_python312_with_apt() {
  if [ "$(id -u)" -ne 0 ]; then
    return 1
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  if apt-get install -y python3.12 python3.12-venv python3.12-dev; then
    return 0
  fi

  apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y python3.12 python3.12-venv python3.12-dev
}

PYTHON="$(find_python || true)"
if [ -z "${PYTHON}" ]; then
  echo "Python ${PYTHON_VERSION} not found; trying apt/deadsnakes install." >&2
  install_python312_with_apt
  PYTHON="$(find_python || true)"
fi

if [ -z "${PYTHON}" ]; then
  echo "Python ${PYTHON_VERSION} is required. Install it or set PYTHON_BIN=/path/to/python3.12." >&2
  exit 2
fi

if [ ! -x "${VENV_DIR}/bin/python" ] || ! "${VENV_DIR}/bin/python" - <<PY >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (${PYTHON_VERSION%.*}, ${PYTHON_VERSION#*.}) else 1)
PY
then
  rm -rf "${VENV_DIR}"
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

PIP_ARGS=()
if [ -n "${PIP_INDEX_URL}" ]; then
  PIP_ARGS+=(--index-url "${PIP_INDEX_URL}")
fi

"${VENV_DIR}/bin/python" -m pip install "${PIP_ARGS[@]}" -U pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install "${PIP_ARGS[@]}" -r requirements.txt

echo "Installed runtime dependencies:"
"${VENV_DIR}/bin/python" --version
"${VENV_DIR}/bin/python" -m pip check
