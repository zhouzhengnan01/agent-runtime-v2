#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./runtime-env.sh

mkdir -p "${RUNTIME_DIR}"

case "${LOAD_ENV_FILE}" in
  1|true|True|yes|Yes)
    if [ -f "${ENV_FILE}" ]; then
      set -a
      # shellcheck disable=SC1090
      source "${ENV_FILE}"
      set +a
    fi
    ;;
  0|false|False|no|No)
    ;;
  *)
    echo "Invalid LOAD_ENV_FILE=${LOAD_ENV_FILE}; expected true/false" >&2
    exit 2
    ;;
esac

is_running() {
  local pid="$1"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

health_url() {
  printf 'http://%s:%s%s' "${APP_HOST}" "${APP_PORT}" "${APP_HEALTH_PATH}"
}

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import uvicorn
PY
then
  echo "missing python dependencies for ${PYTHON_BIN}; run ./install-deps.sh first" >&2
  exit 10
fi

if [ -f "${PID_FILE}" ]; then
  existing_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if is_running "${existing_pid}"; then
    echo "${APP_NAME} already running pid=${existing_pid} url=http://${APP_HOST}:${APP_PORT}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

touch "${LOG_FILE}"
echo "starting ${APP_NAME} host=${APP_HOST} port=${APP_PORT} log=${LOG_FILE}"
export PYTHON_BIN APP_MODULE APP_HOST APP_PORT APP_WORKERS APP_LOG_LEVEL LOG_FILE
pid="$("${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
import subprocess
import sys

python_bin = os.environ["PYTHON_BIN"]
command = [
    python_bin,
    "-m",
    "uvicorn",
    os.environ["APP_MODULE"],
    "--host",
    os.environ["APP_HOST"],
    "--port",
    os.environ["APP_PORT"],
    "--workers",
    os.environ["APP_WORKERS"],
    "--log-level",
    os.environ["APP_LOG_LEVEL"],
]
log = open(os.environ["LOG_FILE"], "ab", buffering=0)
process = subprocess.Popen(
    command,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    cwd=os.getcwd(),
    env=os.environ.copy(),
    start_new_session=True,
)
print(process.pid)
sys.stdout.flush()
PY
)"
echo "${pid}" > "${PID_FILE}"

deadline=$((SECONDS + APP_START_TIMEOUT_SECONDS))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if ! is_running "${pid}"; then
    echo "${APP_NAME} failed to start pid=${pid}; see ${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi
  if curl -fsS "$(health_url)" >/dev/null 2>&1; then
    echo "${APP_NAME} started pid=${pid} url=http://${APP_HOST}:${APP_PORT}"
    exit 0
  fi
  sleep 1
done

echo "${APP_NAME} start timeout pid=${pid}; health=$(health_url); see ${LOG_FILE}" >&2
exit 1
