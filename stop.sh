#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./runtime-env.sh

is_running() {
  local pid="$1"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

if [ ! -f "${PID_FILE}" ]; then
  echo "${APP_NAME} not running: pid file missing"
  exit 0
fi

pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
if ! is_running "${pid}"; then
  echo "${APP_NAME} not running: stale pid=${pid}"
  rm -f "${PID_FILE}"
  exit 0
fi

echo "stopping ${APP_NAME} pid=${pid}"
kill "${pid}" 2>/dev/null || true

deadline=$((SECONDS + APP_STOP_TIMEOUT_SECONDS))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if ! is_running "${pid}"; then
    rm -f "${PID_FILE}"
    echo "${APP_NAME} stopped"
    exit 0
  fi
  sleep 1
done

echo "${APP_NAME} did not stop after ${APP_STOP_TIMEOUT_SECONDS}s; sending SIGKILL pid=${pid}" >&2
kill -9 "${pid}" 2>/dev/null || true
rm -f "${PID_FILE}"
echo "${APP_NAME} stopped"
