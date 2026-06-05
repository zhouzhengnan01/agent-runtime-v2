#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./runtime-env.sh

is_running() {
  local pid="$1"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

health_url() {
  printf 'http://%s:%s%s' "${APP_HOST}" "${APP_PORT}" "${APP_HEALTH_PATH}"
}

pid=""
if [ -f "${PID_FILE}" ]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
fi

if ! is_running "${pid}"; then
  if [ -n "${pid}" ]; then
    echo "status=stopped pid=${pid} reason=stale_pid pid_file=${PID_FILE}"
  else
    echo "status=stopped pid= reason=no_pid_file pid_file=${PID_FILE}"
  fi
  exit 3
fi

if curl -fsS "$(health_url)" >/dev/null 2>&1; then
  echo "status=running pid=${pid} health=ok url=http://${APP_HOST}:${APP_PORT} log=${LOG_FILE}"
  exit 0
fi

echo "status=degraded pid=${pid} health=failed url=http://${APP_HOST}:${APP_PORT} log=${LOG_FILE}"
exit 2
