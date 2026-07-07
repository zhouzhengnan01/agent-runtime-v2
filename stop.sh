#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./runtime-env.sh

MODE="${JETLINKS_AGENT_RUN_MODE:-auto}"
if [ "${1:-}" = "--docker" ]; then
  MODE="docker"
  shift
elif [ "${1:-}" = "--local" ]; then
  MODE="local"
  shift
fi

is_running() {
  local pid="$1"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

if [ "${MODE}" = "docker" ] || { [ "${MODE}" = "auto" ] && [ -n "${JETLINKS_AGENT_IMAGE:-}" ]; }; then
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${JETLINKS_AGENT_CONTAINER_NAME}"; then
    echo "stopping ${APP_NAME} container=${JETLINKS_AGENT_CONTAINER_NAME}"
    docker rm -f "${JETLINKS_AGENT_CONTAINER_NAME}" >/dev/null
    rm -f "${PID_FILE}"
    echo "${APP_NAME} stopped"
    exit 0
  fi
  echo "${APP_NAME} not running: container=${JETLINKS_AGENT_CONTAINER_NAME} missing"
  rm -f "${PID_FILE}"
  exit 0
fi

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
