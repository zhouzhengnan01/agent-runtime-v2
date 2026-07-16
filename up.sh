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

mkdir -p "${RUNTIME_DIR}"

is_running() {
  local pid="$1"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

health_url() {
  printf 'http://%s:%s%s' "${APP_HEALTH_HOST}" "${APP_PORT}" "${APP_HEALTH_PATH}"
}

docker_run() {
  if [ -z "${JETLINKS_AGENT_IMAGE:-}" ]; then
    echo "JETLINKS_AGENT_IMAGE is required for docker mode" >&2
    exit 2
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for docker mode" >&2
    exit 2
  fi

  if is_truthy "${JETLINKS_AGENT_PULL_IMAGE}"; then
    echo "pulling ${JETLINKS_AGENT_IMAGE}"
    if [ -n "${JETLINKS_AGENT_DOCKER_PLATFORM}" ]; then
      docker pull --platform "${JETLINKS_AGENT_DOCKER_PLATFORM}" "${JETLINKS_AGENT_IMAGE}"
    else
      docker pull "${JETLINKS_AGENT_IMAGE}"
    fi
  fi

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${JETLINKS_AGENT_CONTAINER_NAME}"; then
    echo "removing existing container ${JETLINKS_AGENT_CONTAINER_NAME}"
    docker rm -f "${JETLINKS_AGENT_CONTAINER_NAME}" >/dev/null
  fi

  touch "${LOG_FILE}"
  local code_source
  if is_truthy "${JETLINKS_AGENT_MOUNT_CODE}"; then
    code_source="${PROJECT_DIR}"
  else
    code_source="image"
  fi
  echo "starting ${APP_NAME} container=${JETLINKS_AGENT_CONTAINER_NAME} image=${JETLINKS_AGENT_IMAGE} host_port=${APP_PORT} code=${code_source}"

  local run_args=(
    docker run -d
  )
  if [ -n "${JETLINKS_AGENT_DOCKER_PLATFORM}" ]; then
    run_args+=(--platform "${JETLINKS_AGENT_DOCKER_PLATFORM}")
  fi
  run_args+=(
    --name "${JETLINKS_AGENT_CONTAINER_NAME}"
    -p "${APP_PORT}:${JETLINKS_AGENT_CONTAINER_PORT}"
    -w "${JETLINKS_AGENT_CONTAINER_WORKDIR}"
    -e APP_HOST=0.0.0.0
    -e APP_PORT="${JETLINKS_AGENT_CONTAINER_PORT}"
    -e APP_WORKERS="${APP_WORKERS}"
    -e APP_LOG_LEVEL="${APP_LOG_LEVEL}"
    -e APP_HEALTH_PATH="${APP_HEALTH_PATH}"
    -e JETLINKS_REVIEW_MAX_CONCURRENCY="${JETLINKS_REVIEW_MAX_CONCURRENCY}"
    -e JETLINKS_READY_MAX_QUEUE_BACKLOG="${JETLINKS_READY_MAX_QUEUE_BACKLOG}"
    -e JETLINKS_READY_CPU_LOAD_THRESHOLD="${JETLINKS_READY_CPU_LOAD_THRESHOLD}"
    -e JETLINKS_REVIEW_STATE_DIR="${JETLINKS_REVIEW_STATE_DIR}"
    -e JETLINKS_AGENT_FIXED_REPLY_ENABLED="${JETLINKS_AGENT_FIXED_REPLY_ENABLED}"
    -e JETLINKS_AGENT_FIXED_REPLY_FOREVER="${JETLINKS_AGENT_FIXED_REPLY_FOREVER}"
    -e JETLINKS_AGENT_FIXED_REPLY_INTERVAL_SECONDS="${JETLINKS_AGENT_FIXED_REPLY_INTERVAL_SECONDS}"
  )
  if is_truthy "${JETLINKS_AGENT_MOUNT_CODE}"; then
    run_args+=(-v "${PROJECT_DIR}:${JETLINKS_AGENT_CONTAINER_WORKDIR}")
  fi
  if [ -n "${JETLINKS_AGENT_DOCKER_EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    local extra_args=(${JETLINKS_AGENT_DOCKER_EXTRA_ARGS})
    run_args+=("${extra_args[@]}")
  fi
  run_args+=("${JETLINKS_AGENT_IMAGE}")
  if [ "$#" -gt 0 ]; then
    run_args+=("$@")
  else
    run_args+=(
      sh -c "mkdir -p '${RUNTIME_DIR}' && exec python -m uvicorn '${APP_MODULE}' --host 0.0.0.0 --port '${JETLINKS_AGENT_CONTAINER_PORT}' --workers '${APP_WORKERS}' --log-level '${APP_LOG_LEVEL}' 2>&1 | tee -a '${LOG_FILE}'"
    )
  fi

  local container_id
  container_id="$("${run_args[@]}")"
  echo "${container_id}" > "${PID_FILE}"

  deadline=$((SECONDS + APP_START_TIMEOUT_SECONDS))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${JETLINKS_AGENT_CONTAINER_NAME}"; then
      echo "${APP_NAME} container exited; logs:" >&2
      docker logs --tail 80 "${JETLINKS_AGENT_CONTAINER_NAME}" >&2 || true
      rm -f "${PID_FILE}"
      exit 1
    fi
    if curl -fsS "$(health_url)" >/dev/null 2>&1; then
      echo "${APP_NAME} started container=${JETLINKS_AGENT_CONTAINER_NAME} url=http://${APP_HOST}:${APP_PORT}"
      exit 0
    fi
    sleep 1
  done

  echo "${APP_NAME} start timeout container=${JETLINKS_AGENT_CONTAINER_NAME}; health=$(health_url); docker logs --tail 200 ${JETLINKS_AGENT_CONTAINER_NAME}" >&2
  exit 1
}

if [ "${MODE}" = "docker" ] || { [ "${MODE}" = "auto" ] && [ -n "${JETLINKS_AGENT_IMAGE:-}" ]; }; then
  docker_run "$@"
fi

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
