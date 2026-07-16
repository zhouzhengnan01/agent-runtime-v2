#!/usr/bin/env bash

# Shared runtime settings for local/process-manager deployments.
# Java or shell callers may override any value via environment variables.

APP_NAME="${APP_NAME:-jetlinks-agent-runtime-v2}"
APP_MODULE="${APP_MODULE:-app.main:app}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_HEALTH_HOST="${APP_HEALTH_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8012}"
APP_WORKERS="${APP_WORKERS:-4}"
APP_LOG_LEVEL="${APP_LOG_LEVEL:-info}"
APP_HEALTH_PATH="${APP_HEALTH_PATH:-/health}"
APP_START_TIMEOUT_SECONDS="${APP_START_TIMEOUT_SECONDS:-30}"
APP_STOP_TIMEOUT_SECONDS="${APP_STOP_TIMEOUT_SECONDS:-15}"
JETLINKS_REVIEW_MAX_CONCURRENCY="${JETLINKS_REVIEW_MAX_CONCURRENCY:-2}"
JETLINKS_READY_MAX_QUEUE_BACKLOG="${JETLINKS_READY_MAX_QUEUE_BACKLOG:-32}"
JETLINKS_READY_CPU_LOAD_THRESHOLD="${JETLINKS_READY_CPU_LOAD_THRESHOLD:-3.5}"
JETLINKS_REVIEW_STATE_DIR="${JETLINKS_REVIEW_STATE_DIR:-/tmp/jetlinks-agent-runtime}"

RUNTIME_DIR="${RUNTIME_DIR:-.runtime/server}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/server.pid}"
LOG_FILE="${LOG_FILE:-${RUNTIME_DIR}/server.log}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

UVICORN_ARGS=(
  "-m" "uvicorn" "${APP_MODULE}"
  "--host" "${APP_HOST}"
  "--port" "${APP_PORT}"
  "--workers" "${APP_WORKERS}"
  "--log-level" "${APP_LOG_LEVEL}"
)
