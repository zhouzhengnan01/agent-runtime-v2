#!/usr/bin/env bash

# Shared runtime settings for local/process-manager deployments.
# Java or shell callers may override any value via environment variables.

APP_NAME="${APP_NAME:-jetlinks-agent-runtime-v2}"
APP_MODULE="${APP_MODULE:-app.main:app}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_WORKERS="${APP_WORKERS:-1}"
APP_LOG_LEVEL="${APP_LOG_LEVEL:-info}"
APP_HEALTH_PATH="${APP_HEALTH_PATH:-/health}"
APP_START_TIMEOUT_SECONDS="${APP_START_TIMEOUT_SECONDS:-30}"
APP_STOP_TIMEOUT_SECONDS="${APP_STOP_TIMEOUT_SECONDS:-15}"

# Docker deployment settings. If JETLINKS_AGENT_IMAGE is set, ./up.sh starts the
# service from that image and mounts this project directory into the container.
JETLINKS_AGENT_IMAGE="${JETLINKS_AGENT_IMAGE:-}"
JETLINKS_AGENT_CONTAINER_NAME="${JETLINKS_AGENT_CONTAINER_NAME:-${APP_NAME}}"
JETLINKS_AGENT_CONTAINER_WORKDIR="${JETLINKS_AGENT_CONTAINER_WORKDIR:-/workspace/code/jetlinks-agent-runtime-agent-v2}"
JETLINKS_AGENT_CONTAINER_PORT="${JETLINKS_AGENT_CONTAINER_PORT:-${APP_PORT}}"
JETLINKS_AGENT_PULL_IMAGE="${JETLINKS_AGENT_PULL_IMAGE:-true}"
JETLINKS_AGENT_DOCKER_PLATFORM="${JETLINKS_AGENT_DOCKER_PLATFORM:-}"
JETLINKS_AGENT_DOCKER_EXTRA_ARGS="${JETLINKS_AGENT_DOCKER_EXTRA_ARGS:-}"

# Debug-only fixed reply for Java/ACP integration tests. Leave disabled for real model runs.
JETLINKS_AGENT_FIXED_REPLY_ENABLED="${JETLINKS_AGENT_FIXED_REPLY_ENABLED:-false}"
JETLINKS_AGENT_FIXED_REPLY_FOREVER="${JETLINKS_AGENT_FIXED_REPLY_FOREVER:-false}"
JETLINKS_AGENT_FIXED_REPLY_INTERVAL_SECONDS="${JETLINKS_AGENT_FIXED_REPLY_INTERVAL_SECONDS:-5}"

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
