#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x ".venv312/bin/python" ]; then
    PYTHON_BIN=".venv312/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-18012}"
BASE_URL="${BASE_URL:-http://${APP_HOST}:${APP_PORT}}"
RUN_SMOKE="${RUN_SMOKE:-auto}"
RUN_UI_SMOKE="${RUN_UI_SMOKE:-false}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-/tmp/jetlinks-app-smoke-matrix.json}"
SMOKE_REPORT="${SMOKE_REPORT:-/tmp/jetlinks-app-smoke-matrix.md}"
SMOKE_RETRIES="${SMOKE_RETRIES:-1}"
SMOKE_MIN_TEMPLATES="${SMOKE_MIN_TEMPLATES:-16}"
SMOKE_EXPECT_CONFIG_APPS="${SMOKE_EXPECT_CONFIG_APPS:-true}"
SMOKE_FAIL_ON_RETRY="${SMOKE_FAIL_ON_RETRY:-true}"
SMOKE_TOKEN="${SMOKE_TOKEN:-${RUNTIME_API_TOKEN:-}}"

echo "== pytest quality gate =="
"${PYTHON_BIN}" -m pytest \
  tests/test_agent_tool_loop.py \
  tests/test_cli.py \
  tests/test_runtime.py \
  tests/test_acp_stdio.py \
  tests/test_acp_external_backend.py \
  tests/test_apps.py \
  tests/test_acp_ws.py \
  tests/test_app_smoke_matrix.py \
  tests/test_workbench_runtime_events.py \
  tests/test_observability.py \
  -q

case "${RUN_UI_SMOKE}" in
  0|false|False|no|No)
    ;;
  1|true|True|yes|Yes)
    if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      echo "Workbench UI smoke requires reachable Runtime: ${BASE_URL}/health" >&2
      exit 1
    fi
    if [ ! -d "frontend/workbench/node_modules" ]; then
      echo "Workbench UI smoke requires frontend dependencies: cd frontend/workbench && npm install" >&2
      exit 1
    fi
    echo "== workbench UI smoke =="
    (cd frontend/workbench && APP_URL="${BASE_URL}/static/workbench.html" npm run test:ui)
    ;;
  *)
    echo "Invalid RUN_UI_SMOKE=${RUN_UI_SMOKE}; expected true/false" >&2
    exit 2
    ;;
esac

case "${RUN_SMOKE}" in
  0|false|False|no|No)
    echo "== live smoke skipped: RUN_SMOKE=${RUN_SMOKE} =="
    exit 0
    ;;
  1|true|True|yes|Yes|auto)
    ;;
  *)
    echo "Invalid RUN_SMOKE=${RUN_SMOKE}; expected auto/true/false" >&2
    exit 2
    ;;
esac

if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
  if [ "${RUN_SMOKE}" = "auto" ]; then
    echo "== live smoke skipped: ${BASE_URL}/health is not reachable =="
    echo "Start the runtime or set RUN_SMOKE=true to require live smoke."
    exit 0
  fi
  echo "Runtime health check failed: ${BASE_URL}/health" >&2
  exit 1
fi

echo "== live app smoke matrix =="
smoke_args=(
  --base-url "${BASE_URL}"
  --output "${SMOKE_OUTPUT}"
  --report-markdown "${SMOKE_REPORT}"
  --retries "${SMOKE_RETRIES}"
  --min-templates "${SMOKE_MIN_TEMPLATES}"
)
case "${SMOKE_EXPECT_CONFIG_APPS}" in
  1|true|True|yes|Yes)
    smoke_args+=(--expect-config-apps)
    ;;
  0|false|False|no|No)
    ;;
  *)
    echo "Invalid SMOKE_EXPECT_CONFIG_APPS=${SMOKE_EXPECT_CONFIG_APPS}; expected true/false" >&2
    exit 2
    ;;
esac
case "${SMOKE_FAIL_ON_RETRY}" in
  1|true|True|yes|Yes)
    smoke_args+=(--fail-on-retry)
    ;;
  0|false|False|no|No)
    ;;
  *)
    echo "Invalid SMOKE_FAIL_ON_RETRY=${SMOKE_FAIL_ON_RETRY}; expected true/false" >&2
    exit 2
    ;;
esac
if [ -n "${SMOKE_TOKEN}" ]; then
  RUNTIME_API_TOKEN="${SMOKE_TOKEN}" "${PYTHON_BIN}" tools/app_smoke_matrix.py "${smoke_args[@]}"
else
  "${PYTHON_BIN}" tools/app_smoke_matrix.py "${smoke_args[@]}"
fi

echo "Smoke JSON: ${SMOKE_OUTPUT}"
echo "Smoke report: ${SMOKE_REPORT}"
