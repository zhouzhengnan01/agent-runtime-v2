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
BASE_URL_EXPLICIT="false"
if [ -n "${BASE_URL:-}" ]; then
  BASE_URL_EXPLICIT="true"
fi
BASE_URL="${BASE_URL:-http://${APP_HOST}:${APP_PORT}}"
RUN_SMOKE="${RUN_SMOKE:-auto}"
RUN_UI_SMOKE="${RUN_UI_SMOKE:-false}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-/tmp/jetlinks-app-smoke-matrix.json}"
SMOKE_REPORT="${SMOKE_REPORT:-/tmp/jetlinks-app-smoke-matrix.md}"
SMOKE_RETRIES="${SMOKE_RETRIES:-1}"
SMOKE_MIN_TEMPLATES_EXPLICIT="false"
if [ -n "${SMOKE_MIN_TEMPLATES:-}" ]; then
  SMOKE_MIN_TEMPLATES_EXPLICIT="true"
fi
SMOKE_ONLY="${SMOKE_ONLY:-}"
if [ -n "${SMOKE_ONLY}" ] && [ "${SMOKE_MIN_TEMPLATES_EXPLICIT}" != "true" ]; then
  SMOKE_MIN_TEMPLATES="$("${PYTHON_BIN}" - "${SMOKE_ONLY}" <<'PY'
import sys

items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(max(1, len(items)))
PY
)"
else
  SMOKE_MIN_TEMPLATES="${SMOKE_MIN_TEMPLATES:-16}"
fi
SMOKE_EXPECT_CONFIG_APPS="${SMOKE_EXPECT_CONFIG_APPS:-true}"
SMOKE_FAIL_ON_RETRY="${SMOKE_FAIL_ON_RETRY:-true}"
SMOKE_TOKEN="${SMOKE_TOKEN:-${RUNTIME_API_TOKEN:-}}"
AUTO_START_RUNTIME="${AUTO_START_RUNTIME:-true}"
RUNTIME_PID=""
RUNTIME_LOG="${RUNTIME_LOG:-/tmp/jetlinks-quality-gate-runtime.log}"

runtime_required=false
case "${RUN_UI_SMOKE}" in
  1|true|True|yes|Yes)
    runtime_required=true
    ;;
esac
case "${RUN_SMOKE}" in
  1|true|True|yes|Yes)
    runtime_required=true
    ;;
esac

cleanup_runtime() {
  if [ -n "${RUNTIME_PID}" ] && kill -0 "${RUNTIME_PID}" >/dev/null 2>&1; then
    kill "${RUNTIME_PID}" >/dev/null 2>&1 || true
    wait "${RUNTIME_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_runtime EXIT

if [ "${runtime_required}" = "true" ] && [ "${AUTO_START_RUNTIME}" = "true" ] && [ "${BASE_URL_EXPLICIT}" != "true" ]; then
  APP_PORT="$("${PYTHON_BIN}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
  BASE_URL="http://${APP_HOST}:${APP_PORT}"
  RUNTIME_LOG="/tmp/jetlinks-quality-gate-runtime-${APP_PORT}.log"
  echo "== starting temporary runtime: ${BASE_URL} =="
  "${PYTHON_BIN}" -m uvicorn app.main:create_app --factory --host "${APP_HOST}" --port "${APP_PORT}" >"${RUNTIME_LOG}" 2>&1 &
  RUNTIME_PID="$!"
  for _ in $(seq 1 60); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${RUNTIME_PID}" >/dev/null 2>&1; then
      echo "Temporary runtime exited early. Log: ${RUNTIME_LOG}" >&2
      tail -100 "${RUNTIME_LOG}" >&2 || true
      exit 1
    fi
    sleep 0.5
  done
  if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "Temporary runtime did not become healthy. Log: ${RUNTIME_LOG}" >&2
    tail -100 "${RUNTIME_LOG}" >&2 || true
    exit 1
  fi
fi

echo "== pytest quality gate =="
"${PYTHON_BIN}" -m pytest \
  tests/test_agent_tool_loop.py \
  tests/test_tools.py \
  tests/test_mcp.py \
  tests/test_skill_plugins.py \
  tests/test_sandbox.py \
  tests/test_local_subprocess.py \
  tests/test_cli.py \
  tests/test_runtime.py \
  tests/test_workflow_config.py \
  tests/test_acp_stdio.py \
  tests/test_acp_external_backend.py \
  tests/test_apps.py \
  tests/test_acp_ws.py \
  tests/test_app_smoke_matrix.py \
  tests/test_data_auto_annotation_sam3.py \
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
  auto)
    if [ "${BASE_URL_EXPLICIT}" != "true" ] && [ -z "${RUNTIME_PID}" ]; then
      echo "== live smoke skipped: RUN_SMOKE=auto without explicit BASE_URL =="
      echo "Set RUN_SMOKE=true to start a temporary runtime, or set BASE_URL to verify an existing environment."
      exit 0
    fi
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
if [ -n "${SMOKE_ONLY}" ]; then
  smoke_args+=(--only "${SMOKE_ONLY}")
fi
if [ -n "${SMOKE_TOKEN}" ]; then
  RUNTIME_API_TOKEN="${SMOKE_TOKEN}" "${PYTHON_BIN}" tools/app_smoke_matrix.py "${smoke_args[@]}"
else
  "${PYTHON_BIN}" tools/app_smoke_matrix.py "${smoke_args[@]}"
fi

echo "Smoke JSON: ${SMOKE_OUTPUT}"
echo "Smoke report: ${SMOKE_REPORT}"
