#!/usr/bin/env sh
set -eu

WORKDIR_PATH="${JETLINKS_AGENT_CONTAINER_WORKDIR:-/workspace/code/jetlinks-agent-runtime-agent-v2}"
DEFAULTS_DIR="${JETLINKS_AGENT_DEFAULTS_DIR:-/opt/jetlinks-agent-runtime-agent-v2-defaults}"
export JETLINKS_AGENT_BUILTIN_ROOT="${JETLINKS_AGENT_BUILTIN_ROOT:-${DEFAULTS_DIR}}"
export JETLINKS_AGENT_UPLOAD_ROOT="${JETLINKS_AGENT_UPLOAD_ROOT:-${JETLINKS_AGENT_USER_ROOT:-${WORKDIR_PATH}}}"
export JETLINKS_AGENT_CONFIG_UPLOAD_DIR="${JETLINKS_AGENT_CONFIG_UPLOAD_DIR:-${JETLINKS_AGENT_UPLOAD_ROOT}/config/upload}"
export JETLINKS_AGENT_PLUGINS_UPLOAD_DIR="${JETLINKS_AGENT_PLUGINS_UPLOAD_DIR:-${JETLINKS_AGENT_UPLOAD_ROOT}/plugins/upload}"
export JETLINKS_AGENT_STATIC_UPLOAD_DIR="${JETLINKS_AGENT_STATIC_UPLOAD_DIR:-${JETLINKS_AGENT_UPLOAD_ROOT}/static/upload}"
export APP_MODULE="${APP_MODULE:-app.main:app}"
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8000}"
export APP_WORKERS="${APP_WORKERS:-auto}"
export APP_LOG_LEVEL="${APP_LOG_LEVEL:-info}"
export APP_WORKERS_AUTO_MAX="${APP_WORKERS_AUTO_MAX:-4}"

mkdir -p \
  "${JETLINKS_AGENT_CONFIG_UPLOAD_DIR}" \
  "${JETLINKS_AGENT_PLUGINS_UPLOAD_DIR}" \
  "${JETLINKS_AGENT_STATIC_UPLOAD_DIR}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

debug_frp_digits() {
  value="$(printf '%s' "${1:-}" | tr -cd '0-9')"
  if [ -n "${value}" ]; then
    printf '%s' "${value}"
    return 0
  fi
  return 1
}

resolve_debug_frp_remote_port() {
  explicit_port="${JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT:-}"
  if [ -n "${explicit_port}" ]; then
    printf '%s' "${explicit_port}"
    return 0
  fi

  pod_ordinal=""
  if debug_frp_digits "${HOSTNAME##*-}" >/dev/null 2>&1; then
    pod_ordinal="$(debug_frp_digits "${HOSTNAME##*-}")"
  elif debug_frp_digits "${POD_NAME##*-}" >/dev/null 2>&1; then
    pod_ordinal="$(debug_frp_digits "${POD_NAME##*-}")"
  else
    pod_ordinal="0"
  fi

  base_port="${JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT_BASE:-30088}"
  printf '%s' "$((base_port + pod_ordinal))"
}

start_debug_frp() {
  if ! is_truthy "${JETLINKS_AGENT_DEBUG_FRP_ENABLED:-0}" && [ -z "${JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT:-}" ]; then
    echo "debug frp disabled by JETLINKS_AGENT_DEBUG_FRP_ENABLED"
    return 0
  fi

  authorized_keys_file="${JETLINKS_AGENT_DEBUG_AUTHORIZED_KEYS_FILE:-/etc/debug-frp/authorized_keys}"
  frpc_config_file="${JETLINKS_AGENT_DEBUG_FRPC_CONFIG_FILE:-/etc/frp/frpc.ini}"
  debug_run_dir="${JETLINKS_AGENT_DEBUG_RUN_DIR:-/run/sshd}"
  debug_root_ssh_dir="${JETLINKS_AGENT_DEBUG_ROOT_SSH_DIR:-/root/.ssh}"
  debug_sshd_config_dir="${JETLINKS_AGENT_DEBUG_SSHD_CONFIG_DIR:-/etc/ssh/sshd_config.d}"
  debug_frp_config_dir="${JETLINKS_AGENT_DEBUG_FRP_CONFIG_DIR:-/etc/frp}"
  debug_sshd_bin="${JETLINKS_AGENT_DEBUG_SSHD_BIN:-/usr/sbin/sshd}"
  debug_frpc_bin="${JETLINKS_AGENT_DEBUG_FRPC_BIN:-/usr/local/bin/frpc}"
  if [ ! -s "${authorized_keys_file}" ]; then
    echo "debug frp skipped: missing authorized_keys file ${authorized_keys_file}"
    return 0
  fi

  mkdir -p "${debug_run_dir}" "${debug_root_ssh_dir}" "${debug_frp_config_dir}" "${debug_sshd_config_dir}"
  chmod 700 "${debug_root_ssh_dir}"
  cp "${authorized_keys_file}" "${debug_root_ssh_dir}/authorized_keys"
  chmod 600 "${debug_root_ssh_dir}/authorized_keys"
  ssh-keygen -A >/dev/null 2>&1 || true

  cat >"${debug_sshd_config_dir}/99-jetlinks-agent-debug.conf" <<'EOF'
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AllowTcpForwarding yes
X11Forwarding no
EOF

  if ! "${debug_sshd_bin}" -e; then
    echo "debug sshd failed to start; continuing normal service startup"
    return 0
  fi
  echo "debug sshd started on 127.0.0.1:22"

  remote_port="$(resolve_debug_frp_remote_port)"
  proxy_name="${JETLINKS_AGENT_DEBUG_FRP_PROXY_NAME:-ai-agent-debug-ssh-${HOSTNAME:-pod}-${remote_port}}"
  if [ ! -s "${frpc_config_file}" ]; then
    server_addr="${JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR:-}"
    server_port="${JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT:-}"
    token="${JETLINKS_AGENT_DEBUG_FRP_TOKEN:-}"
    if [ -z "${server_addr}" ] || [ -z "${server_port}" ] || [ -z "${token}" ]; then
      echo "debug frp skipped: missing ${frpc_config_file} or JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR/JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT/JETLINKS_AGENT_DEBUG_FRP_TOKEN"
      return 0
    fi
    cat >"${frpc_config_file}" <<EOF
[common]
server_addr = ${server_addr}
server_port = ${server_port}
token = ${token}

[${proxy_name}]
type = tcp
local_ip = 127.0.0.1
local_port = 22
remote_port = ${remote_port}
EOF
    chmod 0400 "${frpc_config_file}"
  fi

  "${debug_frpc_bin}" -c "${frpc_config_file}" &
  echo "debug frpc started config=${frpc_config_file} proxy_name=${proxy_name} remote_port=${remote_port}"
}

cpu_count() {
  if [ -n "${APP_WORKERS_AUTO_CPU_COUNT:-}" ]; then
    printf '%s' "${APP_WORKERS_AUTO_CPU_COUNT}"
    return 0
  fi
  if command -v getconf >/dev/null 2>&1; then
    count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    if [ -n "${count}" ]; then
      printf '%s' "${count}"
      return 0
    fi
  fi
  if command -v nproc >/dev/null 2>&1; then
    count="$(nproc 2>/dev/null || true)"
    if [ -n "${count}" ]; then
      printf '%s' "${count}"
      return 0
    fi
  fi
  printf '1'
}

resolve_app_workers() {
  raw="${APP_WORKERS:-auto}"
  case "${raw}" in
    ""|auto|AUTO|Auto)
      count="$(cpu_count)"
      max_workers="${APP_WORKERS_AUTO_MAX:-4}"
      case "${count}" in
        ''|*[!0-9]*)
          count=1
          ;;
      esac
      case "${max_workers}" in
        ''|*[!0-9]*)
          max_workers=4
          ;;
      esac
      if [ "${count}" -lt 1 ]; then
        count=1
      fi
      if [ "${max_workers}" -lt 1 ]; then
        max_workers=1
      fi
      if [ "${count}" -gt "${max_workers}" ]; then
        count="${max_workers}"
      fi
      printf '%s' "${count}"
      ;;
    *)
      printf '%s' "${raw}"
      ;;
  esac
}

should_start_default_server() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi
  if [ "${1:-}" = "server" ]; then
    return 0
  fi
  if [ "$#" -ge 6 ] \
    && [ "${1:-}" = "python" ] \
    && [ "${2:-}" = "-m" ] \
    && [ "${3:-}" = "uvicorn" ] \
    && [ "${4:-}" = "app.main:app" ] \
    && [ "${5:-}" = "--host" ] \
    && [ "${6:-}" = "0.0.0.0" ]; then
    return 0
  fi
  return 1
}

start_default_server() {
  resolved_workers="$(resolve_app_workers)"
  export APP_WORKERS="${resolved_workers}"
  echo "starting uvicorn module=${APP_MODULE} host=${APP_HOST} port=${APP_PORT} workers=${APP_WORKERS} log_level=${APP_LOG_LEVEL}"
  exec python -m uvicorn "${APP_MODULE}" --host "${APP_HOST}" --port "${APP_PORT}" --workers "${APP_WORKERS}" --log-level "${APP_LOG_LEVEL}"
}

start_debug_frp

if should_start_default_server "$@"; then
  start_default_server
fi

exec "$@"
