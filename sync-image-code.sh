#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd -P)"
if [ -f ./runtime-env.sh ]; then
  source ./runtime-env.sh
else
  JETLINKS_AGENT_IMAGE="${JETLINKS_AGENT_IMAGE:-}"
  JETLINKS_AGENT_CONTAINER_WORKDIR="${JETLINKS_AGENT_CONTAINER_WORKDIR:-/workspace/code/jetlinks-agent-runtime-agent-v2}"
  JETLINKS_AGENT_PULL_IMAGE="${JETLINKS_AGENT_PULL_IMAGE:-true}"
  JETLINKS_AGENT_DOCKER_PLATFORM="${JETLINKS_AGENT_DOCKER_PLATFORM:-}"
fi

SYNC_TARGET_DIR="${JETLINKS_AGENT_SYNC_TARGET_DIR:-${PROJECT_DIR}}"
SYNC_SOURCE_DIR="${JETLINKS_AGENT_SYNC_SOURCE_DIR:-${JETLINKS_AGENT_CONTAINER_WORKDIR}}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [ -z "${JETLINKS_AGENT_IMAGE}" ]; then
  echo "JETLINKS_AGENT_IMAGE is required" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
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

mkdir -p "${SYNC_TARGET_DIR}"
TARGET_DIR="$(cd "${SYNC_TARGET_DIR}" && pwd -P)"

if [ "${TARGET_DIR}" = "/" ]; then
  echo "refusing to sync image code into /" >&2
  exit 2
fi

TEMP_CONTAINER="jetlinks-agent-image-sync-$RANDOM-$$"
create_args=(docker create --name "${TEMP_CONTAINER}")
if [ -n "${JETLINKS_AGENT_DOCKER_PLATFORM}" ]; then
  create_args+=(--platform "${JETLINKS_AGENT_DOCKER_PLATFORM}")
fi
create_args+=("${JETLINKS_AGENT_IMAGE}" sh -c "test -d '${SYNC_SOURCE_DIR}'")

cleanup() {
  docker rm -f "${TEMP_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${create_args[@]}" >/dev/null
if ! docker start -a "${TEMP_CONTAINER}" >/dev/null; then
  echo "image source directory not found: ${SYNC_SOURCE_DIR}" >&2
  exit 1
fi

echo "syncing image code image=${JETLINKS_AGENT_IMAGE} source=${SYNC_SOURCE_DIR} target=${TARGET_DIR}"
docker cp "${TEMP_CONTAINER}:${SYNC_SOURCE_DIR}/." "${TARGET_DIR}"
echo "synced image code to ${TARGET_DIR}"
