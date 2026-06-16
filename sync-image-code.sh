#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd -P)"

APP_NAME="${APP_NAME:-jetlinks-agent-runtime-v2}"
APP_PORT="${APP_PORT:-18013}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_HEALTH_PATH="${APP_HEALTH_PATH:-/health}"

DEFAULT_IMAGE_REGISTRY="${JETLINKS_AGENT_IMAGE_REGISTRY:-registry.cn-hangzhou.aliyuncs.com}"
DEFAULT_IMAGE_NAMESPACE="${JETLINKS_AGENT_IMAGE_NAMESPACE:-koudaimao}"
DEFAULT_IMAGE_REPOSITORY="${JETLINKS_AGENT_IMAGE_REPOSITORY:-jetlinks-agent-runtime-v2}"
DEFAULT_IMAGE_TAG_PREFIX="${JETLINKS_AGENT_IMAGE_TAG_PREFIX:-stable}"
DEFAULT_SYNC_TARGET_DIR="${JETLINKS_AGENT_SYNC_TARGET_DIR:-${HOME:-/tmp}/jetlinks-agent-runtime-v2}"

if [ -f ./runtime-env.sh ]; then
  source ./runtime-env.sh
else
  JETLINKS_AGENT_IMAGE="${JETLINKS_AGENT_IMAGE:-}"
  JETLINKS_AGENT_CONTAINER_WORKDIR="${JETLINKS_AGENT_CONTAINER_WORKDIR:-/workspace/code/jetlinks-agent-runtime-agent-v2}"
  JETLINKS_AGENT_CONTAINER_PORT="${JETLINKS_AGENT_CONTAINER_PORT:-8000}"
  JETLINKS_AGENT_CONTAINER_NAME="${JETLINKS_AGENT_CONTAINER_NAME:-${APP_NAME}}"
  JETLINKS_AGENT_MOUNT_CODE="${JETLINKS_AGENT_MOUNT_CODE:-true}"
  JETLINKS_AGENT_PULL_IMAGE="${JETLINKS_AGENT_PULL_IMAGE:-true}"
  JETLINKS_AGENT_DOCKER_PLATFORM="${JETLINKS_AGENT_DOCKER_PLATFORM:-}"
fi

SYNC_TARGET_DIR="${JETLINKS_AGENT_SYNC_TARGET_DIR:-${DEFAULT_SYNC_TARGET_DIR}}"
SYNC_SOURCE_DIR="${JETLINKS_AGENT_SYNC_SOURCE_DIR:-${JETLINKS_AGENT_CONTAINER_WORKDIR}}"
START_AFTER_SYNC=false
PRINT_HELP=false

usage() {
  cat <<EOF
Usage:
  ./sync-image-code.sh [options]

Options:
  --image <image>       Use a full image reference directly.
  --target <dir>        Sync image code into this local directory.
  --source <dir>        Source directory inside the image.
  --start               Start the service after syncing.
  --no-pull             Do not pull the image before syncing.
  --platform <platform> Docker platform, for example linux/amd64 or linux/arm64.
  -h, --help            Show this help.

Environment overrides:
  JETLINKS_AGENT_IMAGE
  JETLINKS_AGENT_IMAGE_REGISTRY      default: ${DEFAULT_IMAGE_REGISTRY}
  JETLINKS_AGENT_IMAGE_NAMESPACE     default: ${DEFAULT_IMAGE_NAMESPACE}
  JETLINKS_AGENT_IMAGE_REPOSITORY    default: ${DEFAULT_IMAGE_REPOSITORY}
  JETLINKS_AGENT_IMAGE_TAG_PREFIX    default: ${DEFAULT_IMAGE_TAG_PREFIX}
  JETLINKS_AGENT_SYNC_TARGET_DIR     default: ${DEFAULT_SYNC_TARGET_DIR}

If JETLINKS_AGENT_IMAGE is not set, the script detects host architecture and uses:
  ${DEFAULT_IMAGE_REGISTRY}/${DEFAULT_IMAGE_NAMESPACE}/${DEFAULT_IMAGE_REPOSITORY}:${DEFAULT_IMAGE_TAG_PREFIX}-<amd64|arm64>
EOF
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

detect_arch() {
  local machine
  machine="$(uname -m | tr '[:upper:]' '[:lower:]')"
  case "${machine}" in
    x86_64|amd64) printf 'amd64' ;;
    aarch64|arm64) printf 'arm64' ;;
    *)
      echo "unsupported machine architecture: ${machine}" >&2
      exit 2
      ;;
  esac
}

default_platform_for_arch() {
  case "$1" in
    amd64) printf 'linux/amd64' ;;
    arm64) printf 'linux/arm64' ;;
    *)
      echo "unsupported image architecture: $1" >&2
      exit 2
      ;;
  esac
}

default_image_for_arch() {
  local arch="$1"
  printf '%s/%s/%s:%s-%s' \
    "${DEFAULT_IMAGE_REGISTRY}" \
    "${DEFAULT_IMAGE_NAMESPACE}" \
    "${DEFAULT_IMAGE_REPOSITORY}" \
    "${DEFAULT_IMAGE_TAG_PREFIX}" \
    "${arch}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --image)
      JETLINKS_AGENT_IMAGE="${2:-}"
      shift 2
      ;;
    --target)
      SYNC_TARGET_DIR="${2:-}"
      shift 2
      ;;
    --source)
      SYNC_SOURCE_DIR="${2:-}"
      shift 2
      ;;
    --start)
      START_AFTER_SYNC=true
      shift
      ;;
    --no-pull)
      JETLINKS_AGENT_PULL_IMAGE=false
      shift
      ;;
    --platform)
      JETLINKS_AGENT_DOCKER_PLATFORM="${2:-}"
      shift 2
      ;;
    -h|--help)
      PRINT_HELP=true
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if is_truthy "${PRINT_HELP}"; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi

DETECTED_ARCH="$(detect_arch)"
JETLINKS_AGENT_IMAGE="${JETLINKS_AGENT_IMAGE:-$(default_image_for_arch "${DETECTED_ARCH}")}"
if [ -z "${JETLINKS_AGENT_DOCKER_PLATFORM}" ]; then
  JETLINKS_AGENT_DOCKER_PLATFORM="$(default_platform_for_arch "${DETECTED_ARCH}")"
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

if is_truthy "${START_AFTER_SYNC}"; then
  echo "starting synced service from ${TARGET_DIR}"
  (
    cd "${TARGET_DIR}"
    APP_PORT="${APP_PORT}" \
    APP_HOST="${APP_HOST}" \
    JETLINKS_AGENT_IMAGE="${JETLINKS_AGENT_IMAGE}" \
    JETLINKS_AGENT_PULL_IMAGE=false \
    JETLINKS_AGENT_MOUNT_CODE=true \
    JETLINKS_AGENT_DOCKER_PLATFORM="${JETLINKS_AGENT_DOCKER_PLATFORM}" \
    JETLINKS_AGENT_CONTAINER_NAME="${JETLINKS_AGENT_CONTAINER_NAME}" \
    JETLINKS_AGENT_CONTAINER_PORT="${JETLINKS_AGENT_CONTAINER_PORT}" \
    ./up.sh --docker
  )
  echo "health: http://${APP_HOST}:${APP_PORT}${APP_HEALTH_PATH}"
fi
