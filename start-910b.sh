#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 910B/NPU 训练必须先加载 Ascend/CANN 环境，否则 torch_npu 可能找不到
# libhccl.so、libascendcl.so 等动态库。路径可通过 ASCEND_ENV_SH 覆盖。
ASCEND_ENV_SH="${ASCEND_ENV_SH:-}"
if [ -z "${ASCEND_ENV_SH}" ]; then
  if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    ASCEND_ENV_SH="/usr/local/Ascend/ascend-toolkit/set_env.sh"
  elif [ -f "/usr/local/Ascend/cann-8.5.0/set_env.sh" ]; then
    ASCEND_ENV_SH="/usr/local/Ascend/cann-8.5.0/set_env.sh"
  else
    echo "Ascend set_env.sh not found. Set ASCEND_ENV_SH before starting." >&2
    exit 2
  fi
fi
echo "loading Ascend environment: ${ASCEND_ENV_SH}"
# shellcheck disable=SC1090
source "${ASCEND_ENV_SH}"

# 910B 服务器通常使用 agent-runtime-v2 conda 环境。优先激活 conda，
# 如果 conda 初始化脚本不存在，则直接使用常见环境里的 python。
CONDA_ENV_NAME="${CONDA_ENV_NAME:-agent-runtime-v2}"
if [ -f "/data/nvme0n1/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/data/nvme0n1/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
elif [ -f "/data/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/data/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
elif [ -x "/data/nvme0n1/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]; then
  export PYTHON_BIN="/data/nvme0n1/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
else
  echo "Conda environment '${CONDA_ENV_NAME}' not found. Set PYTHON_BIN or CONDA_ENV_NAME." >&2
  exit 2
fi

export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8069}"
export APP_WORKERS="${APP_WORKERS:-1}"
export APP_LOG_LEVEL="${APP_LOG_LEVEL:-info}"
export JETLINKS_AGENT_RUN_MODE="local"

# NPU 服务通常不建议开 --reload，多进程 reload 会让环境和子进程排查变复杂。
# up.sh 会写入 .runtime/server/server.pid 和 .runtime/server/server.log。
echo "checking torch_npu runtime..."
"${PYTHON_BIN:-python}" - <<'PY'
import torch_npu  # noqa: F401
print("torch_npu ok")
PY

exec ./up.sh --local "$@"
