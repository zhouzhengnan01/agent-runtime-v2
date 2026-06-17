# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG APT_DEBIAN_MIRROR=https://mirrors.aliyun.com/debian
ARG APT_DEBIAN_SECURITY_MIRROR=https://mirrors.aliyun.com/debian-security
ARG FRP_VERSION=0.61.1
ARG JETLINKS_AGENT_DEBUG_AUTHORIZED_KEYS=""
ARG JETLINKS_AGENT_DEBUG_FRP_ENABLED=0
ARG JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT_BASE=30088
ARG JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT=""
ARG JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR=""
ARG JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT=""
ARG JETLINKS_AGENT_DEBUG_FRP_TOKEN=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    RUNTIME_MCP_DISCOVERY_TIMEOUT_SECONDS=3 \
    APP_PORT=8000 \
    JETLINKS_AGENT_BUILTIN_ROOT=/opt/jetlinks-agent-runtime-agent-v2-defaults \
    JETLINKS_AGENT_UPLOAD_ROOT=/workspace/code/jetlinks-agent-runtime-agent-v2 \
    JETLINKS_AGENT_DEBUG_FRP_ENABLED=${JETLINKS_AGENT_DEBUG_FRP_ENABLED} \
    JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT_BASE=${JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT_BASE} \
    JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT=${JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT} \
    JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR=${JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR} \
    JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT=${JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT} \
    JETLINKS_AGENT_DEBUG_FRP_TOKEN=${JETLINKS_AGENT_DEBUG_FRP_TOKEN}

WORKDIR /workspace/code/jetlinks-agent-runtime-agent-v2

COPY README.md pyproject.toml ./
COPY runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh ./
COPY app ./app
COPY config ./config
COPY plugins ./plugins
COPY static ./static

RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
      if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s#http://deb.debian.org/debian#${APT_DEBIAN_MIRROR}#g; s#http://security.debian.org/debian-security#${APT_DEBIAN_SECURITY_MIRROR}#g" /etc/apt/sources.list.d/debian.sources; \
      elif [ -f /etc/apt/sources.list ]; then \
        sed -i "s#http://deb.debian.org/debian#${APT_DEBIAN_MIRROR}#g; s#http://security.debian.org/debian-security#${APT_DEBIAN_SECURITY_MIRROR}#g" /etc/apt/sources.list; \
      fi; \
      apt-get update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        openssh-server; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    python -m pip install --no-cache-dir --upgrade pip; \
    python -m pip install --no-cache-dir -e . imageio-ffmpeg; \
    python -c "from pathlib import Path; import imageio_ffmpeg; target = Path('/usr/local/bin/ffmpeg'); target.unlink(missing_ok=True); target.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())"; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64|amd64) frp_arch="amd64" ;; \
      aarch64|arm64) frp_arch="arm64" ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    if ! command -v frpc >/dev/null 2>&1; then \
      curl -fsSL "https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_linux_${frp_arch}.tar.gz" -o /tmp/frp.tar.gz; \
      tar -xzf /tmp/frp.tar.gz -C /tmp; \
      cp "/tmp/frp_${FRP_VERSION}_linux_${frp_arch}/frpc" /usr/local/bin/frpc; \
      chmod +x /usr/local/bin/frpc; \
      rm -rf /tmp/frp.tar.gz "/tmp/frp_${FRP_VERSION}_linux_${frp_arch}"; \
    fi; \
    mkdir -p /opt/jetlinks-agent-runtime-agent-v2-defaults; \
    cp -a config plugins static /opt/jetlinks-agent-runtime-agent-v2-defaults/; \
    cp docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint; \
    if [ -n "${JETLINKS_AGENT_DEBUG_AUTHORIZED_KEYS}" ]; then \
      mkdir -p /etc/debug-frp; \
      printf '%s\n' "${JETLINKS_AGENT_DEBUG_AUTHORIZED_KEYS}" > /etc/debug-frp/authorized_keys; \
      chmod 0400 /etc/debug-frp/authorized_keys; \
    fi; \
    chmod +x runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint

ENTRYPOINT ["jetlinks-agent-runtime-v2-entrypoint"]
CMD ["serve"]
