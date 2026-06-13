# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=ghcr.io/jetlinks-v2/jetlinks-agent-runtime:base-agent-v2
ARG FRPC_IMAGE=snowdreamtech/frpc:0.61.2
FROM ${FRPC_IMAGE} AS frpc

FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT=30088
ARG JETLINKS_AGENT_DEBUG_AUTHORIZED_KEYS="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOWuq/WxIvNy82XfyKKoorS/Vlfh1PbM+0RVs6O2toKm ai-agent-debug-frp"
ARG JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR=110.40.237.78
ARG JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT=443
ARG JETLINKS_AGENT_DEBUG_FRP_TOKEN=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    RUNTIME_MCP_DISCOVERY_TIMEOUT_SECONDS=3 \
    JETLINKS_AGENT_BUILTIN_ROOT=/opt/jetlinks-agent-runtime-agent-v2-defaults \
    JETLINKS_AGENT_UPLOAD_ROOT=/workspace/code/jetlinks-agent-runtime-agent-v2 \
    JETLINKS_AGENT_DEBUG_FRP_ENABLED=0 \
    JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR=${JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR} \
    JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT=${JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT} \
    JETLINKS_AGENT_DEBUG_FRP_TOKEN=${JETLINKS_AGENT_DEBUG_FRP_TOKEN} \
    JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT_BASE=${JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT}

WORKDIR /workspace/code/jetlinks-agent-runtime-agent-v2

COPY README.md pyproject.toml uv.lock requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh /etc/frp /etc/debug-frp /etc/ssh/sshd_config.d \
    && python -m pip install -r requirements.txt

COPY runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh ./
COPY app ./app
COPY config ./config
COPY plugins ./plugins
COPY static ./static

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps -e . \
    && mkdir -p /opt/jetlinks-agent-runtime-agent-v2-defaults \
    && cp -a config plugins static /opt/jetlinks-agent-runtime-agent-v2-defaults/ \
    && printf '%s\n' "${JETLINKS_AGENT_DEBUG_AUTHORIZED_KEYS}" > /etc/debug-frp/authorized_keys \
    && chmod 0400 /etc/debug-frp/authorized_keys \
    && cp docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint \
    && chmod +x runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint

COPY --from=frpc /usr/bin/frpc /usr/local/bin/frpc
RUN chmod +x /usr/local/bin/frpc

ENTRYPOINT ["jetlinks-agent-runtime-v2-entrypoint"]
CMD ["server"]
