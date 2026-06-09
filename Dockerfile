# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=ghcr.io/jetlinks-v2/jetlinks-agent-runtime:base-agent-v2
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    RUNTIME_MCP_DISCOVERY_TIMEOUT_SECONDS=3 \
    JETLINKS_AGENT_BUILTIN_ROOT=/opt/jetlinks-agent-runtime-agent-v2-defaults \
    JETLINKS_AGENT_UPLOAD_ROOT=/workspace/code/jetlinks-agent-runtime-agent-v2

WORKDIR /workspace/code/jetlinks-agent-runtime-agent-v2

COPY README.md pyproject.toml uv.lock ./
COPY runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh ./
COPY app ./app
COPY config ./config
COPY plugins ./plugins
COPY static ./static

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps -e . \
    && mkdir -p /opt/jetlinks-agent-runtime-agent-v2-defaults \
    && cp -a config plugins static /opt/jetlinks-agent-runtime-agent-v2-defaults/ \
    && cp docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint \
    && chmod +x runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint

ENTRYPOINT ["jetlinks-agent-runtime-v2-entrypoint"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
