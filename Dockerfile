ARG BASE_IMAGE=python:3.12-slim
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

COPY README.md pyproject.toml ./
COPY runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh ./
COPY app ./app
COPY config ./config
COPY plugins ./plugins
COPY static ./static

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e . imageio-ffmpeg \
    && python -c "from pathlib import Path; import imageio_ffmpeg; target = Path('/usr/local/bin/ffmpeg'); target.unlink(missing_ok=True); target.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())" \
    && mkdir -p /opt/jetlinks-agent-runtime-agent-v2-defaults \
    && cp -a config plugins static /opt/jetlinks-agent-runtime-agent-v2-defaults/ \
    && cp docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint \
    && chmod +x runtime-env.sh up.sh status.sh stop.sh sync-image-code.sh docker-entrypoint.sh /usr/local/bin/jetlinks-agent-runtime-v2-entrypoint

ENTRYPOINT ["jetlinks-agent-runtime-v2-entrypoint"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]
