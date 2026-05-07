# syntax=docker/dockerfile:1
ARG OPENSANDBOX_BASE=sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.2
FROM ${OPENSANDBOX_BASE}

USER root

RUN if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && \
      apt-get install -y --no-install-recommends libreoffice fonts-noto-cjk && \
      rm -rf /var/lib/apt/lists/*; \
    fi

RUN python -m pip install --no-cache-dir \
      "python-pptx>=1.0.2" \
      "openpyxl>=3.1.5" \
      "xlsxwriter>=3.2.0" \
      "pillow>=11.0.0"

RUN mkdir -p /opt/jetlinks/skills
COPY sandbox/skills/run_skill.py /opt/jetlinks/skills/run_skill.py
RUN chmod +x /opt/jetlinks/skills/run_skill.py
