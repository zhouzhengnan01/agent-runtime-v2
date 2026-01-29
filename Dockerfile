FROM python:3.10-slim

WORKDIR /app

# Keep build lean: prefer wheels, avoid apt-get.
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir \
    --progress-bar off \
    aiofiles==24.1.0 \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.11.9 \
    pydantic-settings==2.10.1 \
    sqlalchemy==2.0.23 \
    pymysql==1.1.1 \
    psycopg2-binary==2.9.9 \
    aiomysql==0.2.0 \
    redis==5.2.0 \
    pymilvus==2.6.2 \
    python-multipart==0.0.20 \
    python-dotenv==1.0.1 \
    websockets==15.0.1 \
    openai==1.109.1 \
    langchain==0.3.27 \
    langchain-community==0.3.31 \
    langchain-openai==0.3.35 \
    dashscope==1.24.5 \
    opencv-python-headless==4.12.0.88 \
    httpx==0.28.1 \
    python-docx==1.1.2 \
    reportlab==4.4.4 \
    pandas==2.2.3

# ffmpeg/ffprobe static bundle shipped with the repo.
COPY third_party/ffmpeg/ffmpeg-release-amd64-static.tar.xz /tmp/ffmpeg.tar.xz

RUN set -e; \
    tmp_script=/tmp/install_ffmpeg.py; \
    printf '%s\n' \
      'import os' \
      'import shutil' \
      'import stat' \
      'import tarfile' \
      '' \
      'archive_path = os.environ.get("FFMPEG_TARBALL_PATH", "/tmp/ffmpeg.tar.xz")' \
      'with tarfile.open(archive_path, mode="r:*") as tf:' \
      '    wanted = {}' \
      '    for member in tf.getmembers():' \
      '        base = os.path.basename(member.name)' \
      '        if base in {"ffmpeg", "ffprobe"} and member.isfile():' \
      '            wanted[base] = member' \
      '' \
      '    missing = {"ffmpeg", "ffprobe"} - set(wanted)' \
      '    if missing:' \
      '        raise RuntimeError(f"ffmpeg bundle missing: {sorted(missing)}")' \
      '' \
      '    for base, member in wanted.items():' \
      '        extracted = tf.extractfile(member)' \
      '        if extracted is None:' \
      '            raise RuntimeError(f"cannot extract: {member.name}")' \
      '        out_path = f"/usr/local/bin/{base}"' \
      '        with open(out_path, "wb") as f:' \
      '            shutil.copyfileobj(extracted, f)' \
      '        os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)' \
      > "$tmp_script"; \
    python "$tmp_script"; \
    rm -f /tmp/ffmpeg.tar.xz; \
    rm -f "$tmp_script"

ENV FFMPEG_BIN=/usr/local/bin/ffmpeg
ENV FFPROBE_BIN=/usr/local/bin/ffprobe

COPY . .

RUN mkdir -p storage/logs storage/uploads storage/cache

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8005/health', timeout=3).read()" || exit 1

CMD ["python", "main.py"]
