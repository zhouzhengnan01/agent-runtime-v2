from __future__ import annotations

import os
import subprocess
from pathlib import Path
import shlex


def test_docker_entrypoint_auto_workers_uses_cpu_count_cap(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf 'python %s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "APP_WORKERS": "auto",
            "APP_WORKERS_AUTO_CPU_COUNT": "6",
            "APP_WORKERS_AUTO_MAX": "4",
            "JETLINKS_AGENT_DEBUG_FRP_ENABLED": "0",
            "JETLINKS_AGENT_CONTAINER_WORKDIR": str(tmp_path / "workdir"),
            "JETLINKS_AGENT_USER_ROOT": str(tmp_path / "user-root"),
        }
    )

    completed = subprocess.run(
        ["sh", "docker-entrypoint.sh", "server"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    output = completed.stdout + completed.stderr
    assert "starting uvicorn module=app.main:app host=0.0.0.0 port=8000 workers=4 log_level=info" in output
    assert "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info" in output


def test_up_sh_docker_mode_uses_entrypoint_server_command(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)

    docker_calls = tmp_path / "docker-calls.log"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "log_path = Path(os.environ['FAKE_DOCKER_CALLS'])\n"
        "with log_path.open('a', encoding='utf-8') as fh:\n"
        "    fh.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "cmd = sys.argv[1:]\n"
        "if cmd[:2] == ['ps', '-a']:\n"
        "    sys.exit(0)\n"
        "if cmd[:1] == ['ps']:\n"
        "    print(os.environ.get('JETLINKS_AGENT_CONTAINER_NAME', 'jetlinks-agent-runtime-v2'))\n"
        "    sys.exit(0)\n"
        "if cmd[:1] == ['pull']:\n"
        "    sys.exit(0)\n"
        "if cmd[:2] == ['run', '-d']:\n"
        "    print('fake-container-id')\n"
        "    sys.exit(0)\n"
        "if cmd[:2] == ['logs', '--tail']:\n"
        "    sys.exit(0)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    fake_curl = fake_bin / "curl"
    fake_curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_curl.chmod(0o755)

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "FAKE_DOCKER_CALLS": str(docker_calls),
            "JETLINKS_AGENT_IMAGE": "example.com/jetlinks-agent:test",
            "JETLINKS_AGENT_PULL_IMAGE": "false",
            "JETLINKS_AGENT_MOUNT_CODE": "false",
            "APP_WORKERS": "auto",
            "APP_WORKERS_AUTO_MAX": "3",
            "JETLINKS_AGENT_CONTAINER_NAME": "jetlinks-agent-runtime-v2",
        }
    )

    completed = subprocess.run(
        ["bash", "up.sh", "--docker"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "started container=jetlinks-agent-runtime-v2" in (completed.stdout + completed.stderr)

    calls = docker_calls.read_text(encoding="utf-8").splitlines()
    run_call = next(line for line in calls if line.startswith("run -d "))
    tokens = shlex.split(run_call)
    assert "server" in tokens
    assert "sh" not in tokens
    assert "python" not in tokens
