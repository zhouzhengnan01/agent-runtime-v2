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


def test_docker_entrypoint_explicit_debug_frp_port_enables_tunnel(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    fake_sbin = tmp_path / "sbin"
    fake_sbin.mkdir(parents=True)
    fake_etc = tmp_path / "etc"
    fake_debug_frp = fake_etc / "debug-frp"
    fake_frp = fake_etc / "frp"
    fake_debug_frp.mkdir(parents=True)
    fake_frp.mkdir(parents=True)
    (fake_debug_frp / "authorized_keys").write_text("ssh-ed25519 test\n", encoding="utf-8")

    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nprintf 'python %s\\n' \"$*\"\n", encoding="utf-8")
    fake_python.chmod(0o755)

    fake_ssh_keygen = fake_bin / "ssh-keygen"
    fake_ssh_keygen.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ssh_keygen.chmod(0o755)

    fake_sshd = fake_sbin / "sshd"
    fake_sshd.write_text("#!/bin/sh\necho sshd \"$@\"\nexit 0\n", encoding="utf-8")
    fake_sshd.chmod(0o755)

    fake_frpc = fake_bin / "frpc"
    fake_frpc.write_text("#!/bin/sh\necho frpc \"$@\"\nexit 0\n", encoding="utf-8")
    fake_frpc.chmod(0o755)

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "JETLINKS_AGENT_CONTAINER_WORKDIR": str(tmp_path / "workdir"),
            "JETLINKS_AGENT_USER_ROOT": str(tmp_path / "user-root"),
            "JETLINKS_AGENT_DEBUG_AUTHORIZED_KEYS_FILE": str(fake_debug_frp / "authorized_keys"),
            "JETLINKS_AGENT_DEBUG_FRPC_CONFIG_FILE": str(fake_frp / "frpc.ini"),
            "JETLINKS_AGENT_DEBUG_RUN_DIR": str(tmp_path / "run" / "sshd"),
            "JETLINKS_AGENT_DEBUG_ROOT_SSH_DIR": str(tmp_path / "root" / ".ssh"),
            "JETLINKS_AGENT_DEBUG_SSHD_CONFIG_DIR": str(fake_etc / "ssh" / "sshd_config.d"),
            "JETLINKS_AGENT_DEBUG_FRP_CONFIG_DIR": str(fake_frp),
            "JETLINKS_AGENT_DEBUG_SSHD_BIN": str(fake_sshd),
            "JETLINKS_AGENT_DEBUG_FRPC_BIN": str(fake_frpc),
            "JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT": "30191",
            "JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR": "110.40.237.78",
            "JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT": "443",
            "JETLINKS_AGENT_DEBUG_FRP_TOKEN": "test-token",
            "APP_WORKERS": "1",
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
    assert "debug sshd started on 127.0.0.1:22" in output
    assert "debug frpc started" in output
    generated_config = (fake_frp / "frpc.ini").read_text(encoding="utf-8")
    assert "remote_port = 30191" in generated_config
    assert "local_port = 22" in generated_config


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
