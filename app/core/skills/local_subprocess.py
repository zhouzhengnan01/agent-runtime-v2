from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


DEFAULT_MAX_ENVS = 20


@dataclass(frozen=True)
class LocalSubprocessEnvironment:
    requirements_hash: str | None
    python: Path
    status: str
    message: str = ""


class LocalSubprocessEnvironmentCache:
    """Prepare per-skill virtualenvs for trusted local subprocess execution."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.cache_dir = self.root_dir / ".runtime" / "skill-venvs"
        self.index_path = self.cache_dir / "index.json"

    def prepare(
        self,
        *,
        skill_name: str,
        package_root: Path | None,
        base_python: str | None = None,
        install_timeout_seconds: int = 1800,
    ) -> LocalSubprocessEnvironment:
        python = base_python or sys.executable
        requirements_text = self.requirements_text(package_root)
        if not requirements_text:
            return LocalSubprocessEnvironment(requirements_hash=None, python=Path(python), status="base")

        identity = f"# python: {self._python_identity(python)}\n{requirements_text}"
        requirements_hash = requirements_hash_for_text(identity)
        env_dir = self.cache_dir / self._safe_skill_name(skill_name) / requirements_hash[:16]
        env_python = self._env_python(env_dir)
        index = self._read_index()
        entry = index.get(requirements_hash)
        if (
            isinstance(entry, dict)
            and entry.get("status") == "ready"
            and entry.get("skill_name") == skill_name
            and env_python.is_file()
        ):
            entry["last_used_at"] = time.time()
            self._write_index(index)
            return LocalSubprocessEnvironment(requirements_hash=requirements_hash, python=env_python, status="ready")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        index[requirements_hash] = {
            "status": "warming",
            "skill_name": skill_name,
            "env_dir": str(env_dir),
            "python": str(env_python),
            "requirements_hash": requirements_hash,
            "created_at": time.time(),
            "last_used_at": time.time(),
        }
        self._write_index(index)
        try:
            self._build_env(
                env_dir=env_dir,
                python=python,
                requirements_text=requirements_text,
                timeout=install_timeout_seconds,
            )
        except Exception as exc:
            index = self._read_index()
            index[requirements_hash] = {
                **(index.get(requirements_hash) if isinstance(index.get(requirements_hash), dict) else {}),
                "status": "failed",
                "message": str(exc),
                "last_used_at": time.time(),
            }
            self._write_index(index)
            return LocalSubprocessEnvironment(
                requirements_hash=requirements_hash,
                python=Path(python),
                status="failed",
                message=str(exc),
            )

        index = self._read_index()
        index[requirements_hash] = {
            **(index.get(requirements_hash) if isinstance(index.get(requirements_hash), dict) else {}),
            "status": "ready",
            "python": str(env_python),
            "last_used_at": time.time(),
        }
        self._write_index(index)
        self.prune()
        return LocalSubprocessEnvironment(requirements_hash=requirements_hash, python=env_python, status="ready")

    def status(self, *, skill_name: str, package_root: Path | None, base_python: str | None = None) -> dict[str, object]:
        requirements_text = self.requirements_text(package_root)
        if not requirements_text:
            return {"status": "base", "python": base_python or sys.executable}
        requirements_hash = requirements_hash_for_text(f"# python: {self._python_identity(base_python or sys.executable)}\n{requirements_text}")
        entry = self._read_index().get(requirements_hash)
        if isinstance(entry, dict):
            return entry
        return {"status": "pending", "requirements_hash": requirements_hash, "python": None}

    def prune(self, *, max_envs: int = DEFAULT_MAX_ENVS) -> None:
        index = self._read_index()
        ready = [
            (key, entry)
            for key, entry in index.items()
            if isinstance(entry, dict) and entry.get("status") == "ready" and isinstance(entry.get("env_dir"), str)
        ]
        if len(ready) <= max_envs:
            return
        ready.sort(key=lambda item: float(item[1].get("last_used_at") or 0))
        for key, entry in ready[: max(0, len(ready) - max_envs)]:
            shutil.rmtree(str(entry["env_dir"]), ignore_errors=True)
            index[key] = {**entry, "status": "stale"}
        self._write_index(index)

    @staticmethod
    def requirements_text(package_root: Path | None) -> str:
        if package_root is None:
            return ""
        path = package_root / "requirements.txt"
        if not path.is_file():
            return ""
        return normalized_requirements(path.read_text(encoding="utf-8"))

    def _build_env(self, *, env_dir: Path, python: str, requirements_text: str, timeout: int) -> None:
        if env_dir.exists():
            shutil.rmtree(env_dir)
        subprocess.run([python, "-m", "venv", str(env_dir)], text=True, capture_output=True, timeout=timeout, check=True)
        env_python = self._env_python(env_dir)
        subprocess.run([str(env_python), "-m", "ensurepip", "--upgrade"], text=True, capture_output=True, timeout=timeout, check=True)
        requirements_path = env_dir / "requirements.txt"
        requirements_path.write_text(requirements_text + "\n", encoding="utf-8")
        install = subprocess.run(
            [str(env_python), "-m", "pip", "install", "-r", str(requirements_path)],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        (env_dir / "install.log").write_text((install.stdout or "") + (install.stderr or ""), encoding="utf-8")
        if install.returncode != 0:
            raise RuntimeError((install.stderr or install.stdout or "pip install failed").strip())

    @staticmethod
    def _env_python(env_dir: Path) -> Path:
        if sys.platform == "win32":
            return env_dir / "Scripts" / "python.exe"
        return env_dir / "bin" / "python"

    @staticmethod
    def _safe_skill_name(skill_name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in skill_name.strip())
        return cleaned.strip(".-")[:96] or "skill"

    @staticmethod
    def _python_identity(python: str) -> str:
        try:
            result = subprocess.run(
                [python, "-c", "import sys; print(sys.executable + '|' + sys.version.split()[0])"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            return result.stdout.strip() or python
        except Exception:
            return python

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_index(self, data: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.index_path.with_suffix(".json.tmp")
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.replace(self.index_path)


def normalized_requirements(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if " #" in stripped:
            stripped = stripped.split(" #", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(sorted(dict.fromkeys(lines)))


def requirements_hash_for_text(text: str) -> str:
    return hashlib.sha256(normalized_requirements(text).encode("utf-8")).hexdigest()
