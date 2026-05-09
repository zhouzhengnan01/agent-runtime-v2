from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.sandbox.config import SandboxConfig
from app.core.sandbox.policy import SandboxProfile


DEFAULT_CACHE_IMAGE_PREFIX = "jetlinks-python-skill-deps"
DEFAULT_MAX_IMAGES = 20
DEFAULT_MAX_BYTES = 20 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class SkillEnvironment:
    requirements_hash: str | None
    image: str
    status: str
    message: str = ""


class SkillEnvironmentCache:
    """Build and reuse sandbox dependency images keyed by normalized requirements."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.cache_dir = self.root_dir / ".runtime" / "skill-envs"
        self.index_path = self.cache_dir / "index.json"

    def prepare(
        self,
        *,
        skill_name: str,
        package_root: Path | None,
        profile: SandboxProfile,
        config: SandboxConfig,
    ) -> SkillEnvironment:
        requirements_text = self.requirements_text(package_root)
        if not requirements_text:
            return SkillEnvironment(requirements_hash=None, image=profile.image, status="base")
        base_image = self._base_image(profile.image, requirements_text, config)
        requirements_hash = requirements_hash_for_text(f"# base-image: {base_image}\n{requirements_text}")
        image = f"{self._image_prefix(config)}:{requirements_hash[:16]}"
        index = self._read_index()
        entry = index.get(requirements_hash)
        if (
            isinstance(entry, dict)
            and entry.get("status") == "ready"
            and entry.get("base_image") == base_image
            and self._image_exists(image)
        ):
            entry["last_used_at"] = time.time()
            entry["skills"] = sorted({*entry.get("skills", []), skill_name})
            self._write_index(index)
            return SkillEnvironment(requirements_hash=requirements_hash, image=image, status="ready")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        index[requirements_hash] = {
            "status": "warming",
            "image": image,
            "base_image": base_image,
            "skills": sorted({*entry.get("skills", [])}) if isinstance(entry, dict) else [skill_name],
            "created_at": time.time(),
            "last_used_at": time.time(),
        }
        self._write_index(index)
        try:
            self._build_image(image=image, base_image=base_image, requirements_text=requirements_text, config=config)
        except Exception as exc:
            index = self._read_index()
            index[requirements_hash] = {
                **(index.get(requirements_hash) if isinstance(index.get(requirements_hash), dict) else {}),
                "status": "failed",
                "image": image,
                "base_image": base_image,
                "message": str(exc),
                "last_used_at": time.time(),
            }
            self._write_index(index)
            return SkillEnvironment(requirements_hash=requirements_hash, image=profile.image, status="failed", message=str(exc))

        index = self._read_index()
        index[requirements_hash] = {
            **(index.get(requirements_hash) if isinstance(index.get(requirements_hash), dict) else {}),
            "status": "ready",
            "image": image,
            "base_image": base_image,
            "requirements_hash": requirements_hash,
            "last_used_at": time.time(),
        }
        self._write_index(index)
        self.prune(config)
        return SkillEnvironment(requirements_hash=requirements_hash, image=image, status="ready")

    def prune(self, config: SandboxConfig) -> None:
        index = self._read_index()
        ready = [
            (key, entry)
            for key, entry in index.items()
            if isinstance(entry, dict) and entry.get("status") == "ready" and isinstance(entry.get("image"), str)
        ]
        max_images = config.skill_env_cache_max_images or DEFAULT_MAX_IMAGES
        max_bytes = config.skill_env_cache_max_bytes or DEFAULT_MAX_BYTES
        total_bytes = 0
        for _key, entry in ready:
            size = self._image_size(str(entry["image"]))
            if size is not None:
                entry["size_bytes"] = size
                total_bytes += size
        if len(ready) <= max_images and total_bytes <= max_bytes:
            self._write_index(index)
            return
        ready.sort(key=lambda item: float(item[1].get("last_used_at") or 0))
        stale: set[str] = set()
        for key, _entry in ready[: max(0, len(ready) - max_images)]:
            stale.add(key)
        for key, entry in ready:
            if total_bytes <= max_bytes:
                break
            stale.add(key)
            total_bytes -= int(entry.get("size_bytes") or 0)
        for key, entry in ready:
            if key not in stale:
                continue
            image = str(entry["image"])
            self._remove_image(image)
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

    def _build_image(
        self,
        *,
        image: str,
        base_image: str,
        requirements_text: str,
        config: SandboxConfig,
    ) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI is not available for skill dependency image build.")
        build_dir = self.cache_dir / "build" / image.replace("/", "_").replace(":", "_")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        try:
            (build_dir / "requirements.txt").write_text(requirements_text + "\n", encoding="utf-8")
            (build_dir / "Dockerfile").write_text(
                "\n".join(
                    [
                        f"FROM {base_image}",
                        "COPY requirements.txt /tmp/skill-requirements.txt",
                        "RUN python -m pip install --no-cache-dir -r /tmp/skill-requirements.txt",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            timeout = config.skill_env_cache_build_timeout_seconds
            result = subprocess.run(
                ["docker", "build", "-t", image, str(build_dir)],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout or "docker build failed").strip())
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    @staticmethod
    def _image_exists(image: str) -> bool:
        if shutil.which("docker") is None:
            return False
        result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, check=False)
        return result.returncode == 0

    @staticmethod
    def _remove_image(image: str) -> None:
        if shutil.which("docker") is None:
            return
        subprocess.run(["docker", "rmi", image], capture_output=True, text=True, check=False)

    @staticmethod
    def _image_size(image: str) -> int | None:
        if shutil.which("docker") is None:
            return None
        result = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None

    @staticmethod
    def _image_prefix(config: SandboxConfig) -> str:
        return config.skill_env_cache_image_prefix.strip() or DEFAULT_CACHE_IMAGE_PREFIX

    @staticmethod
    def _base_image(default_image: str, requirements_text: str, config: SandboxConfig | None = None) -> str:
        heavy_base = (config.skill_env_cache_torch_base_image if config is not None else "").strip()
        if heavy_base and _has_torch_requirement(requirements_text):
            return heavy_base
        return default_image

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
    lines = []
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
    normalized = normalized_requirements(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _has_torch_requirement(text: str) -> bool:
    packages = {"torch", "torchvision", "torchaudio"}
    for line in normalized_requirements(text).splitlines():
        name = line.split(";", 1)[0].strip()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(separator, 1)[0].strip()
        if name.lower().replace("_", "-") in packages:
            return True
    return False
