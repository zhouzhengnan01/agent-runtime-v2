from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config.agent_config import AgentConfig
from app.core.config.secrets import SecretCodec


class AgentConfigLoader:
    """Load built-in stateless agent configs from config/agents."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.config_dir = self.root_dir / "config" / "agents"
        self.secret_codec = SecretCodec(self.root_dir)

    def list_agents(self) -> list[AgentConfig]:
        agents: list[AgentConfig] = []
        for path in sorted(self.config_dir.glob("*.json")):
            if path.name.endswith(".local.json"):
                continue
            agents.append(self.load(path.stem))
        return agents

    def load(self, name: str) -> AgentConfig:
        safe_name = name.strip().replace("/", "").replace("\\", "")
        path = self.config_dir / f"{safe_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Agent config not found: {safe_name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        local_path = self.config_dir / f"{safe_name}.local.json"
        if local_path.is_file():
            local_data = json.loads(local_path.read_text(encoding="utf-8"))
            local_data = self._encrypt_plain_local_model_secrets(safe_name, local_path, local_data)
            data = self._deep_merge(data, local_data)
        data = self._decrypt_model_secrets(safe_name, data)
        return AgentConfig.model_validate(data)

    @staticmethod
    def public_payload(agent: AgentConfig) -> dict[str, Any]:
        payload = agent.model_dump()
        model = payload.get("model")
        if isinstance(model, dict) and model.get("api_key"):
            model["api_key"] = "********"
        if isinstance(model, dict) and model.get("api_key_enc"):
            model["api_key_enc"] = "********"
        return payload

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            base_value = merged.get(key)
            if isinstance(base_value, dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge(base_value, value)
            else:
                merged[key] = value
        return merged

    def _decrypt_model_secrets(self, agent_name: str, data: dict[str, Any]) -> dict[str, Any]:
        model = data.get("model")
        if not isinstance(model, dict) or model.get("api_key"):
            return data
        encrypted = model.get("api_key_enc")
        if not isinstance(encrypted, str) or not encrypted.strip():
            return data
        decrypted = self.secret_codec.decrypt(encrypted.strip(), purpose=self._secret_purpose(agent_name, "api_key"))
        merged = dict(data)
        merged_model = dict(model)
        merged_model["api_key"] = decrypted
        merged["model"] = merged_model
        return merged

    def _encrypt_plain_local_model_secrets(
        self,
        agent_name: str,
        local_path: Path,
        local_data: Any,
    ) -> dict[str, Any]:
        if not isinstance(local_data, dict):
            return {}
        model = local_data.get("model")
        if not isinstance(model, dict):
            return local_data
        api_key = model.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            return local_data

        encrypted = self.secret_codec.encrypt(api_key, purpose=self._secret_purpose(agent_name, "api_key"))
        sanitized_model = dict(model)
        sanitized_model.pop("api_key", None)
        sanitized_model["api_key_enc"] = encrypted
        sanitized_data = dict(local_data)
        sanitized_data["model"] = sanitized_model
        local_path.write_text(json.dumps(sanitized_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        local_path.chmod(0o600)

        # Keep the plaintext only in memory for the current load. Future loads
        # will decrypt api_key_enc from disk.
        runtime_model = dict(sanitized_model)
        runtime_model["api_key"] = api_key
        runtime_data = dict(sanitized_data)
        runtime_data["model"] = runtime_model
        return runtime_data

    @staticmethod
    def _secret_purpose(agent_name: str, field_name: str) -> str:
        return f"agent:{agent_name}:model:{field_name}"
