import json
from pathlib import Path

from app.cli import set_encrypted_api_key
from app.core.config import AgentConfigLoader
from app.core.config.secrets import SecretCodec


def test_load_builtin_agent() -> None:
    agent = AgentConfigLoader().load("artifact-generator")
    assert agent.name == "artifact-generator"
    assert agent.runtime.stateless is True
    assert "markdown-rendering" in agent.skills
    assert agent.model.model == "Qwen3.6-35B-A3B"
    assert agent.model.base_url == "http://124.132.152.75:62091/v1"
    assert agent.model.api_key is None
    assert agent.routing.llm_workflow_router is True
    assert agent.routing.llm_workflow_router_env == "LLM_WORKFLOW_ROUTER"


def test_list_agents() -> None:
    names = {agent.name for agent in AgentConfigLoader().list_agents()}
    assert {"default", "artifact-generator", "behavior-detector"} <= names


def test_local_agent_config_overrides_base_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    base_config = {
        "name": "default",
        "display_name": "Default",
        "model": {
            "model": "json-model",
            "base_url": "http://base.local/v1",
            "temperature": 0.4,
            "max_tokens": 2048,
        },
        "runtime": {"max_tool_rounds": 6},
        "skills": ["markdown-rendering"],
    }
    local_config = {
        "model": {
            "api_key": "local-key",
            "temperature": 0.1,
        },
        "runtime": {"max_tool_rounds": 2},
    }
    (config_dir / "default.json").write_text(json.dumps(base_config), encoding="utf-8")
    (config_dir / "default.local.json").write_text(json.dumps(local_config), encoding="utf-8")

    agent = AgentConfigLoader(root_dir=tmp_path).load("default")

    assert agent.name == "default"
    assert agent.model.model == "json-model"
    assert agent.model.base_url == "http://base.local/v1"
    assert agent.model.api_key == "local-key"
    assert agent.model.temperature == 0.1
    assert agent.model.max_tokens == 2048
    assert agent.runtime.max_tool_rounds == 2
    assert agent.skills == ["markdown-rendering"]


def test_list_agents_ignores_local_override_files(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    base_config = {
        "name": "default",
        "display_name": "Default",
        "model": {"model": "json-model", "base_url": "http://base.local/v1"},
    }
    (config_dir / "default.json").write_text(json.dumps(base_config), encoding="utf-8")
    (config_dir / "default.local.json").write_text(
        json.dumps({"model": {"api_key": "local-key"}}),
        encoding="utf-8",
    )

    agents = AgentConfigLoader(root_dir=tmp_path).list_agents()

    assert [agent.name for agent in agents] == ["default"]
    assert agents[0].model.api_key == "local-key"


def test_public_agent_payload_redacts_api_key(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    base_config = {
        "name": "default",
        "display_name": "Default",
        "model": {"model": "json-model", "base_url": "http://base.local/v1"},
    }
    (config_dir / "default.json").write_text(json.dumps(base_config), encoding="utf-8")
    (config_dir / "default.local.json").write_text(
        json.dumps({"model": {"api_key": "local-key"}}),
        encoding="utf-8",
    )
    loader = AgentConfigLoader(root_dir=tmp_path)
    agent = loader.load("default")

    payload = loader.public_payload(agent)

    assert agent.model.api_key == "local-key"
    assert payload["model"]["api_key"] == "********"


def test_load_agent_decrypts_encrypted_api_key_from_local_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    base_config = {
        "name": "default",
        "display_name": "Default",
        "model": {"model": "json-model", "base_url": "http://base.local/v1"},
    }
    loader = AgentConfigLoader(root_dir=tmp_path)
    encrypted = SecretCodec(tmp_path).encrypt(
        "encrypted-local-key",
        purpose=loader._secret_purpose("default", "api_key"),
    )
    (config_dir / "default.json").write_text(json.dumps(base_config), encoding="utf-8")
    (config_dir / "default.local.json").write_text(
        json.dumps({"model": {"api_key_enc": encrypted}}),
        encoding="utf-8",
    )

    agent = loader.load("default")
    payload = loader.public_payload(agent)

    assert agent.model.api_key == "encrypted-local-key"
    assert payload["model"]["api_key"] == "********"
    assert payload["model"]["api_key_enc"] == "********"


def test_plain_api_key_takes_precedence_over_encrypted_api_key(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    base_config = {
        "name": "default",
        "display_name": "Default",
        "model": {"model": "json-model", "base_url": "http://base.local/v1"},
    }
    loader = AgentConfigLoader(root_dir=tmp_path)
    encrypted = SecretCodec(tmp_path).encrypt("encrypted-key", purpose=loader._secret_purpose("default", "api_key"))
    (config_dir / "default.json").write_text(json.dumps(base_config), encoding="utf-8")
    (config_dir / "default.local.json").write_text(
        json.dumps({"model": {"api_key": "plain-key", "api_key_enc": encrypted}}),
        encoding="utf-8",
    )

    agent = loader.load("default")

    assert agent.model.api_key == "plain-key"


def test_set_encrypted_api_key_writes_local_override_and_master_key(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    base_config = {
        "name": "default",
        "display_name": "Default",
        "model": {"model": "json-model", "base_url": "http://base.local/v1", "temperature": 0.4},
    }
    (config_dir / "default.json").write_text(json.dumps(base_config), encoding="utf-8")
    loader = AgentConfigLoader(root_dir=tmp_path)

    local_path = set_encrypted_api_key(loader, "default", "cli-secret-key")

    local_data = json.loads(local_path.read_text(encoding="utf-8"))
    assert local_data["model"]["api_key_enc"].startswith("enc.fernet.v1.")
    assert "api_key" not in local_data["model"]
    assert (tmp_path / ".runtime" / "secrets" / "master.key").is_file()
    assert loader.load("default").model.api_key == "cli-secret-key"
