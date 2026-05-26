from __future__ import annotations

import sys
from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRunner
from app.core.skills.local_subprocess import LocalSubprocessEnvironmentCache


def test_local_subprocess_environment_uses_base_python_without_requirements(tmp_path: Path) -> None:
    package_root = tmp_path / "plugin"
    package_root.mkdir()
    cache = LocalSubprocessEnvironmentCache(tmp_path)

    env = cache.prepare(skill_name="demo", package_root=package_root)

    assert env.status == "base"
    assert env.requirements_hash is None
    assert env.python == Path(sys.executable)


def test_python_script_skill_runs_with_local_subprocess_runtime(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins" / "skills" / "local-subprocess-plugin"
    skill_root = plugin_root / "skills" / "local-subprocess-skill"
    scripts = skill_root / "scripts"
    scripts.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        """
{
  "id": "local-subprocess-plugin",
  "name": "Local Subprocess Plugin",
  "version": "1.0.0",
  "skills": ["skills/*/manifest.json"]
}
""",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        """
{
  "name": "local-subprocess-skill",
  "description": "Local subprocess skill",
  "output_kind": "markdown",
  "generation": true,
  "quality_template": [],
  "execution": {
    "type": "python_script",
    "runtime": "local_subprocess",
    "script": "scripts/run_skill.py"
  },
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )
    (scripts / "run_skill.py").write_text(
        """
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
outputs = Path(payload["outputs_dir"])
(outputs / "result.md").write_text("# " + payload["spec"]["title"] + "\\n", encoding="utf-8")
print(json.dumps({"ok": True}))
""",
        encoding="utf-8",
    )

    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("local-subprocess-thread")
    result = SkillRunner(store, root_dir=tmp_path).run("local-subprocess-skill", {"title": "Local"}, paths)

    assert result.data["execution_runtime"] == "local_subprocess"
    assert result.outputs[0].name == "result.md"
    assert (paths.outputs / "result.md").read_text(encoding="utf-8") == "# Local\n"
