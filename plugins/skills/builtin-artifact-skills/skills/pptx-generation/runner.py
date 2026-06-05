from __future__ import annotations

from runner import SkillRunner


def run(skill_name, spec, paths, artifact_store):
    return SkillRunner(artifact_store).run(skill_name, spec, paths)

