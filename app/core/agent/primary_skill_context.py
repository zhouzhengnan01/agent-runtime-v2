from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRegistry
from app.schemas import RuntimeOptions


_SKILL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]{2,128}$")


@dataclass(frozen=True)
class PrimarySkillStage:
    id: str
    name: str
    owner_skills: tuple[str, ...]
    outputs: tuple[str, ...]
    ready_gate: str = ""
    done_gate: str = ""


@dataclass(frozen=True)
class PrimarySkillContext:
    skill_name: str
    description: str
    summary: str
    stages: tuple[PrimarySkillStage, ...]
    blocked_stage_signals: tuple[str, ...]
    completed_stage_ids: tuple[str, ...]
    active_stage_id: str | None
    active_stage_name: str
    active_stage_owner_skills: tuple[str, ...]
    completion_status: str
    source: str

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def completed_stage_count(self) -> int:
        return len(self.completed_stage_ids)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "primary_skill_name": self.skill_name,
            "primary_skill_stage_count": self.stage_count,
            "primary_skill_completed_stage_count": self.completed_stage_count,
            "primary_skill_completion_status": self.completion_status,
            "active_stage_id": self.active_stage_id,
            "active_stage_name": self.active_stage_name,
            "active_stage_owner_skills": list(self.active_stage_owner_skills),
            "blocked_stages": list(self.blocked_stage_signals),
            "primary_skill_context_source": self.source,
        }


def load_primary_skill_context(
    *,
    root_dir: Path,
    artifact_store: ArtifactStore,
    runtime_options: RuntimeOptions | None,
    thread_id: str,
    available_artifacts: list[dict[str, Any]],
    current_artifacts: list[dict[str, Any]],
    required_inputs: list[dict[str, Any]],
) -> PrimarySkillContext | None:
    primary_skill_name = _primary_skill_name(runtime_options)
    if not primary_skill_name:
        return None
    description = ""
    source = "selected_skill"
    raw_stages: list[dict[str, Any]] = []
    try:
        skill = SkillRegistry(root_dir).get(primary_skill_name)
        description = skill.description
        raw_stages = [dict(item) for item in skill.stages if isinstance(item, dict)]
        source = "skill_manifest"
    except KeyError:
        skill = None

    merged_artifacts = _merge_artifacts(available_artifacts, current_artifacts)
    payload = _primary_skill_payload(artifact_store, thread_id, primary_skill_name, merged_artifacts)
    if payload is not None:
        payload_stages = payload.get("stages")
        if isinstance(payload_stages, list) and payload_stages:
            raw_stages = [dict(item) for item in payload_stages if isinstance(item, dict)]
            source = "thread_artifact"
        if not description:
            description = str(payload.get("summary") or "")
    if not raw_stages:
        if skill is None:
            return None
        return PrimarySkillContext(
            skill_name=primary_skill_name,
            description=description,
            summary=str(payload.get("summary") or description) if payload else description,
            stages=(),
            blocked_stage_signals=(),
            completed_stage_ids=(),
            active_stage_id=None,
            active_stage_name="",
            active_stage_owner_skills=(),
            completion_status="unknown",
            source=source,
        )

    stages = tuple(_normalize_stage(item) for item in raw_stages if _normalize_stage(item) is not None)
    if not stages:
        return None
    artifact_names = _artifact_names(merged_artifacts)
    blocked_stage_signals = _blocked_stage_signals(required_inputs, payload)
    completed_stage_ids = tuple(stage.id for stage in stages if _stage_completed(stage, artifact_names))
    active_stage = _active_stage(stages, blocked_stage_signals, completed_stage_ids)
    completion_status = "completed" if len(completed_stage_ids) == len(stages) and not blocked_stage_signals else "pending"
    summary = str(payload.get("summary") or description) if payload else description
    return PrimarySkillContext(
        skill_name=primary_skill_name,
        description=description,
        summary=summary,
        stages=stages,
        blocked_stage_signals=blocked_stage_signals,
        completed_stage_ids=completed_stage_ids,
        active_stage_id=active_stage.id if active_stage is not None else None,
        active_stage_name=active_stage.name if active_stage is not None else "",
        active_stage_owner_skills=active_stage.owner_skills if active_stage is not None else (),
        completion_status=completion_status,
        source=source,
    )


def primary_skill_stage_prompt(context: PrimarySkillContext | None) -> str:
    if context is None or not context.stages:
        return ""
    lines = [
        "Primary skill execution context is active.",
        f"Primary skill: {context.skill_name}",
        f"Stage progress: {context.completed_stage_count}/{context.stage_count}",
    ]
    if context.active_stage_name:
        lines.append(f"Active stage: {context.active_stage_name}")
    if context.active_stage_owner_skills:
        lines.append(f"Preferred stage skills: {', '.join(context.active_stage_owner_skills)}")
    if context.blocked_stage_signals:
        lines.append(f"Pending stage signals: {', '.join(context.blocked_stage_signals)}")
    return "\n".join(lines)


def _primary_skill_name(runtime_options: RuntimeOptions | None) -> str:
    if runtime_options is None or not runtime_options.selected_skills:
        return ""
    return str(runtime_options.selected_skills[0] or "").strip()


def _merge_artifacts(
    available_artifacts: list[dict[str, Any]],
    current_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (available_artifacts, current_artifacts):
        for item in source:
            if not isinstance(item, dict):
                continue
            key = str(item.get("path") or item.get("name") or "")
            if key and key in seen:
                continue
            merged.append(dict(item))
            if key:
                seen.add(key)
    return merged


def _primary_skill_payload(
    artifact_store: ArtifactStore,
    thread_id: str,
    skill_name: str,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    target_name = f"{skill_name}.json"
    for item in reversed(artifacts):
        if str(item.get("name") or "") != target_name:
            continue
        file_path = _artifact_local_path(artifact_store, thread_id, item)
        if file_path is None or not file_path.is_file():
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _artifact_local_path(
    artifact_store: ArtifactStore,
    thread_id: str,
    artifact: dict[str, Any],
) -> Path | None:
    raw_path = str(artifact.get("path") or "").strip()
    if not raw_path:
        return None
    if raw_path.startswith("/mnt/user-data/outputs/"):
        try:
            return artifact_store.resolve_virtual_path(thread_id, raw_path)
        except ValueError:
            return None
    if raw_path.startswith("outputs/"):
        paths = artifact_store.prepare_thread(thread_id)
        try:
            return artifact_store.output_path(paths, raw_path)
        except ValueError:
            return None
    return None


def _normalize_stage(item: dict[str, Any]) -> PrimarySkillStage | None:
    name = str(item.get("name") or item.get("id") or "").strip()
    if not name:
        return None
    stage_id = str(item.get("id") or _slug(name)).strip() or _slug(name)
    owner_skills = _owner_skills(item)
    outputs = tuple(_string_list(item.get("outputs")))
    return PrimarySkillStage(
        id=stage_id,
        name=name,
        owner_skills=owner_skills,
        outputs=outputs,
        ready_gate=str(item.get("ready_gate") or "").strip(),
        done_gate=str(item.get("done_gate") or "").strip(),
    )


def _owner_skills(item: dict[str, Any]) -> tuple[str, ...]:
    explicit = _string_list(item.get("owner_skills"))
    if explicit:
        return tuple(explicit)
    owner_text = str(item.get("owner") or item.get("skill") or "").strip()
    if not owner_text:
        return ()
    normalized = owner_text.replace(" and ", "+").replace(" AND ", "+")
    tokens = re.split(r"\s*\+\s*|\s*,\s*", normalized)
    skills = [token.strip(" `") for token in tokens if _SKILL_TOKEN_RE.fullmatch(token.strip(" `"))]
    return tuple(skills)


def _artifact_names(artifacts: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        if name:
            names.add(name)
        raw_path = str(item.get("path") or "").strip()
        if raw_path:
            names.add(Path(raw_path).name.lower())
    return names


def _blocked_stage_signals(
    required_inputs: list[dict[str, Any]],
    payload: dict[str, Any] | None,
) -> tuple[str, ...]:
    signals = [
        str(item.get("stage") or "").strip()
        for item in required_inputs
        if isinstance(item, dict) and str(item.get("stage") or "").strip()
    ]
    if not signals and isinstance(payload, dict):
        signals = _string_list(payload.get("blocked_stages"))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in signals:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return tuple(deduped)


def _active_stage(
    stages: tuple[PrimarySkillStage, ...],
    blocked_stage_signals: tuple[str, ...],
    completed_stage_ids: tuple[str, ...],
) -> PrimarySkillStage | None:
    completed = set(completed_stage_ids)
    for signal in blocked_stage_signals:
        for stage in stages:
            if _stage_matches_signal(stage, signal):
                return stage
    for stage in stages:
        if stage.id not in completed:
            return stage
    return stages[-1] if stages else None


def _stage_matches_signal(stage: PrimarySkillStage, signal: str) -> bool:
    normalized = signal.strip().lower()
    if not normalized:
        return False
    if stage.id.lower() == normalized or stage.name.lower() == normalized:
        return True
    return any(owner.lower() == normalized for owner in stage.owner_skills)


def _stage_completed(stage: PrimarySkillStage, artifact_names: set[str]) -> bool:
    if stage.owner_skills:
        return all(_owner_skill_completed(name, artifact_names) for name in stage.owner_skills)
    file_outputs = [name for name in stage.outputs if "." in name]
    return bool(file_outputs) and all(output.lower() in artifact_names for output in file_outputs)


def _owner_skill_completed(skill_name: str, artifact_names: set[str]) -> bool:
    prefix = f"{skill_name.lower()}."
    return any(name == f"{skill_name.lower()}.md" or name == f"{skill_name.lower()}.json" or name.startswith(prefix) for name in artifact_names)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text.strip().lower()).strip("-")
    return value or "stage"
