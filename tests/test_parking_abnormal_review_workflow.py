from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.skills import SkillDefinition
from app.core.skills.runner_types import SkillRunResult
from app.schemas import Attachment, Message, RuntimeOptions


def _load_workflow_module() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "workflows"
        / "builtin-artifact-workflows"
        / "parking_abnormal_review.py"
    )
    spec = importlib.util.spec_from_file_location("parking_abnormal_review_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load workflow module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_video_attachments_are_detected_by_mime_type() -> None:
    workflow = _load_workflow_module()
    image = Attachment(name="image.jpg", mime_type="image/jpeg")
    video = Attachment(name="clip.mp4", mime_type="video/mp4")
    text = Attachment(name="note.txt", mime_type="text/plain")

    assert workflow._image_attachments([image, video, text]) == [image]
    assert workflow._video_attachments([image, video, text]) == [video]


def test_candidate_frame_indices_are_uniformly_sampled() -> None:
    workflow = _load_workflow_module()

    indices = workflow._candidate_frame_indices(100, 5)

    assert indices == [0, 25, 50, 74, 99]


def test_video_frame_selection_prefers_distinct_high_quality_frames() -> None:
    workflow = _load_workflow_module()

    class FakeCv2:
        @staticmethod
        def absdiff(left: Any, right: Any) -> Any:
            return abs(left - right)

    class FakeDiff:
        def __init__(self, value: float) -> None:
            self.value = value

        def mean(self) -> float:
            return self.value

    class FakeCv2WithDiff(FakeCv2):
        @staticmethod
        def absdiff(left: Any, right: Any) -> FakeDiff:
            return FakeDiff(abs(float(left) - float(right)))

    candidates = [
        workflow.VideoFrameCandidate(0, 0, 10, 120, 10, 1, "frame-0"),
        workflow.VideoFrameCandidate(1, 1000, 100, 120, 100, 1, "frame-1-duplicate"),
        workflow.VideoFrameCandidate(2, 2000, 90, 120, 90, 40, "frame-2-distinct"),
        workflow.VideoFrameCandidate(3, 3000, 80, 120, 80, 80, "frame-3-distinct"),
    ]

    selected = workflow._select_video_frame_candidates(FakeCv2WithDiff(), candidates, max_frames=3)

    assert [candidate.frame_index for candidate in selected] == [1, 2, 3]


def test_video_frame_attachments_convert_extracted_frames_to_image_attachments(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    workflow = _load_workflow_module()
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = artifact_store.prepare_thread("thread-video")
    video = paths.uploads / "clip.mp4"
    video.write_bytes(b"fake-video")
    frame = paths.workspace / "frame-001.jpg"
    frame.write_bytes(b"fake-jpeg")

    def fake_extract_video_frames(video_path: Path, output_dir: Path, *, prefix: str) -> list[Any]:
        assert video_path == video
        assert output_dir == paths.workspace / workflow.VIDEO_FRAME_OUTPUT_DIR
        assert prefix.startswith("video_1_clip_")
        return [
            workflow.ExtractedVideoFrame(
                path=frame,
                source_video=str(video),
                frame_index=12,
                timestamp_ms=1500.0,
                sharpness=88.0,
                brightness=120.0,
                score=180.0,
            )
        ]

    monkeypatch.setattr(workflow, "_extract_video_frames", fake_extract_video_frames)

    attachments, reports = workflow._video_frame_attachments(
        [Attachment(name="clip.mp4", path="/mnt/user-data/uploads/clip.mp4", mime_type="video/mp4")],
        paths,
    )

    assert len(attachments) == 1
    assert attachments[0].name == "frame-001.jpg"
    assert attachments[0].mime_type == "image/jpeg"
    assert attachments[0].metadata["source"] == "video_frame"
    assert attachments[0].metadata["timestamp_ms"] == 1500.0
    assert reports == [
        {
            "name": "clip.mp4",
            "mime_type": "video/mp4",
            "path": "/mnt/user-data/uploads/clip.mp4",
            "status": "completed",
            "frame_count": 1,
            "frames": [
                {
                    "name": "frame-001.jpg",
                    "frame_index": 12,
                    "timestamp_ms": 1500.0,
                    "sharpness": 88.0,
                    "brightness": 120.0,
                }
            ],
        }
    ]


def test_video_frame_attachments_report_missing_opencv(tmp_path: Path, monkeypatch: Any) -> None:
    workflow = _load_workflow_module()
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = artifact_store.prepare_thread("thread-video")
    video = paths.uploads / "clip.mp4"
    video.write_bytes(b"fake-video")

    def missing_opencv(video_path: Path, output_dir: Path, *, prefix: str) -> list[Any]:
        raise RuntimeError("OpenCV is unavailable; install opencv-python to enable video frame extraction.")

    monkeypatch.setattr(workflow, "_extract_video_frames", missing_opencv)

    attachments, reports = workflow._video_frame_attachments(
        [Attachment(name="clip.mp4", path="/mnt/user-data/uploads/clip.mp4", mime_type="video/mp4")],
        paths,
    )

    assert attachments == []
    assert reports[0]["status"] == "failed"
    assert reports[0]["error_code"] == "opencv_unavailable"


def test_review_skill_candidates_expand_composite_and_score_objective() -> None:
    workflow = _load_workflow_module()

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "behavior-group": SkillDefinition(
                    name="behavior-group",
                    description="组合行为识别",
                    output_kind="json",
                    skill_type="composite",
                    child_skills=("smoking-review", "fall-review"),
                ),
                "smoking-review": SkillDefinition(
                    name="smoking-review",
                    description="抽烟 SmokingDetection 复判",
                    output_kind="json",
                    routing={"keywords": ["抽烟", "SmokingDetection"]},
                ),
                "fall-review": SkillDefinition(
                    name="fall-review",
                    description="跌倒 FallDetection 复判",
                    output_kind="json",
                    routing={"keywords": ["跌倒", "FallDetection"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

    candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["behavior-group"],
        objective="SmokingDetection",
        prompt_text="本次复判的识别目标为[SmokingDetection]",
    )

    assert [candidate.skill.name for candidate in candidates] == ["smoking-review", "fall-review"]
    assert candidates[0].score > candidates[1].score


def test_review_objective_and_skill_score_use_task_target_payload() -> None:
    workflow = _load_workflow_module()
    prompt_text = """
    请复判下面事件：
    {
      "id": "abf26546-03e4-495a-bc92-fa68c3287c3b",
      "modelId": "CustomerBehaviorDetection",
      "modelName": "顾客行为监管",
      "taskName": "顾客行为检测",
      "taskTarget": {
        "value": "FallDetection",
        "text": "人员跌倒/倒地检测"
      }
    }
    """

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "behavior-group": SkillDefinition(
                    name="behavior-group",
                    description="组合行为识别",
                    output_kind="json",
                    skill_type="composite",
                    child_skills=("smoking-review", "fall-review"),
                ),
                "smoking-review": SkillDefinition(
                    name="smoking-review",
                    description="抽烟 SmokingDetection 复判",
                    output_kind="json",
                    routing={"keywords": ["抽烟", "SmokingDetection"]},
                ),
                "fall-review": SkillDefinition(
                    name="fall-review",
                    description="跌倒 FallDetection 复判，人员跌倒/倒地检测",
                    output_kind="json",
                    routing={"keywords": ["跌倒", "FallDetection", "人员跌倒/倒地检测"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

    candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["behavior-group"],
        objective=workflow._objective(prompt_text),
        prompt_text=prompt_text,
    )

    assert workflow._objective(prompt_text) == "FallDetection"
    assert [candidate.skill.name for candidate in candidates] == ["smoking-review", "fall-review"]
    assert candidates[1].score > candidates[0].score
    assert "keyword=人员跌倒/倒地检测" in candidates[1].score_reasons


def test_review_objective_prefers_fall_target_over_smoking_summary_text() -> None:
    workflow = _load_workflow_module()
    prompt_text = """
    人员跌倒/倒地检测
    置信度
    fall
    85%
    告警摘要
    基于所附3张监控图片复判，未见明显手持香烟、手口吸食动作、烟雾扩散轨迹或暗光火点等抽烟特征。
    """

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "smoking-review": SkillDefinition(
                    name="smoking-review",
                    description="抽烟 SmokingDetection 复判",
                    output_kind="json",
                    routing={"keywords": ["抽烟", "SmokingDetection"]},
                ),
                "fall-review": SkillDefinition(
                    name="fall-review",
                    description="跌倒 FallDetection 复判，人员跌倒/倒地检测",
                    output_kind="json",
                    routing={"keywords": ["跌倒", "FallDetection", "人员跌倒/倒地检测"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

    objective = workflow._objective(prompt_text)
    candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["smoking-review", "fall-review"],
        objective=objective,
        prompt_text=prompt_text,
    )

    assert objective == "FallDetection"
    assert candidates[1].skill.name == "fall-review"
    assert candidates[1].score > candidates[0].score


def test_parking_review_skill_score_routes_parking_violation_and_congestion_targets() -> None:
    workflow = _load_workflow_module()

    violation_skill = SkillDefinition(
        name="parking-violation-review",
        description="车辆违停 违规停车 ParkingViolationDetection",
        output_kind="markdown",
        routing={"keywords": ["违规停车", "车辆违停", "车辆违停检测", "ParkingViolationDetection"]},
    )
    congestion_skill = SkillDefinition(
        name="parking-congestion-review",
        description="停车场通道拥堵检测 车辆拥堵 ParkingCongestionDetection",
        output_kind="markdown",
        routing={"keywords": ["停车场通道拥堵检测", "车辆拥堵", "ParkingCongestionDetection"]},
    )
    violation_prompt = (
        '{"taskTarget":{"value":"ParkingViolationDetection","text":"车辆违停检测"},'
        '"modelName":"车场异常事件监控"}'
    )
    congestion_prompt = (
        '{"taskTarget":{"value":"ParkingCongestionDetection","text":"停车场通道拥堵检测"},'
        '"modelName":"车场异常事件监控"}'
    )

    assert workflow._objective(violation_prompt) == "ParkingViolationDetection"
    assert workflow._objective(congestion_prompt) == "ParkingCongestionDetection"

    violation_score, violation_reasons = workflow._score_review_skill(
        violation_skill,
        workflow._skill_context(violation_skill),
        objective=workflow._objective(violation_prompt),
        prompt_text=violation_prompt,
    )
    wrong_violation_score, _ = workflow._score_review_skill(
        congestion_skill,
        workflow._skill_context(congestion_skill),
        objective=workflow._objective(violation_prompt),
        prompt_text=violation_prompt,
    )
    congestion_score, congestion_reasons = workflow._score_review_skill(
        congestion_skill,
        workflow._skill_context(congestion_skill),
        objective=workflow._objective(congestion_prompt),
        prompt_text=congestion_prompt,
    )
    wrong_congestion_score, _ = workflow._score_review_skill(
        violation_skill,
        workflow._skill_context(violation_skill),
        objective=workflow._objective(congestion_prompt),
        prompt_text=congestion_prompt,
    )

    assert violation_score > wrong_violation_score
    assert congestion_score > wrong_congestion_score
    assert "keyword=车辆违停检测" in violation_reasons
    assert "keyword=停车场通道拥堵检测" in congestion_reasons


def test_parking_review_invokes_one_selected_skill_and_injects_result(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    workflow = _load_workflow_module()
    captured: dict[str, Any] = {}

    class FakeRegistry:
        def get(self, name: str) -> SkillDefinition:
            if name == "behavior-group":
                return SkillDefinition(
                    name="behavior-group",
                    description="组合行为识别",
                    output_kind="json",
                    skill_type="composite",
                    child_skills=("smoking-review", "fall-review"),
                )
            if name == "smoking-review":
                return SkillDefinition(
                    name="smoking-review",
                    description="抽烟 SmokingDetection 复判",
                    output_kind="json",
                    routing={"keywords": ["抽烟", "SmokingDetection"]},
                )
            if name == "fall-review":
                return SkillDefinition(
                    name="fall-review",
                    description="跌倒 FallDetection 复判",
                    output_kind="json",
                    routing={"keywords": ["跌倒", "FallDetection"]},
                )
            raise KeyError(name)

    class FakeRunner:
        def run(self, skill_name: str, spec: dict[str, Any], paths: Any, on_event: Any = None) -> SkillRunResult:
            captured["skill_name"] = skill_name
            captured["spec"] = spec
            return SkillRunResult(skill_name=skill_name, data={"review_decision": "hit", "reason": "skill says smoking"})

    def fake_complete_review(
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        image_attachments: list[Attachment],
        paths: Any,
        review_source_id: str,
        objective: str,
        visual_regions: list[dict[str, Any]],
        skill_selection: Any,
        skill_result: SkillRunResult | None,
        skill_error: str,
    ) -> str:
        captured["final_skill_selection"] = skill_selection
        captured["final_skill_result"] = skill_result
        captured["final_skill_error"] = skill_error
        return '[{"reviewSourceId":"source-1","reviewEventId":"event-1","hit":1,"result":"发现抽烟行为"}]'

    monkeypatch.setattr(workflow.ParkingAbnormalReviewWorkflow, "_complete_review", staticmethod(fake_complete_review))

    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("thread-review")
    image = paths.uploads / "image.jpg"
    image.write_bytes(b"fake-image")
    review_workflow = workflow.ParkingAbnormalReviewWorkflow(
        store,
        skill_registry=FakeRegistry(),
        skill_runner=FakeRunner(),
    )

    result, events = review_workflow.run_with_events(
        AgentConfig(name="default", display_name="Default"),
        [
            Message(
                role="user",
                content=(
                    "当前复判事件来源reviewSourceId为[source-1]。\n"
                    "本次复判的识别目标为[SmokingDetection]"
                ),
            )
        ],
        [Attachment(name="image.jpg", path="/mnt/user-data/uploads/image.jpg", mime_type="image/jpeg")],
        "thread-review",
        runtime_options=RuntimeOptions(selected_skills=["behavior-group"]),
    )

    assert captured["skill_name"] == "smoking-review"
    assert captured["spec"]["objective"] == "SmokingDetection"
    assert captured["spec"]["has_visual_evidence"] is True
    assert captured["final_skill_selection"].skill_name == "smoking-review"
    assert captured["final_skill_result"].data["review_decision"] == "hit"
    assert captured["final_skill_error"] == ""
    assert result.reply == '[{"reviewSourceId":"source-1","reviewEventId":"event-1","hit":1,"result":"发现抽烟行为"}]'
    event_types = [event.type for event in events]
    assert "review.skill_selection.completed" in event_types
    assert "review.skill_invocation.completed" in event_types


def test_parking_review_logs_llm_raw_reply_and_normalized_result(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    workflow = _load_workflow_module()

    def fake_complete_review(
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        image_attachments: list[Attachment],
        paths: Any,
        review_source_id: str,
        objective: str,
        visual_regions: list[dict[str, Any]],
        skill_selection: Any,
        skill_result: SkillRunResult | None,
        skill_error: str,
    ) -> str:
        return '[{"reviewSourceId":"source-log","reviewEventId":"event-log","hit":1,"result":"模型判定命中"}]'

    monkeypatch.setattr(workflow.ParkingAbnormalReviewWorkflow, "_complete_review", staticmethod(fake_complete_review))

    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("thread-log")
    image = paths.uploads / "image.jpg"
    image.write_bytes(b"fake-image")
    review_workflow = workflow.ParkingAbnormalReviewWorkflow(store)

    with caplog.at_level("INFO", logger="uvicorn.error"):
        result, events = review_workflow.run_with_events(
            AgentConfig(name="default", display_name="Default"),
            [
                Message(
                    role="user",
                    content=(
                        "当前复判事件来源reviewSourceId为[source-log]。\n"
                        "本次复判的识别目标为[ClutterDetection]"
                    ),
                )
            ],
            [Attachment(name="image.jpg", path="/mnt/user-data/uploads/image.jpg", mime_type="image/jpeg")],
            "thread-log",
            runtime_options=RuntimeOptions(),
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert result.reply == '[{"reviewSourceId":"source-log","reviewEventId":"event-log","hit":1,"result":"模型判定命中"}]'
    assert "parking review llm raw reply review_source_id=source-log" in logs
    assert "raw_chars=" in logs
    assert "模型判定命中" in logs
    assert "复判归一化结果 | parking review normalized result" in logs
    assert "复判最终结果 | parking review final result" in logs
    assert "reviewSourceId: source-log" in logs
    assert '"result": "模型判定命中"' in logs
    assert "结果字符数:" in logs
    event_payloads = {event.type: event.data for event in events}
    assert event_payloads["review.llm.raw_reply"]["raw_reply"] == (
        '[{"reviewSourceId":"source-log","reviewEventId":"event-log","hit":1,"result":"模型判定命中"}]'
    )
    assert event_payloads["review.normalized_result"]["result"] == result.reply


def test_parking_review_logs_image_sources(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    workflow = _load_workflow_module()

    def fake_complete_review(
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        image_attachments: list[Attachment],
        paths: Any,
        review_source_id: str,
        objective: str,
        visual_regions: list[dict[str, Any]],
        skill_selection: Any,
        skill_result: SkillRunResult | None,
        skill_error: str,
    ) -> str:
        return '[{"reviewSourceId":"source-image","reviewEventId":"event-image","hit":0,"result":"未命中"}]'

    monkeypatch.setattr(workflow.ParkingAbnormalReviewWorkflow, "_complete_review", staticmethod(fake_complete_review))

    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("thread-image")
    image = paths.uploads / "image-remote.jpg"
    image.write_bytes(b"fake-image")
    review_workflow = workflow.ParkingAbnormalReviewWorkflow(store)
    image_url = "https://example.test/api/ai/task/history/_read/image.jpg?accessKey=abc"

    with caplog.at_level("INFO", logger="uvicorn.error"):
        result, events = review_workflow.run_with_events(
            AgentConfig(name="default", display_name="Default"),
            [
                Message(
                    role="user",
                    content=(
                        "当前复判事件来源reviewSourceId为[source-image]。\n"
                        "本次复判的识别目标为[ParkingAbnormalDetection]"
                    ),
                )
            ],
            [
                Attachment(
                    name="image.jpg",
                    path="/mnt/user-data/uploads/image-remote.jpg",
                    mime_type="image/jpeg",
                    metadata={
                        "original_uri": image_url,
                        "dataId": "data-1",
                        "sourceId": "source-camera-1",
                        "timestamp": 1780656596708,
                        "sha1": "sha1-value",
                    },
                )
            ],
            "thread-image",
            runtime_options=RuntimeOptions(),
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert result.reply == '[{"reviewSourceId":"source-image","reviewEventId":"event-image","hit":0,"result":"未命中"}]'
    assert "复判图片来源 | parking review image sources" in logs
    assert "图片链接/路径:" in logs
    assert image_url in logs
    assert '"dataId": "data-1"' in logs
    assert "复判最终结果 | parking review final result" in logs
    event_payloads = {event.type: event.data for event in events}
    assert event_payloads["review.input"]["image_sources"][0]["url"] == image_url
    assert event_payloads["review.input"]["image_sources"][0]["dataId"] == "data-1"


def test_parking_review_logs_and_passes_visual_regions(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    workflow = _load_workflow_module()
    captured: dict[str, Any] = {}

    class FakeRegistry:
        def get(self, name: str) -> SkillDefinition:
            if name == "parking-violation-review":
                return SkillDefinition(
                    name="parking-violation-review",
                    description="违规停车 车辆违停",
                    output_kind="markdown",
                    routing={"keywords": ["违规停车", "车辆违停"]},
                )
            raise KeyError(name)

    class FakeRunner:
        def run(self, skill_name: str, spec: dict[str, Any], paths: Any, on_event: Any = None) -> SkillRunResult:
            captured["skill_name"] = skill_name
            captured["spec"] = spec
            return SkillRunResult(skill_name=skill_name, data={"review_decision": "needs_model"})

    def fake_complete_review(
        *,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        image_attachments: list[Attachment],
        paths: Any,
        review_source_id: str,
        objective: str,
        visual_regions: list[dict[str, Any]],
        skill_selection: Any,
        skill_result: SkillRunResult | None,
        skill_error: str,
    ) -> str:
        captured["visual_regions"] = visual_regions
        return '[{"reviewSourceId":"source-region","reviewEventId":"event-region","hit":0,"result":"未命中"}]'

    monkeypatch.setattr(workflow.ParkingAbnormalReviewWorkflow, "_complete_review", staticmethod(fake_complete_review))

    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("thread-region")
    image = paths.uploads / "image-region.jpg"
    image.write_bytes(b"fake-image")
    review_workflow = workflow.ParkingAbnormalReviewWorkflow(
        store,
        skill_registry=FakeRegistry(),
        skill_runner=FakeRunner(),
    )
    prompt_text = (
        '{"reviewSourceId":"source-region","taskTarget":{"value":"ParkingViolationDetection","text":"车辆违停检测"},'
        '"schemaResults":{"roi":[[0,0],[100,0],[100,100],[0,100]],"alarmBox":[10,20,80,90]}}'
    )

    with caplog.at_level("INFO", logger="uvicorn.error"):
        result, events = review_workflow.run_with_events(
            AgentConfig(name="default", display_name="Default"),
            [Message(role="user", content=prompt_text)],
            [
                Attachment(
                    name="image.jpg",
                    path="/mnt/user-data/uploads/image-region.jpg",
                    mime_type="image/jpeg",
                    metadata={
                        "original_uri": "https://example.test/image.jpg",
                        "dataId": "data-region",
                        "bbox": [11, 22, 33, 44],
                        "roi": {"points": [[1, 2], [3, 4], [5, 6]]},
                    },
                )
            ],
            "thread-region",
            runtime_options=RuntimeOptions(selected_skills=["parking-violation-review"]),
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    event_payloads = {event.type: event.data for event in events}
    assert result.reply == '[{"reviewSourceId":"source-region","reviewEventId":"event-region","hit":0,"result":"未命中"}]'
    assert captured["skill_name"] == "parking-violation-review"
    assert captured["spec"]["visual_regions"] == captured["visual_regions"]
    assert captured["visual_regions"][0]["source"] == "attachment_metadata"
    assert captured["visual_regions"][0]["regions"]["bbox"] == [11, 22, 33, 44]
    assert captured["visual_regions"][1]["source"] == "prompt_json"
    assert captured["visual_regions"][1]["regions"]["alarm_box"] == [10, 20, 80, 90]
    assert "复判视觉区域 | parking review visual regions" in logs
    assert '"bbox": [' in logs
    assert '"alarm_box": [' in logs
    assert event_payloads["review.input"]["visual_regions"] == captured["visual_regions"]


def test_parking_violation_review_prompt_rejects_marked_parking_spaces(monkeypatch: Any) -> None:
    workflow = _load_workflow_module()
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, agent_config: AgentConfig, runtime_options: RuntimeOptions | None = None) -> None:
            pass

        def complete_sync(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = messages[0]["content"]
            return '[{"reviewSourceId":"source-parking","reviewEventId":"event-parking","hit":0,"result":"未发现明确违规停车证据"}]'

    monkeypatch.setattr(workflow, "OpenAICompatibleClient", FakeClient)
    selection = workflow.ReviewSkillSelection(
        skill_name="parking-violation-review",
        method="single_candidate",
        confidence=1.0,
        reason="test",
        candidates=(),
        selected_context=(
            "skill_name: parking-violation-review\n"
            "如果车辆停在正常车位、停车线内、划定停车区域内，应判定为不命中。"
        ),
    )

    reply = workflow.ParkingAbnormalReviewWorkflow._complete_review(
        agent_config=AgentConfig(name="default", display_name="Default"),
        runtime_options=RuntimeOptions(),
        prompt_text=(
            '{"reviewSourceId":"source-parking",'
            '"taskTarget":{"value":"ParkingViolationDetection","text":"车辆违停检测"},'
            '"schemaResults":{"alarmBox":[10,20,80,90],"area":[[0,0],[100,0],[100,100],[0,100]]}}'
        ),
        image_attachments=[Attachment(name="parking.jpg", mime_type="image/jpeg")],
        paths=None,
        review_source_id="source-parking",
        objective="ParkingViolationDetection",
        visual_regions=[
            {
                "source": "prompt_json",
                "regions": {
                    "alarm_box": [10, 20, 80, 90],
                    "area": [[0, 0], [100, 0], [100, 100], [0, 100]],
                },
            }
        ],
        skill_selection=selection,
        skill_result=SkillRunResult(skill_name="parking-violation-review", data={"review_decision": "needs_model"}),
        skill_error="",
    )

    assert '"hit":0' in reply
    assert "车辆违停专项硬规则" in captured["user_prompt"]
    assert "正常停车位、停车线内、划定停车区域内，必须判定 hit=0" in captured["user_prompt"]
    assert "告警框、检测框、bbox 通常只是车辆检测框" in captured["user_prompt"]
    assert "ROI/area 可能只是算法识别范围" in captured["user_prompt"]
    assert "不能把框内有车当作违停证据" in captured["user_prompt"]
    assert "未发现明确违规停车证据" in captured["user_prompt"]
