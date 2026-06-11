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


def test_effective_video_candidate_count_prefers_small_sampling_budget() -> None:
    workflow = _load_workflow_module()

    assert workflow._effective_video_candidate_count(max_frames=6, max_candidates=24) == 12
    assert workflow._effective_video_candidate_count(max_frames=2, max_candidates=24) == 4
    assert workflow._effective_video_candidate_count(max_frames=20, max_candidates=10) == 10


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


def test_read_target_video_frame_candidates_scans_short_video_sequentially() -> None:
    workflow = _load_workflow_module()

    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0
            self.grab_calls = 0
            self.read_calls = 0

        def grab(self) -> bool:
            self.grab_calls += 1
            self.index += 1
            return True

        def read(self) -> tuple[bool, float]:
            self.read_calls += 1
            frame = float(self.index)
            return True, frame

        def get(self, prop: Any) -> float:
            return 0.0

    class FakeCv2:
        CAP_PROP_POS_MSEC = object()

    fake_capture = FakeCapture()
    requested: list[int] = []

    def fake_candidate(_: Any, frame: Any, *, frame_index: int, timestamp_ms: float) -> Any:
        requested.append(frame_index)
        return workflow.VideoFrameCandidate(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            sharpness=frame,
            brightness=frame,
            score=frame,
            thumbnail=frame,
            frame=frame,
        )

    workflow._video_frame_candidate = fake_candidate  # type: ignore[attr-defined]

    candidates = workflow._read_target_video_frame_candidates(
        FakeCv2(),
        fake_capture,
        fps=10.0,
        frame_count=100,
        max_candidates=5,
        deadline=10**9,
    )

    assert requested == [0, 25, 50, 74, 99]
    assert [candidate.frame_index for candidate in candidates] == requested
    assert fake_capture.grab_calls == 95
    assert fake_capture.read_calls == 5


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

    def fake_extract_video_frames(
        video_path: Path,
        output_dir: Path,
        *,
        prefix: str,
        timeout_seconds: int,
    ) -> list[Any]:
        assert video_path == video
        assert output_dir == paths.workspace / workflow.VIDEO_FRAME_OUTPUT_DIR
        assert prefix.startswith("video_1_clip_")
        assert timeout_seconds == workflow.VIDEO_FRAME_TIMEOUT_SECONDS
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

    def missing_opencv(
        video_path: Path,
        output_dir: Path,
        *,
        prefix: str,
        timeout_seconds: int,
    ) -> list[Any]:
        raise RuntimeError("OpenCV is unavailable; install opencv-python to enable video frame extraction.")

    monkeypatch.setattr(workflow, "_extract_video_frames", missing_opencv)

    attachments, reports = workflow._video_frame_attachments(
        [Attachment(name="clip.mp4", path="/mnt/user-data/uploads/clip.mp4", mime_type="video/mp4")],
        paths,
    )

    assert attachments == []
    assert reports[0]["status"] == "failed"
    assert reports[0]["error_code"] == "opencv_unavailable"


def test_video_frame_attachments_skip_when_image_inputs_are_enough(tmp_path: Path, monkeypatch: Any) -> None:
    workflow = _load_workflow_module()
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = artifact_store.prepare_thread("thread-video-skip")
    image_attachments = [
        Attachment(name=f"image-{index}.jpg", path=f"/mnt/user-data/uploads/image-{index}.jpg", mime_type="image/jpeg")
        for index in range(workflow.VIDEO_FRAME_SKIP_WHEN_IMAGE_COUNT_AT_LEAST)
    ]
    video_attachments = [
        Attachment(name="clip.mp4", path="/mnt/user-data/uploads/clip.mp4", mime_type="video/mp4")
    ]

    def fail_extract_video_frames(*_: Any, **__: Any) -> list[Any]:
        raise AssertionError("video extraction should be skipped when image inputs are enough")

    monkeypatch.setattr(workflow, "_extract_video_frames", fail_extract_video_frames)

    attachments, reports = workflow._video_frame_attachments_for_review(image_attachments, video_attachments, paths)

    assert attachments == []
    assert workflow._review_error_summary("", reports) == []
    assert reports == [
        {
            "name": "clip.mp4",
            "mime_type": "video/mp4",
            "path": "/mnt/user-data/uploads/clip.mp4",
            "status": "skipped",
            "error_code": "image_attachments_preferred",
            "image_attachment_count": workflow.VIDEO_FRAME_SKIP_WHEN_IMAGE_COUNT_AT_LEAST,
            "threshold": workflow.VIDEO_FRAME_SKIP_WHEN_IMAGE_COUNT_AT_LEAST,
        }
    ]


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
    by_name = {candidate.skill.name: candidate for candidate in candidates}
    assert candidates[0].skill.name == "fall-review"
    assert by_name["fall-review"].score > by_name["smoking-review"].score
    assert "keyword=人员跌倒/倒地检测" in by_name["fall-review"].score_reasons


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
    by_name = {candidate.skill.name: candidate for candidate in candidates}
    assert candidates[0].skill.name == "fall-review"
    assert by_name["fall-review"].score > by_name["smoking-review"].score


def test_sanitize_review_prompt_text_removes_summary_sections_and_fields() -> None:
    workflow = _load_workflow_module()
    text_prompt = """
    场景
    车辆违停检测
    告警摘要
    画面中央存在白色杂物堆积，符合杂物检测命中特征。
    置信度
    car
    91%
    """
    json_prompt = (
        '{"taskTarget":{"value":"ParkingViolationDetection","text":"车辆违停检测"},'
        '"summary":"画面中央存在白色杂物堆积","result":"符合杂物检测命中特征",'
        '"schemaResults":{"alarmBox":[1,2,3,4]}}'
    )

    sanitized_text = workflow._sanitize_review_prompt_text(text_prompt)
    sanitized_json = workflow._sanitize_review_prompt_text(json_prompt)

    assert "白色杂物堆积" not in sanitized_text
    assert "告警摘要" not in sanitized_text
    assert "车辆违停检测" in sanitized_text
    assert "置信度" in sanitized_text
    assert '"summary"' not in sanitized_json
    assert '"result"' not in sanitized_json
    assert '"taskTarget"' in sanitized_json
    assert '"schemaResults"' in sanitized_json


def test_review_objective_uses_edge_detection_type_format() -> None:
    workflow = _load_workflow_module()
    prompt_text = (
        "当前复判事件来源reviewSourceId为[source-1]，"
        "检测类型为[IllegalParkingDetection/车辆违停]。"
        "这是边端真实历史车辆违停高分视频。"
    )

    assert workflow._objective(prompt_text) == "ParkingViolationDetection"


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


def test_kitchen_hygiene_task_name_routes_to_clutter_review_skill() -> None:
    workflow = _load_workflow_module()
    prompt_text = """
    视频接入名称
    温州印象城网关·D29-L5-21后门
    任务名称
    后厨通道卫生安全监管
    模型名称
    后厨通道卫生安全监管
    来源
    device
    """

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "17803963378248hh02dvt": SkillDefinition(
                    name="17803963378248hh02dvt",
                    description="用于机器视觉复判杂物检测识别目标的情况。",
                    output_kind="markdown",
                    routing={
                        "keywords": [
                            "杂物检测",
                            "杂物",
                            "后厨通道卫生",
                            "通道卫生",
                            "乱堆乱放",
                            "ClutterDetection",
                        ]
                    },
                ),
                "garbage-overflow-review": SkillDefinition(
                    name="garbage-overflow-review",
                    description="用于机器视觉复判垃圾桶满溢、垃圾外溢、垃圾堆积等后厨通道卫生事件。",
                    output_kind="markdown",
                    routing={"keywords": ["垃圾满溢", "垃圾外溢", "垃圾堆积", "GarbageOverflowDetection"]},
                ),
                "smoking-review": SkillDefinition(
                    name="smoking-review",
                    description="抽烟 SmokingDetection 复判",
                    output_kind="markdown",
                    routing={"keywords": ["抽烟", "SmokingDetection"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

    objective = workflow._objective(prompt_text)
    candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["17803963378248hh02dvt", "garbage-overflow-review", "smoking-review"],
        objective=objective,
        prompt_text=prompt_text,
    )

    assert objective == "ClutterDetection"
    assert candidates[0].skill.name == "17803963378248hh02dvt"
    assert candidates[0].score > candidates[1].score
    assert "smoking-review" not in {candidate.skill.name for candidate in candidates}


def test_kitchen_hygiene_result_text_does_not_enable_smoking_review_skill() -> None:
    workflow = _load_workflow_module()
    prompt_text = """
    视频接入名称
    温州印象城网关·D29-L5-19后门
    任务名称
    后厨通道卫生安全监管
    模型名称
    后厨通道卫生安全监管
    来源
    device
    识别结果
    画面中可见人员手中持有细长条状物体，但经观察并非香烟，未见烟雾或典型吸烟动作。
    """

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "17803963378248hh02dvt": SkillDefinition(
                    name="17803963378248hh02dvt",
                    description="用于机器视觉复判杂物检测识别目标的情况。",
                    output_kind="markdown",
                    routing={"keywords": ["杂物检测", "后厨通道卫生", "ClutterDetection"]},
                ),
                "smoking-review": SkillDefinition(
                    name="smoking-review",
                    description="用于机器视觉复判抽烟、吸烟、持烟等行为事件。",
                    output_kind="markdown",
                    routing={"keywords": ["抽烟", "吸烟", "香烟", "SmokingDetection"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

    candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["17803963378248hh02dvt", "smoking-review"],
        objective=workflow._objective(prompt_text),
        prompt_text=prompt_text,
    )

    assert workflow._objective(prompt_text) == "ClutterDetection"
    assert [candidate.skill.name for candidate in candidates] == ["17803963378248hh02dvt"]


def test_kitchen_garbage_overflow_model_overrides_hygiene_category_and_person_summary() -> None:
    workflow = _load_workflow_module()
    prompt_text = """
    任务名称
    后厨通道卫生安全监管
    模型名称
    垃圾满溢检测
    场景
    垃圾满溢检测
    告警摘要
    检测框区域（bbox: 210, 123, 326, 407）内清晰可见一名身穿黑色长裤和白色鞋子的人员正在走廊行走。
    """

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "17803963378248hh02dvt": SkillDefinition(
                    name="17803963378248hh02dvt",
                    description="用于机器视觉复判杂物检测识别目标的情况。",
                    output_kind="markdown",
                    routing={"keywords": ["杂物检测", "后厨通道卫生", "ClutterDetection"]},
                ),
                "garbage-overflow-review": SkillDefinition(
                    name="garbage-overflow-review",
                    description="用于机器视觉复判垃圾桶满溢、垃圾外溢、垃圾堆积等后厨通道卫生事件。",
                    output_kind="markdown",
                    routing={"keywords": ["垃圾满溢检测", "垃圾满溢", "垃圾桶满溢", "GarbageOverflowDetection"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

    objective = workflow._objective(prompt_text)
    candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["17803963378248hh02dvt", "garbage-overflow-review"],
        objective=objective,
        prompt_text=prompt_text,
    )

    assert objective == "GarbageOverflowDetection"
    assert candidates[0].skill.name == "garbage-overflow-review"


def test_review_objective_uses_scene_template_names_instead_of_unknown() -> None:
    workflow = _load_workflow_module()

    examples = {
        "任务名称\n顾客行为检测\n模型名称\n顾客行为监管": "CustomerBehaviorDetection",
        "任务名称\n消防通道监管\n模型名称\n消防通道监管": "FireLaneComplianceDetection",
        "任务名称\n车场异常监控\n模型名称\n车场异常监控": "ParkingAbnormalDetection",
        "任务名称\n车场异常事件监管\n模型名称\n--": "ParkingAbnormalDetection",
        "任务名称\n后厨通道卫生安全监管\n模型名称\n后厨通道卫生安全监管": "ClutterDetection",
    }

    for prompt_text, expected_objective in examples.items():
        assert workflow._objective(prompt_text) == expected_objective


def test_scene_template_name_keeps_relevant_review_skill_candidates() -> None:
    workflow = _load_workflow_module()

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "smoking-review": SkillDefinition(
                    name="smoking-review",
                    description="抽烟 SmokingDetection 复判",
                    output_kind="markdown",
                    routing={"keywords": ["抽烟", "SmokingDetection"]},
                ),
                "fall-review": SkillDefinition(
                    name="fall-review",
                    description="跌倒 FallDetection 复判",
                    output_kind="markdown",
                    routing={"keywords": ["跌倒", "FallDetection"]},
                ),
                "fight-review": SkillDefinition(
                    name="fight-review",
                    description="争吵 打架 FightDetection 复判",
                    output_kind="markdown",
                    routing={"keywords": ["争吵", "打架", "FightDetection"]},
                ),
                "fire-lane-compliance-review": SkillDefinition(
                    name="fire-lane-compliance-review",
                    description="消防通道合规检测 消防通道监管 FireLaneComplianceDetection",
                    output_kind="markdown",
                    routing={"keywords": ["消防通道合规检测", "消防通道监管", "FireLaneComplianceDetection"]},
                ),
                "parking-violation-review": SkillDefinition(
                    name="parking-violation-review",
                    description="车辆违停 违规停车 ParkingViolationDetection",
                    output_kind="markdown",
                    routing={"keywords": ["车辆违停", "违规停车", "ParkingViolationDetection"]},
                ),
                "parking-congestion-review": SkillDefinition(
                    name="parking-congestion-review",
                    description="停车场通道拥堵检测 车辆拥堵 ParkingCongestionDetection",
                    output_kind="markdown",
                    routing={"keywords": ["停车场通道拥堵检测", "车辆拥堵", "ParkingCongestionDetection"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

    fire_candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["fire-lane-compliance-review", "smoking-review"],
        objective=workflow._objective("任务名称\n消防通道监管\n模型名称\n消防通道监管"),
        prompt_text="任务名称\n消防通道监管\n模型名称\n消防通道监管",
    )
    parking_candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["parking-violation-review", "parking-congestion-review"],
        objective=workflow._objective("任务名称\n车场异常监控\n模型名称\n车场异常监控"),
        prompt_text="任务名称\n车场异常监控\n模型名称\n车场异常监控",
    )
    behavior_candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["smoking-review", "fall-review", "fight-review"],
        objective=workflow._objective("任务名称\n顾客行为检测\n模型名称\n顾客行为监管"),
        prompt_text="任务名称\n顾客行为检测\n模型名称\n顾客行为监管",
    )

    assert fire_candidates[0].skill.name == "fire-lane-compliance-review"
    assert {candidate.skill.name for candidate in parking_candidates} == {
        "parking-violation-review",
        "parking-congestion-review",
    }
    assert {candidate.skill.name for candidate in behavior_candidates} == {
        "smoking-review",
        "fall-review",
        "fight-review",
    }


def test_review_skill_candidates_do_not_auto_expand_when_selected_skills_exist() -> None:
    workflow = _load_workflow_module()

    class FakeRegistry:
        def __init__(self) -> None:
            self.skills = {
                "parking-violation-review": SkillDefinition(
                    name="parking-violation-review",
                    description="车辆违停 违规停车 ParkingViolationDetection",
                    output_kind="markdown",
                    routing={"keywords": ["违规停车", "车辆违停", "车辆违停检测", "ParkingViolationDetection"]},
                ),
                "17803963378248hh02dvt": SkillDefinition(
                    name="17803963378248hh02dvt",
                    description="杂物检测 ClutterDetection",
                    output_kind="markdown",
                    routing={"keywords": ["杂物", "杂物检测", "ClutterDetection"]},
                ),
            }

        def get(self, name: str) -> SkillDefinition:
            return self.skills[name]

        def list(self, executable_only: bool = True) -> list[SkillDefinition]:
            return list(self.skills.values())

    prompt_text = (
        '{"taskTarget":{"value":"ParkingViolationDetection","text":"车辆违停检测"},'
        '"modelName":"车场异常事件监控","summary":"画面中央存在白色杂物堆积"}'
    )

    candidates = workflow._review_skill_candidates(
        FakeRegistry(),
        ["parking-violation-review"],
        objective="ParkingViolationDetection",
        prompt_text=prompt_text,
    )

    assert [candidate.skill.name for candidate in candidates] == ["parking-violation-review"]


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
    assert "parking review final result review_source_id=source-log" in logs
    assert "模型判定命中" in logs
    assert "result_chars=" in logs
    assert "runtime event type=review.normalized_result" not in logs
    assert "parking review llm raw reply" not in logs
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
    monkeypatch.setattr(workflow, "PARKING_REVIEW_LOG_INPUT_SUMMARY", True)

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
    assert "parking review input summary review_source_id=source-image" in logs
    assert "link_count=1" in logs
    assert "remote_link_count=1" in logs
    assert "https://example.test/api/ai/task/history/_read/image.jpg?..." in logs
    assert '"dataId":"data-1"' in logs
    assert "parking review final result review_source_id=source-image" in logs
    event_payloads = {event.type: event.data for event in events}
    assert event_payloads["review.input"]["image_sources"][0]["url"] == image_url
    assert event_payloads["review.input"]["image_sources"][0]["dataId"] == "data-1"


def test_parking_review_logs_and_passes_visual_regions(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    workflow = _load_workflow_module()
    monkeypatch.setattr(workflow, "PARKING_REVIEW_LOG_INPUT_SUMMARY", True)
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
    assert "parking review input summary review_source_id=source-region" in logs
    assert "region_count=2" in logs
    assert "parking review final result review_source_id=source-region" in logs
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


def test_parking_violation_review_prompt_strips_clutter_summary_and_marks_it_as_history(monkeypatch: Any) -> None:
    workflow = _load_workflow_module()
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, agent_config: AgentConfig, runtime_options: RuntimeOptions | None = None) -> None:
            pass

        def complete_sync(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
            captured["user_prompt"] = messages[0]["content"]
            return '[{"reviewSourceId":"source-history","reviewEventId":"event-history","hit":0,"result":"未发现明确违规停车证据"}]'

    monkeypatch.setattr(workflow, "OpenAICompatibleClient", FakeClient)
    selection = workflow.ReviewSkillSelection(
        skill_name="parking-violation-review",
        method="single_candidate",
        confidence=1.0,
        reason="test",
        candidates=(),
        selected_context="skill_name: parking-violation-review\n仅判断车辆违停。",
    )

    workflow.ParkingAbnormalReviewWorkflow._complete_review(
        agent_config=AgentConfig(name="default", display_name="Default"),
        runtime_options=RuntimeOptions(),
        prompt_text=(
            "场景\n"
            "车辆违停检测\n"
            "告警摘要\n"
            "画面中央行车通道地面上存在明显的白色杂物堆积，形状类似废弃的塑料托盘或框架。\n"
            "置信度\n"
            "car\n"
            "91%"
        ),
        image_attachments=[Attachment(name="parking.jpg", mime_type="image/jpeg")],
        paths=None,
        review_source_id="source-history",
        objective="ParkingViolationDetection",
        visual_regions=[],
        skill_selection=selection,
        skill_result=SkillRunResult(skill_name="parking-violation-review", data={"review_decision": "needs_model"}),
        skill_error="",
    )

    assert "白色杂物堆积" not in captured["user_prompt"]
    assert "塑料托盘" not in captured["user_prompt"]
    assert "历史摘要或旧结果" in captured["user_prompt"]
    assert "不可作为本次命中的判定依据" in captured["user_prompt"]


def test_parking_review_caps_llm_timeout(monkeypatch: Any) -> None:
    workflow = _load_workflow_module()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(workflow, "REVIEW_LLM_TIMEOUT_SECONDS", 60)

    class FakeClient:
        def __init__(self, agent_config: AgentConfig, runtime_options: RuntimeOptions | None = None) -> None:
            captured["timeout"] = runtime_options.request_timeout_seconds if runtime_options is not None else None

        def complete_sync(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
            return '[{"reviewSourceId":"source-timeout","reviewEventId":"event-timeout","hit":0,"result":"未命中"}]'

    monkeypatch.setattr(workflow, "OpenAICompatibleClient", FakeClient)

    workflow.ParkingAbnormalReviewWorkflow._complete_review(
        agent_config=AgentConfig(name="default", display_name="Default"),
        runtime_options=RuntimeOptions(request_timeout_seconds=180),
        prompt_text="当前复判事件来源reviewSourceId为[source-timeout]",
        image_attachments=[Attachment(name="image.jpg", mime_type="image/jpeg")],
        paths=None,
        review_source_id="source-timeout",
        objective="ClutterDetection",
        visual_regions=[],
        skill_selection=None,
        skill_result=None,
        skill_error="",
    )

    assert captured["timeout"] == 60


def test_parking_review_returns_conservative_json_when_llm_fails(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    workflow = _load_workflow_module()

    def failing_complete_review(**kwargs: Any) -> str:
        raise TimeoutError("upstream timed out")

    monkeypatch.setattr(workflow.ParkingAbnormalReviewWorkflow, "_complete_review", staticmethod(failing_complete_review))
    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("thread-llm-failed")
    image = paths.uploads / "image.jpg"
    image.write_bytes(b"fake-image")
    review_workflow = workflow.ParkingAbnormalReviewWorkflow(store)

    with caplog.at_level("INFO", logger="uvicorn.error"):
        result, events = review_workflow.run_with_events(
            AgentConfig(name="default", display_name="Default"),
            [Message(role="user", content="当前复判事件来源reviewSourceId为[source-failed]。\n本次复判的识别目标为[ClutterDetection]")],
            [Attachment(name="image.jpg", path="/mnt/user-data/uploads/image.jpg", mime_type="image/jpeg")],
            "thread-llm-failed",
            runtime_options=RuntimeOptions(),
        )

    assert '"reviewSourceId":"source-failed"' in result.reply
    assert '"hit":0' in result.reply
    assert "模型请求失败或超时" in result.reply
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "parking review failed review_source_id=source-failed" in logs
    assert "reason=模型请求失败或超时，已按证据不足处理。" in logs
    assert "error=upstream timed out" in logs
    event_payloads = {event.type: event.data for event in events}
    assert event_payloads["review.llm.failed"]["error"] == "upstream timed out"
    assert event_payloads["review.normalized_result"]["result"] == result.reply


def test_parking_review_logs_routing_details(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    workflow = _load_workflow_module()

    class FakeRegistry:
        def get(self, name: str) -> SkillDefinition:
            if name == "parking-violation-review":
                return SkillDefinition(
                    name="parking-violation-review",
                    description="车辆违停 违规停车 ParkingViolationDetection",
                    output_kind="markdown",
                    routing={"keywords": ["违规停车", "车辆违停", "车辆违停检测", "ParkingViolationDetection"]},
                )
            raise KeyError(name)

    class FakeRunner:
        def run(self, skill_name: str, spec: dict[str, Any], paths: Any, on_event: Any = None) -> SkillRunResult:
            return SkillRunResult(skill_name=skill_name, data={"review_decision": "needs_model"})

    def fake_complete_review(**kwargs: Any) -> str:
        return '[{"reviewSourceId":"source-routing","reviewEventId":"event-routing","hit":0,"result":"未命中"}]'

    monkeypatch.setattr(workflow.ParkingAbnormalReviewWorkflow, "_complete_review", staticmethod(fake_complete_review))
    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("thread-routing")
    image = paths.uploads / "image.jpg"
    image.write_bytes(b"fake-image")
    review_workflow = workflow.ParkingAbnormalReviewWorkflow(
        store,
        skill_registry=FakeRegistry(),
        skill_runner=FakeRunner(),
    )

    with caplog.at_level("INFO", logger="uvicorn.error"):
        result, events = review_workflow.run_with_events(
            AgentConfig(name="default", display_name="Default"),
            [
                Message(
                    role="user",
                    content=(
                        '{"reviewSourceId":"source-routing",'
                        '"taskTarget":{"value":"ParkingViolationDetection","text":"车辆违停检测"},'
                        '"summary":"画面中央存在白色杂物堆积"}'
                    ),
                )
            ],
            [Attachment(name="image.jpg", path="/mnt/user-data/uploads/image.jpg", mime_type="image/jpeg")],
            "thread-routing",
            runtime_options=RuntimeOptions(
                app_template_name="ParkingAbnormalEventMonitoring",
                selected_skills=["parking-violation-review"],
            ),
        )

    assert result.reply == '[{"reviewSourceId":"source-routing","reviewEventId":"event-routing","hit":0,"result":"未命中"}]'
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "parking review routing review_source_id=source-routing" in logs
    assert "app_template=ParkingAbnormalEventMonitoring" in logs
    assert "objective=ParkingViolationDetection" in logs
    assert "prompt_sanitized=True" in logs
    event_payloads = {event.type: event.data for event in events}
    assert event_payloads["review.input"]["app_template_name"] == "ParkingAbnormalEventMonitoring"
    assert event_payloads["review.input"]["configured_selected_skills"] == ["parking-violation-review"]
    assert event_payloads["review.input"]["prompt_sanitized"] is True


def test_parking_review_skill_router_skips_llm_by_default(monkeypatch: Any) -> None:
    workflow = _load_workflow_module()
    monkeypatch.setattr(workflow, "REVIEW_ENABLE_LLM_SKILL_ROUTER", False)

    def unexpected_llm_router(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("llm skill router should be disabled by default")

    monkeypatch.setattr(workflow, "_llm_select_review_skill", unexpected_llm_router)

    class FakeRegistry:
        def list(self, executable_only: bool = False) -> list[SkillDefinition]:
            return []

        def get(self, name: str) -> SkillDefinition:
            if name == "smoking-review":
                return SkillDefinition(
                    name="smoking-review",
                    description="抽烟 吸烟 smoking",
                    output_kind="markdown",
                    routing={"keywords": ["抽烟", "吸烟", "smoking"]},
                )
            if name == "fall-review":
                return SkillDefinition(
                    name="fall-review",
                    description="跌倒 摔倒 fall",
                    output_kind="markdown",
                    routing={"keywords": ["跌倒", "摔倒", "fall"]},
                )
            raise KeyError(name)

    review_workflow = workflow.ParkingAbnormalReviewWorkflow(
        ArtifactStore(),
        skill_registry=FakeRegistry(),
    )

    selection = review_workflow._select_review_skill(
        AgentConfig(name="default", display_name="Default"),
        RuntimeOptions(selected_skills=["smoking-review", "fall-review"]),
        prompt_text="本次复判的识别目标为[SmokingDetection]",
        review_source_id="source-router",
        objective="SmokingDetection",
    )

    assert selection is not None
    assert selection.skill_name == "smoking-review"
    assert selection.method in {"keyword_score", "score_fallback"}
