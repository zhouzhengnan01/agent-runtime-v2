from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.schemas import Attachment


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
